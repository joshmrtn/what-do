# Design Decisions

Rationale for non-obvious choices made during implementation. In rough chronological order.

---

## Fuzzy matching library: rapidfuzz over thefuzz

**Decision:** Use `rapidfuzz` for all fuzzy string matching.

**Rationale:** rapidfuzz is significantly faster than thefuzz (C extension vs Python), has a
compatible API, and is actively maintained. thefuzz is effectively a slower wrapper around the
same Levenshtein logic. No downside to preferring rapidfuzz.

**Decoupling:** All fuzzy match calls go through `_fuzzy_match(a, b) → float` in
`venue_discovery.py`. If we ever swap libraries (e.g. to embedding+cosine similarity for
richer semantic matching), only that function changes — callers are unaffected.

---

## Venue dedup: name + address, separate thresholds

**Decision:** Two independent fuzzy thresholds — `name_match_threshold` (default 0.92) and
`address_match_threshold` (default 0.85) — both configurable in `config.yaml`.

**Rationale:** A venue chain (e.g. "Holy Cow Ice Cream") can appear at multiple distinct
addresses. Matching on name alone would incorrectly merge them. Requiring both name AND address
to match prevents false merges. Separate thresholds let us tune each dimension independently —
addresses need slightly more tolerance for abbreviations ("St" vs "Street") than names do.

**Bias toward false negatives:** We set thresholds high. A duplicate is a minor annoyance; a
missed unique venue is a lost recommendation.

---

## Blocklist threshold: separate, lower than dedup

**Decision:** `blocklist_name_match_threshold` (default 0.80) is a separate config key from
the venue dedup thresholds.

**Rationale:** For dedup we prefer to keep two records over incorrectly merging them. For the
blocklist, erring slightly toward exclusion is acceptable but we don't want to accidentally
block venues with similar names. 0.80 catches clear matches like `"O'Neil's Bar"` vs `"O'Neils Bar"`
without catching unrelated venues that share a word. Handles (`@...`) in the blocklist are
always matched exactly, regardless of threshold.

---

## Geocoding: GeocoderProvider ABC + Nominatim default

**Decision:** Geocoding is behind a `GeocoderProvider` ABC with `NominatimGeocoder` as the
default implementation.

**Rationale:** Nominatim (OpenStreetMap) is free with no API key and adequate for geocoding a
handful of seed venues per run. The ABC makes it trivially swappable for OpenCage, Positionstack,
or Google if Nominatim proves insufficient. Geocoding failure (None return or exception) is
non-fatal: the venue is stored with null coordinates and a warning is logged.

---

## Venue providers: VenueSource ABC

**Decision:** Geographic venue providers (Overpass API, Google Places, Foursquare) all
implement the `VenueSource` ABC: `fetch_venues(lat, lng, radius_miles, categories) → List[Venue]`.

**Rationale:** Provider independence is a first-class requirement. The ABC means adding or
removing a provider touches zero code outside the new adapter. The discovery service receives
a list of sources at construction time — no provider is hardcoded. Foursquare is a candidate
for implementation; its free tier was confirmed to exist but pricing/limits need verification
before committing to it.

---

## Radius filtering: provider responsibility + service defense-in-depth

**Decision:** Providers are responsible for returning only in-radius venues (they receive the
radius parameter). The discovery service also applies a haversine distance check as
defense-in-depth before persisting.

**Rationale:** Provider APIs may have slightly different radius semantics. The secondary check
in the service ensures correctness regardless of how the provider interprets the radius.

---

## Seed venues: always persisted (bypass radius check)

**Decision:** Seed venues from `seeds.yaml` are always persisted, even if their geocoded
coordinates are outside the configured radius.

**Rationale:** The user explicitly listed the venue in seeds.yaml — this is intentional. The
radius check is for auto-discovered venues. Seed venues with failed geocoding get stored with
null coordinates rather than being discarded.

---

## Seed handles → candidate_entities as active, not probationary

**Decision:** Handles in `seeds.yaml` (e.g. `@cinemasalem`) are written to `candidate_entities`
as `active`, not `probationary`. Seed venue entries (name + address) go directly to the
`venues` table (unchanged from Phase 2).

**Rationale:** A handle is not a venue — it's a social account. But seed handles are explicitly
trusted by the user; they should be scraped immediately without requiring promotion. Only handles
*discovered during scraping* start as `probationary`. Storing seeds as `probationary` would
break bootstrapping — nothing would ever get scraped on first run.

*Correction from earlier design:* an earlier entry stated seed handles were stored as
`probationary`. That was incorrect and has been superseded by this entry.

---

## Venue categories: user-configurable in config.yaml

**Decision:** The list of venue category slugs to search for lives in `config.yaml` under
`venue_discovery.categories`.

**Rationale:** Different users care about different types of venues. Making categories
configurable means no code changes to add or remove a category. Default list covers the common
cases (cafe, bar, restaurant, etc.) but users can freely extend or narrow it.

---

## Failover chain: generic runner, not provider-internal

**Decision:** The Apify → Picuki → Dumpor failover is implemented as a `FailoverChain` class
in the ingestion service layer. Adapters are registered in priority order; the runner tries each
in sequence and catches exceptions. Adapters themselves just raise on failure.

**Rationale:** Provider-internal failover would couple Apify to Picuki. The generic runner means
adapters are independently swappable — adding or removing one touches no other adapter and no
business logic. The runner is also independently testable.

---

## Handle disambiguation: dedicated batch step 3a

**Decision:** LLM-based handle classification (venue vs person) runs as step 3a in the batch
pipeline, after scraping (step 3) and before normalization (step 4). It is not part of the
ingestion layer.

**Rationale:** The ingestion layer spec explicitly prohibits invoking semantic models. Step 3a
is a named pipeline stage with its own `DisambiguationProvider` ABC, keeping ingestion LLM-free
while still classifying handles within the same batch run. The LLM used is `gemma4:e2b`
(lighter model — binary classification task).

---

## Trusted sources for handle promotion: seeds only (v1)

**Decision:** A handle is promoted from `probationary` to `active` only when its
`mention_sources` list contains at least one seed handle AND `mention_count ≥ threshold`.

**Rationale:** If any active handle counted as a trusted source, two low-quality discovered
handles could promote each other in a feedback loop. Seeds are user-curated, so seeds-only
is a conservative trust anchor that prevents runaway discovery. Expanding the trust set is a
post-v1 concern.

---

## `raw_published_at` on EventCandidate

**Decision:** `EventCandidate` has a `raw_published_at: datetime | None` field. Social media
adapters populate it with the post's publish timestamp. Movie adapters leave it `None`.
The lookback window filter applies to this field; `None` values bypass the filter.

**Rationale:** `discovered_at` is when we scraped the content — always "now." The lookback
window should filter on when the original post was published, not when we fetched it. Without
`raw_published_at`, a 60-day-old post scraped today would pass the lookback check incorrectly.
Movie showtimes have no post date and should always pass through.

---

## `depth` and `mention_sources` on candidate_entities

**Decision:** `candidate_entities` tracks two new fields:
- `depth: int` — hops from seed sources (seeds = 0; handles found in their posts = 1; etc.)
- `mention_sources: list[str]` — JSON array of source handles that mentioned this handle

**Rationale:** `depth` is required for `max_depth` enforcement. Without it we can't know how
many hops from a seed a given handle is. `mention_sources` is required for the promotion rule
(distinct trusted sources) — a bare `mention_count` integer can't distinguish one source
mentioning a handle ten times from ten sources each mentioning it once.

---

## Malformed record policy at ingestion time

**Decision:** At ingestion, discard a record only if title, description, *and* start_time are
all absent. Normalization (Phase 4) applies the stricter rule (discard if title and start_time
both absent).

**Rationale:** The ingestion layer should be permissive — partial records may still have useful
content. Phase 4 normalization is the appropriate place for stricter triage. Discarding too
aggressively at ingestion risks silently losing events that normalization could have recovered.

---

## Malformed record policy at normalization time

**Decision:** At normalization, discard a record if both `title` AND `start_time` are absent.
Records missing only one are retained and flagged in `metadata` (`missing_title: true` or
`missing_start_time: true`). Discards are logged with the candidate's source handle and reason.

**Rationale:** Normalization is the last point before canonical `Event` objects enter the
pipeline. A record with neither a title nor a time is unrecoverable — no enrichment or LLM
pass can manufacture those. A record with only one missing can still be surfaced to the user
with partial information. This is stricter than ingestion-time policy (see above) by design.

---

## `source_event_candidates` stores IDs, not full objects

**Decision:** `Event.source_event_candidates` is `list[str]` — a list of `EventCandidate.id`
values. The full `EventCandidate` data is not embedded in the `Event`.

**Rationale:** Full candidates are already persisted in the `event_candidates` table. Carrying
them inside `Event` would duplicate data and bloat in-memory representations during dedup and
enrichment. Attribution is preserved via IDs; any downstream code that needs the original
candidate data can look it up by ID.

---

## Dedup Pass 1: None-field symmetry rule

**Decision:** When comparing two candidates for duplication, each criterion (title, venue,
start_time) follows a symmetric None rule:

- **Both values None** → criterion **passes** (nothing to distinguish them on this axis; the
  other criteria still gate the final decision).
- **Exactly one value None** → criterion **fails** (asymmetric data means we can't confirm
  they're the same event).
- **Both values present** → normal comparison (fuzzy ratio ≥ threshold for title; exact
  canonical match for venue; abs(Δ) ≤ window for start_time).

**Rationale:** Two events with identical venue and overlapping times but no titles are more
likely the same event than not. Treating both-None as a mismatch would silently prevent dedup
for any pair of incomplete records. The other criteria still act as guards, so the risk of
false-positive merges is low. The one-sided-None case is safer to reject — we have no way
to compare a known title against an absent one.

---

## Dedup Pass 1: venue matching is exact on canonical form

**Decision:** The venue criterion in dedup uses exact string equality on the canonicalized
venue name, not a fuzzy match.

**Rationale:** Title already carries the fuzzy comparison; adding fuzziness to venue too
increases false-positive merge risk. Canonicalization (see below) handles the common surface
variations (casing, article position) before the comparison, so exact equality is sufficient
in practice. If the same physical venue appears under genuinely different spellings from two
sources, the title + time criteria alone should still be enough signal.

---

## Venue name canonicalization

**Decision:** Canonical venue name = title-case with leading English article ("The", "A", "An")
moved to the front. Implementation:
1. Strip and collapse whitespace.
2. If name ends with `, The` / `, A` / `, An` (case-insensitive) → move article to front.
3. Title-case the result.

So `"vault, the"` → `"The Vault"`, `"VAULT, THE"` → `"The Vault"`.

**Rationale:** Social media sources and cinema APIs write venue names inconsistently. This
normalization is deterministic, reversible, and covers the two most common variants (prefix
article vs suffix article). We apply it before dedup so "The Vault" and "vault, the" from
two different sources merge correctly.

---

## Naive datetime treatment at normalization

**Decision:** If a datetime from an `EventCandidate` is timezone-naive, assume it is already
in the local timezone (derived from config lat/lng). Attach the config timezone; do not
convert.

**Rationale:** Scraped event times are almost always expressed in local time — a venue
posting "Saturday 8pm" means local 8pm. Treating naive datetimes as UTC would shift times
by several hours for US timezones. The only times we'd want UTC treatment are from standardized
APIs that explicitly return UTC, but those would already be timezone-aware.

---

## NormalizationEngine and DeduplicationEngine are pure (no I/O)

**Decision:** `NormalizationEngine` and `DeduplicationEngine` are pure classes — no DB access,
no logging, no filesystem. `NormalizationService` orchestrates them and owns the I/O
(logging discards, persisting events). Same separation used in Phase 3 (ingestion adapters
are pure; `IngestionService` owns I/O).

**Rationale:** Pure engines are testable in a single function call with no fixtures.
The service layer is tested with a real SQLite tmp file, matching the existing pattern.
Mixing I/O into the engines would make unit tests require DB setup for every normalization
edge case.

---

## Known gap: discovery_context not populated by HandleExtractor

**RESOLVED (Phase 6):** All three fix steps below are implemented and tested.
`HandleExtractor._upsert()` now stores `text[:300]` as `discovery_context` on insert and
backfills it when previously NULL; `OllamaDisambiguationProvider` (gemma4:e2b) is implemented.
Retained here for historical context.

**Gap identified:** Phase 3 built `HandleExtractor` and `DisambiguationStep` but did not wire
them together on the context field. `HandleExtractor` stores discovered handles in
`candidate_entities` but does not populate the `discovery_context` column (the surrounding post
caption that mentioned the handle). `DisambiguationStep` reads `discovery_context` and passes it
to the provider — which will always receive an empty string until this is fixed.

**Impact:** Tests pass because `DisambiguationProvider` is mocked and ignores context. The real
`OllamaDisambiguationProvider` (not yet implemented) will receive no context, making it a blind
binary classifier with degraded accuracy.

**Fix:** In Phase 6, when `OllamaDisambiguationProvider` is implemented:
1. Update `HandleExtractor._upsert()` to accept and store the surrounding text snippet as
   `discovery_context` (truncated to a reasonable length, e.g. 300 chars).
2. Update `HandleExtractor.process()` to pass the source `text` through to `_upsert`.
3. Add a test that `discovery_context` is populated after extraction.

---

## Injectable logger on EnrichmentService

**Decision:** `EnrichmentService.__init__` accepts an optional `logger: StructuredLogger | None`
parameter that defaults to `get_logger("enrichment")`.

**Rationale:** The service logs weather and movie provider failures. Without an injectable logger,
tests that need to assert "error was logged" would have to intercept stdout or the stdlib logging
system — brittle and indirect. An injectable logger follows the same pattern established by
`NormalizationService` and keeps failure-path tests simple and explicit. The default value means
production callers that don't care about log capture pay no cost.

**How to apply:** Any service that logs errors on failure paths and has test coverage on those
paths should accept an injectable `StructuredLogger`. Create one internally as the default.

---

## EnrichmentService: run_date weather fetch not de-duped against event-date cache

**Decision:** In `EnrichmentService.enrich()`, the weather fetch for `run_date` (used for
synthetic activity generation) is made via a direct `_fetch_weather` call after the event loop,
rather than consulting the per-call in-memory cache that was built during event enrichment.

**Rationale:** The cost of the "extra" fetch is one DB read that hits the weather_cache table —
no additional provider call occurs, since the DB cache is always checked first. Fixing it cleanly
requires a sentinel-aware dict lookup to distinguish "not yet cached" from "cached as None", which
adds complexity for a negligible gain. If batch sizes or DB overhead ever become measurable,
revisit by extending the in-memory dict to cover the run_date fetch.

---

## Pluggable LLM backend: `ChatClient` protocol + `LLMError`, Gemini alongside Ollama

**Decision:** Extraction and disambiguation providers depend on a `ChatClient` structural protocol
(a `chat(model, messages, images)` method) and a shared `LLMError` base — not on `OllamaClient`
directly. `GeminiClient` (google-genai) is a drop-in second implementation. Ollama stays the
production default; Gemini is opt-in via `GEMINI_API_KEY`.

**Rationale:** The Phase 6 plan wrote the providers against `OllamaClient` concretely. The trigger
to generalize was the real-Ollama smoke test taking ~10 minutes and being killed by the shell
tool's timeout, which made end-to-end validation painful — a fast hosted model unblocks iteration.
Beyond that immediate need, a backend-agnostic seam gives the *processing* side of the pipeline
flexibility to swap or add LLMs (llama.cpp, Anthropic, DeepSeek) cheaply. It is a convenience and a
hedge, **not** a move away from local-first: production is still targeted at 100% local, and on the
same caption `gemma4:e4b` produced the same five tags, venue, and summary as Gemini flash. Any future backend just implements `ChatClient` and raises an
`LLMError` subclass; callers already `except LLMError`.

---

## Default Gemini model is the `gemini-flash-latest` alias, not a pinned version

**Decision:** The default `gemini_model` is `gemini-flash-latest`, a moving alias, rather than a
pinned version string. Override with `GEMINI_MODEL` when a fixed version is needed.

**Rationale:** The first pick, `gemini-2.0-flash`, was retired by Google *during this session* and
began returning HTTP 404. Pinning a version invites recurring breakage as models retire. The
Gemini tests assert structure (≥5 tags, correct date resolution), not exact wording, so a moving
target is acceptable; anyone needing reproducibility can pin via `GEMINI_MODEL`.

---

## Date grounding anchors on `get_now()` (seen date), not the post's published date

**Decision:** `ExtractionStage` injects `get_now()` (the daily-scrape "today") as the extraction
`reference_date`, so the model resolves relative dates like "this Saturday" against a concrete
anchor. It does **not** use the post's own published date, even though `EventCandidate.raw_published_at`
captures it for some sources.

**Rationale:** LLMs have no clock; without an anchor, models either hallucinate a date
(`gemma4:e4b` produced 2024-10-26 for "this Saturday") or return null (Gemini flash). Grounding on
"today" fixes the common case: most sources have no reliable post date, and a daily scrape makes
seen-date ≈ posted-date in practice. Propagating `raw_published_at` onto the `Event` model (more
accurate for backdated posts) was out of scope — tracked as issue #5.

---

## Tags carry a centrality weight

**Decision:** `Event.tags` and `ExtractionResult.tags` are `list[Tag]`, where `Tag` has `text` and
a `weight` in `[0.0, 1.0]` describing how central the tag is to what the event *is*. LLM Pass 1
assigns the weight (1.0 = defining feature, 0.5 = secondary, 0.1 = incidental context).

**Rationale:** the scoring problem that motivated this is "karaoke at a bar". A venue that is a bar
and hosts karaoke produces tags like `karaoke`, `bar`, `nightlife`, `cocktails`. With unweighted
tags, incidental venue attributes outnumber the one tag describing the activity the user actually
cares about, and a dislike on "bars" sinks an event the user likes. Weights let the activity
dominate the venue's ambient character.

Measured (2026-07-31, six events, both `gemma4:e4b` and `gemini-flash-latest`): weighting is the
difference between 4/5 and 5/5 directional accuracy on the local model, and it consistently widens
the score gap on the designed test case — two events with near-identical tag vocabulary but
inverted prominence (a punk band show at a bar vs a punk-themed dance party).

**Local models do this well.** `gemma4:e4b` produced a wider, better-judged weight spread than
Gemini flash — it ranked karaoke *third* at a pub advertising drinks and sports first, where Gemini
marked karaoke 1.0 at every venue that mentioned it. Production stays 100% local.

---

## Weights multiply the contribution; they are never averaging weights

**Decision:** a tag's contribution is `w × similarity`. The weight scales the magnitude *before*
aggregation. It must not be used as the weight in a weighted mean.

**Rationale:** this is subtle and got implemented wrong once during the spike, producing plausible
but wrong numbers. Using `w` as an averaging weight normalises it away: an event whose only negative
tags are incidental — `bar` at weight 0.20, cosine 0.932 — still yields a negative mean of ≈0.81,
because a weighted mean over one low-weight item is just that item. Down-weighting only re-blends a
tag against others on the same side; it never shrinks that side. With `c = w × s`, `bar` contributes
0.186 and genuinely recedes.

Measured effect: `gate` + multiplier semantics is the only combination scoring correctly on both
tagging models (6/6 Gemini, 5/5 gemma); the same rule with averaging semantics gets 4/5 on gemma.

---

## Scoring formula replaced after measurement: logistic gate + balanced mean

**Decision:** the formula specified in `high-level-design.md` §4.5 Stage 3 — `sum(contributions) /
len(tags)` over raw cosines — is replaced by:

```
gate(s)      = 1 / (1 + exp(-(s - gate_midpoint) / gate_temperature))   # default 0.60, 0.04
contribution = w × (+like_sim × gate(like_sim)  if like_sim > dislike_sim
                    else -dislike_sim × gate(dislike_sim))
tag_score    = mean(positive contributions) - mean(|negative contributions|)
```

**Rationale:** the original formula was measured against real `nomic-embed-text` vectors and the
user's real preference files, and it scored *every* candidate venue negative while ranking them
exactly backwards from the user's stated preference. Two independent defects:

1. **The cosine floor is ~0.42, not ~0.** Unrelated pairs measured `min 0.302, median 0.417,
   p75 0.472` across 198 tag×preference pairs. The specification's assumption that unrelated
   concepts score near zero is false for this embedding model.

2. **Noise decided both the sign and the magnitude.** `sushi` scored `like 0.407 (karaoke)` vs
   `dislike 0.433 (dancing)` — neither is a match — and the winner-takes-full-cosine rule turned a
   0.027 noise gap into a **-0.433** penalty. The user was penalised for the venue serving sushi.
   With ~6 tags per event, a few neutral tags swamp one genuine match.

Worse, the noise is **biased, not random**: over 11 neutral probe tags, mean best-dislike (0.461)
exceeded mean best-like (0.420) and the dislike side won **10 of 11**. Cause: this user's dislikes
are broad category words ("bars", "nightclubs"), the likes are narrow genre terms ("emo music").
Broad terms sit closer to all vocabulary. **The raw rule is biased toward whichever preference list
is written in more generic language** — not the longer one, the more generic one.

The logistic gate maps the entire 0.30–0.47 noise band to ≈0.01, so noise contributes nothing while
a near-miss at 0.59 still counts faintly. It neutralises the asymmetry as a side effect, with no
separate per-side calibration. The balanced mean stops a count of weak incidental negatives from
outvoting one strong positive.

**Rejected alternatives, both measured:** a hard similarity floor (0.50) performed *worse than doing
nothing* (4/6 vs 6/6); a margin rule (`like_sim - dislike_sim`) was worst of all, because the
vocabulary asymmetry **is** a margin offset and a margin-based rule inherits it wholesale.

**Caveat:** n=6 invented events plus three real venues, single runs per event. This justifies the
design; it is not a calibration. `gate_midpoint`, `gate_temperature`, and the aggregator are all
config keys precisely so they can be tuned against real batch output.

### How this was measured

Recorded so the numbers above can be checked and the experiment repeated when recalibrating.

**Embedding model:** `nomic-embed-text` via Ollama `/api/embed`, 768 dimensions, cosine similarity,
no task prefix on either side (tags and preferences are embedded identically so the comparison stays
symmetric).

**Preference input:** the local `data/likes.txt` and `data/dislikes.txt`. These are gitignored, so
their shape matters more than their content for reproduction: **5 general likes** (one activity, three
narrow music-genre terms, one atmosphere phrase), **1 movies-domain like**, and **4 general dislikes**,
all broad venue/scene category words. The vocabulary-asymmetry finding is a direct consequence of that
shape — narrow likes against broad dislikes — and would weaken or invert with differently-worded files.

**Corpus.** Two groups, both scored against expected directions derived from the preference files:

| group | items | provenance | what it tests |
|---|---|---|---|
| venues | Koto, The Castle, O'Neill's | Koto is a verbatim listing from the venue's website; the other two texts were **invented** | three karaoke-hosting bar-restaurants the user ranks Koto > Castle > O'Neill's — near-identical tags, subtle preference differences |
| events | A–F | all **invented** | A/B are the designed pair: near-identical tag vocabulary, inverted prominence (punk band show *at* a bar vs punk-themed *dance party*). C–F are controls at the extremes |

The A/B pair is the reusable idea: hold vocabulary constant and invert which tag is central. It is
the only construction that isolates centrality weighting from ordinary tag matching.

**Procedure.** For each event: tag it with an LLM (both `gemma4:e4b` and `gemini-flash-latest`, to
separate model-specific artifacts from real effects); embed each tag; take `max` cosine against the
like list and against the dislike list, restricted to the event's applicable domains; apply the
contribution rule; aggregate; compare the sign of the result to the expected direction. Directional
accuracy is the count of matching signs; separation is the smallest absolute score among correct
answers, which is what tier thresholds have to clear.

**Parameters swept:** contribution rules `raw`, `hard floor`, `logistic gate`, `margin`,
`margin×gate`; floors `0.00–0.65`; gate midpoint `0.60`, temperature `0.04`; aggregators
`sum/len(tags)` and `mean(pos) − mean(|neg|)`; weight modes none / averaging-weight /
magnitude-multiplier.

**Known limits.** Single run per event — LLM tagging is stochastic and that variance was never
measured. Two of the three venue texts and all six events were invented, so only Koto's text is real.
Domain scoping was initially omitted for the movie event, which invalidated its first result until
corrected. The scripts that produced this were session scratch files and are **not** in the repo, so
none of it is currently re-runnable — see the open question in the Phase 7 plan.

---

## Match classification uses a relative margin, not an absolute dislike threshold

**Decision:** an event is classified `no` when the strongest dislike similarity exceeds the
strongest like similarity by a configured margin — not when any dislike crosses a fixed threshold.

**Rationale:** the absolute rule was measured to force-reject the user's *favourite* venue. `bar`
against the dislike `bars` scores **0.932**, comfortably above any sensible absolute cutoff, at a
restaurant the user likes specifically because it hosts karaoke. Under the relative rule it survives
on `karaoke ↔ karaoke = 1.000` beating `bar ↔ bars = 0.932`.

The margin there is only **0.068**, so the threshold needs care — a default near 0.05 would be
knife-edge. This rule is the mechanism that makes "specificity wins" hold at the classification
layer as well as the scoring layer.

---

## Event persistence: what is stored, and what is deliberately not

**Decision:** `src/storage/events.py` owns `save_events()` / `load_events()`, replacing the private
row helper that lived in `NormalizationService`. Two fields are deliberately excluded:

- **`image_bytes`** — fetched only to feed the multimodal extraction call, which a reloaded event
  skips anyway. Storing photo blobs would bloat the database for no downstream reader.
- **`similarity`** — derived, cheap to recompute (no model calls), and the `recommendations` table
  is its real home once Phase 8 lands.

**Rationale:** before this, `NormalizationService` was the only writer and it ran *before*
enrichment, extraction, and embedding, so tags, summaries, and vectors were computed and discarded
every run. At roughly three minutes per event for local LLM extraction, a 50-event batch threw away
about two and a half hours of work. Nothing in `src/` read the events table at all.

Persisting also activates skip-if-done branches that already existed but were unreachable:
`ExtractionStage` (`if event.tags: continue`) and `EmbeddingStage`
(`if event.tag_embeddings: return`). With a read path, a re-run only pays model time for new
events. Closes issue #11.

---

## SimilarityStage receives preferences already loaded

**Decision:** `SimilarityStage.__init__` takes a `PreferenceSet`, not a `PreferenceRepository` and
file paths.

**Rationale:** the stage stays pure and deterministic — no I/O, no clock — matching
`NormalizationEngine` and `DeduplicationEngine`. It also makes "load preferences once per run" a
structural guarantee rather than a convention: the batch orchestrator loads once and hands the same
set to the stage, so the embedding cache cannot be consulted per event by accident.

---

## Match multiplier is applied direction-aware, not as a plain product

**Decision:** the ranking engine (Phase 8) applies the match multiplier as:

```
final_score = base_score × m + weather_adjustment     if base_score >= 0
final_score = base_score ÷ m + weather_adjustment     if base_score < 0
```

rather than the plain `base_score × match_multiplier` written in the original design.

**Rationale:** base scores are unbounded and negative values are valid, so a plain product inverts
the label's intent on exactly one combination — `no` on a negative base. At `no = 0.5`, a base of
−0.40 becomes −0.20, i.e. the events we are most confident the user dislikes are *rewarded*. This
reorders results rather than merely rescaling them:

```
A: base −0.60, match no    → −0.30      A ranks above B
B: base −0.40, match maybe → −0.40
```

Dividing instead of multiplying makes the multiplier act on *magnitude* with the sign preserved, so
`no` doubles the badness of −0.40 to −0.80 and the intent is symmetric in both directions.

**Scope of the defect, measured against the Phase 7 classifier:** `yes` is only assigned when
`base_score >= match_yes_min` (default 0.30), so `yes` cannot co-occur with a negative base, and
`maybe` is a 1.0 no-op. `no` on a negative base is therefore the only broken case. The fix is
written to be safe regardless, since `match_yes_min` is configurable and could be set negative.

**Consequence:** multipliers must be strictly positive, or a negative base divides by zero.
`load_config` now rejects non-positive multipliers rather than letting Phase 8 crash mid-batch.

**Rejected:** an additive offset (`base + offset(match)`) avoids sign issues entirely and composes
more predictably with absolute tier thresholds, but changes the config schema and needs calibrating
in score units not yet observed. Clamping the base at zero before multiplying was rejected outright
— it destroys ordering among disliked events, which the user explicitly wants visible in order to
confirm there is genuinely nothing on.

**Note for Phase 8:** the `yes` multiplier cannot reorder anything. Since `yes` is assigned by a
threshold on `base_score` itself, scaling that group by a constant is a monotone stretch; its only
real effect is pushing events across `top_picks_min`. The `no` multiplier is the one doing genuine
work, because `no` derives from the relative like/dislike margin — information the base score does
not contain. An event can score positively overall and still be a `no`.

---

## Weather is sampled hourly at the event's own time, not from the daily summary

**Decision:** `OpenMeteoProvider.fetch` requests Open-Meteo's *hourly* series for a local day and
returns all 24 records; `sample_hour` then selects the hour containing the event's `start_time`.
The fetch stays per-day, so one call still serves every event on that date and the existing
`weather_cache` keying on `(date, lat, lng)` is unchanged.

**Rationale:** two independent problems with the previous daily request.

1. It asked for `temperature_2m_max`. A daily high describes mid-afternoon, not a 9pm show, and
   the gap between them is the ordinary diurnal swing — the number was wrong for most events.
2. **Humidity and dew point do not exist at daily granularity in Open-Meteo.** They are hourly-only
   variables. Any comfort model using them forces the hourly switch regardless of point 1.

**Consequence:** anything with no `start_time` takes `weather.default_hour` (default 20) instead of
the daily extreme, so it is judged on a typical evening. This is what synthetic activities use —
they have no start time of their own.

**Also changed:** imperial units are requested natively (`temperature_unit=fahrenheit`,
`wind_speed_unit=mph`) rather than converted from celsius and km/h in our code. Deletes the
conversion arithmetic and the class of bug where it drifts.

**Testing note:** switching the provider to the hourly contract left all 650 tests green, because
every injected fake still returned the old flat dict. In production `event.weather` was receiving a
whole *day* and the synthetic generator would have crashed on the first real run reading
`weather["temperature_f"]`. Updating the fakes to the real shape turned 13 tests red at once. Fakes
that drift from the contract they stand in for test nothing — worth remembering given the
no-network-in-tests rule makes fakes ubiquitous here.

---

## Weather comfort is a signed adjustment with asymmetric caps

**Decision:** the term in the scoring formula is `weather_adjustment`, not `weather_bonus`:

```
final_score = (base_score ×/÷ match_multiplier) + weather_adjustment
adjustment  = comfort × (max_positive_adjustment if comfort >= 0 else max_negative_adjustment)
```

`comfort` spans −1.0..+1.0. Defaults: `max_positive_adjustment 0.15`, `max_negative_adjustment 0.25`.

**Rationale:** a bonus-only term can promote an outdoor event on a perfect night but leaves a
sweltering, muggy, windy evening scoring identically to a beautiful one. Bad weather is real
information about whether the user wants to go, so it should be able to demote.

The caps are asymmetric because the two directions are not symmetric in kind: a thunderstorm is
close to disqualifying, whereas a perfect night is a nudge that should not overturn a strong
preference match. Measured Phase 7 base scores span roughly −0.6..+0.9, so 0.15 lets good weather
lift an outdoor event past a moderately better indoor one without dominating, while 0.25 lets bad
weather sink one decisively.

**Applicability, in order:** `setting != "outdoor"` → no adjustment and no `Reason`; `weather is
None` (beyond the forecast horizon) → no adjustment, explicitly *not* a penalty, since not knowing
is not the same as knowing it is bad; otherwise the signed adjustment plus a `Reason`.

**Rejected:** scaling the adjustment by the centrality weight of an `outdoor` tag. `setting` is a
hard property of the event, not a fuzzy tag match — an outdoor concert is outdoors regardless of
how central "outdoor" was to its description. Phase 7's centrality argument does not transfer.

---

## Comfort curves are asymmetric trapezoids: a plateau, and linear

**Decision:** each weather factor maps its reading through a piecewise-linear trapezoid declared
entirely in config — `+1.0` across the whole `ideal` band, ramping linearly to `0.0` at the `zero`
bounds and on to `-1.0` at the `floor` bounds, then clamped. All four bounds are independent.

**Rationale, three separate choices:**

**A plateau, not a peak.** A bell curve has a single optimum and scores everything else lower. With
a comfortable temperature range of 20–65°F, a bell would call ~42°F perfect and rank 22°F and 63°F
as mediocre. The user does not have an ideal temperature; they have a range they are happy in.
Everything inside the band scores identically, and only leaving it costs anything.

**Independent sides.** The cold and hot ramps are configured separately, so they can decay at
completely different rates. For this user — comfortable to 20°F, walks below 0°F — temperature is
configured to reach 0.0 at −15°F but 78°F, and −1.0 at −40°F but 95°F. A symmetric curve cannot
express that at all.

**Linear, not logistic.** The Phase 7 similarity gate is logistic for a specific reason: cosine
similarity has a ~0.42 noise floor that must be crushed to zero (see "Scoring formula replaced
after measurement"). There is no noise floor here — 63.4°F means exactly 63.4°F. Logistic shaping
would only obscure the relationship between a config number and its effect, whereas "0.0 at 78°F"
is a promise readable off the config file and verifiable in one test.

**Consequence:** bounds must nest outward (`floor` outside `zero` outside `ideal`) or the curve
silently inverts its meaning. `load_config` rejects misordered bounds rather than scoring a
heatwave as pleasant.

---

## Dominant factors cap; they are not averaged

**Decision:** a comfort curve with `supersedes` set is a **capping factor** — excluded from the
weighted mean and applied as `min(comfort, its_value)`. Condition penalties (`rain`, `thunderstorm`,
…) work the same way, and only negative penalties cap: `0.0` means "no objection", not "no comfort".

**Rationale:** this was specified as a weighted factor first, and a red test caught it. With the
default curves, 12mm of rain (curve value −1.0, weight 0.9) alongside a pleasant 58°F (+1.0) and
dew point 50 (+1.0) averages to `(1.0 + 1.0 − 0.9) / 2.9 = +0.38` — a downpour rating as a fine
evening. **A weighted mean cannot express "this one factor disqualifies the rest"**, which is
exactly what precipitation and thunderstorms do. That is the same reasoning that made conditions a
`min()` cap in the first place; the capping factor simply generalises it.

**Consequence:** `weight` is unused on a capping factor. Documented on the `ComfortCurve` docstring
and in `config.example.yaml`, since a silently ignored config key is otherwise a trap.

---

## Precipitation intensity supersedes the categorical rain penalty

**Decision:** `precipitation_mm` declares `supersedes: [rain, snow]`. When a precipitation reading
exists it replaces those condition penalties entirely and the curve alone decides; the `-0.4`
categorical penalty survives only as the fallback for when the amount is missing. **Thunderstorm is
deliberately exempt from supersession.**

**Rationale:** the WMO condition code discards everything separating 0.2mm of drizzle from 12mm of
downpour — both arrive as `rain`. Penalising via both the condition and the curve would hit a
drizzle twice. Rain is genuinely not a simple no: a light drizzle at 58°F is a fine hike, which is
why `rain` softened from `-1.0` to `-0.4` and gradation was worth adding at all.

Thunderstorm stays absolute because its hazard is lightning, not millimetres. A *dry* thunderstorm
reads 0mm and would otherwise supersede its way into scoring as a beautiful evening. This is the
one case where the blunt categorical rule is the correct one.

---

## Correlated factors stand in for each other rather than both voting

**Decision:** `relative_humidity` declares `fallback_for: dew_point_f` — it is scored only when dew
point has no reading.

**Rationale:** humidity and dew point measure nearly the same thing, so counting both double-weights
moisture against temperature, wind, and air quality. Dew point is the better standalone comfort
predictor, so it wins when present. A fallback beats simply setting humidity's weight to zero,
because dew point is an hourly-only variable that may be absent — in that case humidity silently
covers for it instead of moisture dropping out of the score entirely.

**Consequence:** `fallback_for` naming an unconfigured factor is rejected at load. That typo would
otherwise disable a factor silently — it was caught by the validator failing a fixture in this
project's own test suite, which had `fallback_for: dew_point_f` with no `dew_point_f` curve defined.

---

## Missing weather readings renormalise; they are never scored as zero

**Decision:** a factor with no reading is dropped from the weighted mean and the remaining weights
renormalise. All factors missing yields an adjustment of 0.0.

**Rationale:** air quality comes from a separate Open-Meteo endpoint whose forecast horizon is much
shorter than the 16-day weather forecast, so AQI is absent for a large share of events. Scoring an
absent reading as 0.0 comfort would make "we don't know the air quality" indistinguishable from
"the air quality is mediocre", quietly penalising every event beyond the AQI horizon. The same
holds for any variable the provider omits.

---

## Weather is persisted raw, denormalised, with a reserved `observed` slot

**Decision:** `events.weather` stores `{sampled_hour, forecast: {issued_at, hour, day_series},
observed}` — the raw readings for the sampled hour, the full 24-hour series, and when the forecast
was issued. `observed` is written as `null`. Comfort scoring reads `observed` when present and falls
back to `forecast`.

**Rationale, three parts:**

**Raw readings, not a derived score.** Comfort curves will be retuned. Storing the readings means
every historical event can be rescored without refetching anything; storing only the verdict makes
history unusable the moment a bound moves.

**Denormalised onto the event.** The day series also lives in `weather_cache`, keyed
`(date, lat, lng)` and shared by every event that day. There is currently **no retention or purge
code anywhere in `src/`** — `data_retention_days` is configured but unread — so when Phase 10 adds
one, a cache purge would orphan every historical event from its own conditions. ~2KB of JSON per
event removes that coupling.

**`forecast` vs `observed`.** A forecast issued 10 days out is frequently wrong, so a future
"would I go" classifier trained on forecasts would learn what we *predicted*, not what the user
*experienced*. Open-Meteo has a free historical archive endpoint, so a backfill can attach reality
later. The slot exists now so adding that job fills a field rather than migrating every row.

**No schema migration required:** `weather_cache.data` and `events.weather` are already JSON TEXT
columns.

**Note:** the blunt `condition_penalty` table is a first pass at what is genuinely a nuanced
judgement — a drizzle at 58°F is a good hike, and severe conditions are not automatically a no for
this user. Raw retention is what makes replacing it with a classifier trained on real yes/no
history possible later.

---

## `setting` comes from LLM Pass 1; there is no source-type override map

**Decision:** `indoor` / `outdoor` / `unknown` is assigned by LLM Pass 1 alongside tags and summary,
with off-enum values coerced to `unknown` rather than triggering a retry. Synthetic activity rules
declare their own `setting:` in config. **No `setting_by_source_type` config map exists.**

**Rationale:** the plan originally specified exact matching on an `outdoor` *tag*, which HLD §4.5
Stage 3 prohibits and which fails on any event tagged `patio`, `rooftop`, or `beer garden`. A
structured field decided at extraction time avoids string matching entirely.

It then proposed a `weather.setting_by_source_type` map to give cinema sources an `indoor` default,
on the premise that movie events bypass Pass 1. **That premise was wrong.** `enrich_movie_event`
writes only to `event.metadata` and never touches `event.tags`, and `ExtractionStage` bypasses
solely on `if event.tags:` — its docstring says "this handles synthetic events". Movie events go
through Pass 1 like everything else and the model assigns their `setting` itself.

Independently of the factual error, config was the wrong home: a cinema is indoor as a matter of
fact, not deployment preference. Making it configurable invites declaring AMC outdoors and adds
config surface nobody will ever touch. A source-specific constant belongs in the adapter.

**Synthetic activities are the exception, and genuinely are config.** They bypass Pass 1 (they
arrive with tags pre-populated) and are entirely user-authored, so only the rule's author knows
whether "Read a book at home" is indoors.

**Consequence:** ambiguity resolves to `unknown`, which earns no weather adjustment in either
direction — never a penalty for being unclassifiable.

---

## Air quality is a separate endpoint with roughly half the forecast horizon

**Decision:** US AQI is fetched from `air-quality-api.open-meteo.com/v1/air-quality` by a separate,
optional, injected `AirQualityProvider`, gated behind `weather.air_quality.enabled`, and merged
into the sampled weather hour. Any failure degrades to no reading rather than failing enrichment.

**Measured against the live API on 2026-08-04**, rather than assumed:

| | Weather forecast | Air quality |
|---|---|---|
| Horizon | 16 days | **5 days default, 7 with `forecast_days`** |
| Out-of-range date | — | HTTP error with `{"error": true, "reason": "..."}` |

Two consequences fall out of that:

1. **AQI is absent for most events**, since the weather forecast reaches more than twice as far.
   Absence is therefore the normal case, not an error, and is never logged as one.
2. **An out-of-range date returns an error body, not null readings.** The provider checks
   `body.get("error")` explicitly — parsing on would otherwise raise inside the `hourly` lookup and
   be caught by the generic handler, which works by accident rather than by intent.

**Consequence for scoring:** a missing reading is omitted from the hourly record entirely rather
than written as `None`, so `compute_comfort` drops the factor and renormalises. Writing `None` and
scoring it would make "we do not know the air quality" indistinguishable from "the air quality is
mediocre", quietly penalising every event past the horizon. See "Missing weather readings
renormalise".

**Rejected:** making air quality a hard dependency of `EnrichmentService`. A second endpoint with a
shorter horizon and no bearing on most events should not be able to take down weather enrichment,
so the provider is optional and its absence is indistinguishable from a miss.

**Known gap:** air quality is cached only in memory for the duration of a run, whereas weather is
also cached in the `weather_cache` table. Deliberate for now — AQI is volatile and cheap — but it
means the two follow different caching rules, which is worth knowing before adding a TTL to either.

---

## Cached forecasts expire; scoring uses the freshest one for the event's date

**Decision:** `weather_cache` entries are served only while they are younger than
`weather.cache_ttl_hours` (default 12). A stale entry falls through as an ordinary cache miss and
is overwritten in place. Filed during Phase 8 as issue #14; fixed at the start of Phase 9.

**Rationale:** the cache is keyed `(date, latitude, longitude)` with no expiry, and nothing ever
deleted an entry. An event discovered a week ahead therefore cached the forecast issued that day
and was scored against it on every subsequent run — including the run on the night it actually
happens, which is the only run whose ordering the user ever sees. Once `weather_adjustment` feeds
the ranking directly, a stale forecast is not a stale cache entry, it is a wrong recommendation.

**Why under 24 hours specifically:** the batch is nightly, so any TTL below a day guarantees each
run refetches every date it needs. Within a run the cache still collapses all events on a date to
one provider call, so the cost is one call per distinct date per run against a free, keyless
endpoint.

**Consequence:** `_db_weather_put` now stamps `fetched_at` from the injected `get_now` rather than
calling `datetime.now()` directly. That was a standing violation of the injectable-time rule, and
expiry cannot be tested without it.

**Note:** air quality is still cached only in memory for the duration of a run, so the two now
follow different rules for a second reason — weather expires on a clock, AQI expires with the
process.

---

## Comfort is computed at ranking time, not during enrichment

**Decision:** `EnrichmentService` persists raw readings; `RankingEngine` calls `compute_comfort`
when it scores. No adjustment is ever stored alongside the readings.

**Rationale:** Phase 8 deliberately stored raw readings rather than a derived verdict so retuned
curves could rescore history without refetching anything. Storing the adjustment too would give
that up for nothing — the stored value would be the one computed under whichever curves happened
to be configured that night, and no later reader could tell which.

**Consequence:** `select_readings` prefers `observed` over `forecast`, which is the first code to
read the slot Phase 8 reserved. A later backfill of what actually happened will rescore those
events automatically, with no migration and no other change.

---

## Ranking consumes the attached similarity result; it does not re-score

**Decision:** `RankingEngine.rank(events, run_date)` reads `event.similarity`, which
`SimilarityStage` has already attached. It takes no `PreferenceSet` and never calls the similarity
engine.

**Rationale:** two code paths computing the same score will drift, and the drift would be silent
because both look right in isolation. Ranking's job is ordering; semantics belong upstream.

**Consequence:** an event arriving with `similarity is None` scores `base_score = 0.0` and match
`maybe` rather than raising — one unscorable event costs one recommendation, not the batch,
matching the existing stage behaviour. Note where that lands it: at zero, i.e. the middle of the
ranking, which is also where a thin extraction lands. Uncertainty of every kind collects in the
middle, which is the intended failure mode but worth knowing when reading a result list.

---

## `recommendation_id` is derived, never generated

**Decision:** `make_recommendation_id(run_date, event_id)` returns `"2026-08-06:evt-1"`. No uuid4.

**Rationale:** determinism is the phase's headline guarantee, and "two runs of the same batch
produce identical output" is untestable if every row carries a fresh random id. A derived id also
makes re-running a date idempotent by construction rather than by convention.

---

## Tag confidence is symmetric, unlike the match multiplier

**Decision:** `tag_confidence = min(1.0, len(tags) / min_tags_per_event)`, multiplied into
`base_score` **in both directions** — a thin positive falls toward zero and a thin negative rises
toward zero. Synthetic activities are exempt and always score at full confidence.

**Rationale:** `min_tags_per_event` had been configured since Phase 1 and read by nothing. Fewer
than five tags means LLM Pass 1 underperformed, so the score built on those tags rests on less
evidence, and the ranking should say so.

The symmetry is the load-bearing part, and it is deliberately the opposite of the rule immediately
above it in the formula. The **multiplier** expresses how strong a verdict is, so it must preserve
sign — dividing a negative deepens it. **Confidence** expresses how much evidence exists at all,
so it must pull both signs toward zero. Making confidence direction-aware "for consistency" would
deepen a thin negative, punishing an event for evidence we never gathered. Each rule looks like a
bug from the other's perspective; both are commented in place.

**Where a low-confidence event lands:** in the middle of the ranking, not at the bottom. That is
the correct place for something we know almost nothing about.

**Synthetic activities are exempt** because their tags come from hand-written `config.yaml` rules.
A three-tag rule is an authoring choice, not an extraction failure, and scaling it down would
demote the user's own activities unless they padded every rule to five tags.

**No floor and no extra config knob:** one tag out of five scores ×0.2, and zero tags scores 0.0.
Zero tags is total extraction failure; its `tag_score` is already 0, so the only thing additionally
discarded is the summary term, which is not worth ranking on alone.

**Emits a `low_tag_confidence` reason** whenever confidence is below 1.0, so a demoted event is
never demoted invisibly.

---

## The bottom tier is `everything_else`, not `excluded`

**Decision:** the tier below `worth_considering_min` is named `everything_else`.

**Rationale:** ranking withholds nothing. `tier` is a label the CLI renders, and the old name
invited exactly the behaviour the design forbids — one stray `WHERE tier != 'excluded'` would
silently hide events the engine was careful to keep. Renaming makes that mistake impossible to
write by accident.

**Consequence:** because tiers are cut on `final_score`, the weather adjustment can move an event
across a boundary between runs. That is intended — an outdoor event genuinely is a worse option in
a thunderstorm — but it means tier is a property of an event *on a night*, not of the event.

---

## The blocklist applies at ranking too, on venue names only

**Decision:** `RankingEngine` drops events whose venue matches `data/blocklist.json`, using the
matcher extracted to `src/utils/blocklist.py` and shared with venue discovery. It is the only drop
in the system. Dropped events are logged with their venue.

**Rationale:** discovery and ingestion both filter earlier, but neither can reach an event that is
already in the database. Without a check at ranking, a venue blocked today keeps being recommended
forever from events scraped yesterday. The matcher is shared rather than reimplemented so the two
call sites cannot disagree about what "blocked" means.

**Known gap:** `@handle` entries cannot be matched here. An `Event` carries no handle —
`EventNormalizer` keeps only `source_event_candidates` and `source_type` — so ranking matches
venue names alone, and handles stay enforced at ingestion where the candidate still has one. Filed
as issue #15 rather than expanding Phase 9 into the events schema.

**Consequence:** a blank or missing venue name never matches a name entry. An empty string scores
against every entry and would block indiscriminately.

---

## A run's recommendations replace that run's rows, and an empty batch clears nothing

**Decision:** `save_recommendations` deletes only the `run_date`s it is about to write, then
inserts. Saving an empty list returns immediately.

**Rationale:** a batch retried after a partial failure must supersede its own earlier attempt,
or the CLI reads two conflicting orderings for the same night. Scoping the delete to the dates
being written keeps previous nights intact, which matters because those rows are the only record
of what was recommended when.

Treating an empty list as a no-op rather than a clear is the conservative reading: an empty batch
means "nothing to add", and interpreting it as "delete everything" would be both silent and total.

---

## The CLI folds the bottom tier; it never hides it

**Decision:** the default view renders `TOP PICKS` and `WORTH CONSIDERING`, then a single line
reporting how many events sit below them: `+ 14 more events ranked lower (--all)`. `--all` expands
them under an `EVERYTHING ELSE` heading.

**Rationale:** the Phase 10 spec's red test said "excluded events do not appear in default output",
written before the tier was renamed. Nothing is excluded, and hiding a category outright would
undo the rename's whole point. Folding keeps the default view short without making anything
invisible: the count is always on screen, so a mis-tiered event announces itself as a number that
looks wrong rather than by silently never appearing.

**Consequence:** the thresholds are uncalibrated this early, and this is the mechanism for noticing
that. `-v` complements it by printing `final_score`, `base_score`, `weather_adjustment` and
`tag_confidence` per event, so a boundary that looks wrong can be judged on the number the cut was
made on rather than on the tier label alone.

---

## Undated events get a labelled section, not a filter

**Decision:** an event with no `start_time` appears in the default view under
`UNDATED — timing unconfirmed`, below the timed sections. It is dropped only by `--time` and
`--after-sunset`.

**Rationale:** an event with no start time is not *known* to be today; it is known to be in
today's *run*. That argues for labelling it honestly rather than hiding it — dropping it would
lose a real event because extraction failed to find a date, which is a gap in what we know and not
evidence about when it happens. The timing filters are different: each is a claim about when an
event occurs, and we cannot assert it for an event with no time at all.

**Consequence:** `on_date()` deliberately drops undated events, and `_cmd_recommend` adds them back
before rendering. The filter stays a clean predicate; the view decides what to do with the gap.

---

## The CLI renders with the standard library, not `rich`

**Decision:** plain strings and bare ANSI codes, suppressed when stdout is not a terminal.

**Rationale:** the deciding cost is testing, not the dependency. `rich` wraps to terminal width, so
`assert "Karaoke at The Dive" in output` can fail because a line broke mid-title, and every render
test then has to pin console width or strip-and-rejoin. It earns that for dense columnar output;
this view is a list — sections, one or two lines per event, indented reasons.

**Consequence:** `render_recommendations` and `render_raw` are pure `-> str`, so adopting `rich`
later touches one module and no tests outside it. Revisit if the output ever wants real columns.

---

## Database readiness is checked before reading, and injected with the loaders

**Decision:** `has_schema(db_path)` (`src/storage/db.py`) reports whether a database exists *and*
carries the `events` and `recommendations` tables. The CLI calls it before any read, and takes it
as an injected parameter alongside its loaders.

**Rationale:** `sqlite3.connect` creates a zero-byte file for any path it is handed, so checking
`Path.exists()` proves nothing — a stray read leaves behind a file that then looks like a database
and fails with `no such table` on the next run. This was not hypothetical: an early version of the
CLI created exactly such a file in `database/` while a test was failing.

Injecting the check matters for the same reason the loaders are injected. A test that substitutes
the loaders but leaves the readiness check reading the real filesystem passes or fails on whether
the developer's own `database/` happens to exist, which is the kind of coupling that makes a suite
pass on one machine and fail on another.

**Consequence:** a missing database at the *default* path exits 0 with "run the overnight batch" —
it is the normal state before the first run. A missing database at an *explicitly named* `--db`
path exits 1 on stderr, because that is a typo and reporting it as "no events" would hide the
mistake.

---

## Candidate ids are derived from the source's own key, never generated

**Decision:** every ingestion adapter builds its `EventCandidate.id` from
`derive_candidate_id(source_type, *parts)` (`src/ingestion/candidate_id.py`), preferring the
upstream item's own identifier and falling back to a content hash only where the source publishes
none. `uuid4` appears in no adapter.

**Rationale:** the five original adapters minted a fresh `uuid4` on every fetch, so
`INSERT OR REPLACE INTO event_candidates` never matched an existing row and each nightly run
inserted a brand-new duplicate of every candidate it had ever seen. Nothing ran on a schedule, so
this was invisible; the batch orchestrator is what makes it destructive, and unwinding it later
would be a data problem — purging duplicate candidate rows and merging events whose tags are
scattered across the duplicates — rather than a code fix.

Every adapter already carried a usable key in the payload it was parsing and discarded it: apify
`id`, picuki `post_id`, dumpor `shortcode`, Veezi `ScheduledFilmId`, and AMC's showtime `id`, which
the GraphQL query already requested.

**Veezi takes a composite.** `ScheduledFilmId` identifies the *film*, not the screening. Using it
alone would collapse every showtime of a film onto one candidate — turning a duplication bug into a
data-loss one — so the showtime is always part of the material.

**Consequence:** the natural-key tier is stable under edits; the content-hash fallback is not, so an
edited description there mints a new id and re-extracts. That is the accepted cost of a source with
no identifier, and the reason the natural key is always preferred.

**`derive_candidate_id` raises on empty material** rather than returning a shared id. Passing
nothing identifying is a caller error, and a silent collapse would lose an entire source's
candidates while looking like a quiet night.

---

## Normalization does not persist; the orchestrator owns every save point

**Decision:** `NormalizationService.run()` returns its deduplicated events and writes nothing.
`NormalizationResult.persisted` became `normalized`, and `db_path` left the constructor. The batch
orchestrator saves after extraction, after embedding, and after semantic dedup.

**Rationale:** the events normalization produces still carry `Normalizer`'s throwaway uuids.
Persisting them wrote rows that `reconcile` then orphaned the moment it adopted a stored id — the
same nightly row-doubling reconcile exists to prevent, merely relocated one stage earlier. Loading
`stored` before the call does not help: the fresh-uuid rows are already on disk.

The early save was added for issue #11, when `NormalizationService` was the only writer in the
system and tags and embeddings were being discarded every run. Normalization output carries no
expensive work of its own, so once the orchestrator owns the save points the reason for it expires.

**Consequence:** the save points are now exactly where expensive work is produced, which is what
activates the skip-if-done branches in `ExtractionStage` and `EmbeddingStage`. At roughly three
minutes an event, that is the difference between a re-run costing minutes and costing hours.

---

## Reconcile carries the extraction output whole — including `setting`

**Decision:** a fresh event adopting a stored event's id also adopts `tags`, `summary`, `setting`,
`tag_embeddings`, `summary_embedding`, and `astronomical_data`. It never adopts `weather` or
`metadata`. `created_at` comes from storage; the fresh event stays authoritative for scraped
content.

**Rationale, field by field:**

**`setting` is the trap.** It is LLM Pass 1 output, but `ExtractionStage` bypasses on `if
event.tags:` alone. Carry the tags without the setting and extraction skips, leaving `setting` at
`"unknown"` — which earns no weather adjustment in either direction. An outdoor event would
silently stop being weather-scored from its second night onward, and nothing would look broken.
Anything Pass 1 produces has to travel with the tags that cause it to be skipped.

**`weather` must never travel.** `weather.cache_ttl_hours` exists precisely so a nightly batch
rescores against a forecast issued that night. A carried forecast would score an event discovered a
week out on the day it was found, forever.

**`metadata` must not travel either.** The normalizer writes this-run flags there
(`missing_title`, `missing_start_time`), so carrying stored metadata over fresh would resurrect
stale flags from a night when the data was thinner.

**`astronomical_data` may travel** because it is deterministic from date and location and cannot go
stale.

**`created_at` records when the event was first seen.** Taking the fresh value would overwrite it
with tonight's on every run, and `merge_cluster` tiebreaks on it.

---

## A split cluster is merged back onto its stored event

**Decision:** when two fresh events both match one stored event, they are merged into a single
event that adopts its id. Superseded 2026-08-06; see the reversal below.

**Rationale:** the plan covered one fresh event matching many stored ones — a cluster growing. The
mirror happens too: stored `S` carries candidates `[c1, c2]` from an earlier merge, tonight's dedup
keeps them apart, and fresh `A=[c1]` and `B=[c2]` both match `S`.

`S` is not merely an id to inherit — it is the record of a **semantic dedup verdict already paid
for**. Pass 1 is fuzzy and splits what pass 2 merged; pass 2 is the more capable judge by design.
Re-litigating its verdict nightly with the weaker pass is the defect, so the stored event is
treated as the oracle.

**The consequence of a merge being self-perpetuating is accepted.** Once `S` claims both
candidates, every later run re-merges them and the two can never separate again. A wrong merge is
therefore cemented — but it is cemented today regardless, because `S` already carries both
candidates and nothing undoes that. Pre-v1 the relief valve is deleting the database; post-v1 this
would want a real unmerge path.

**Reversed:** the original decision was that the first fresh event adopts the id and the rest keep
their own, on the grounds that letting both adopt would collide on the primary key and
`INSERT OR REPLACE` would make one silently overwrite the other — data loss, where the mirrored
case is only duplication.

The collision reasoning was right; the conclusion was not. The alternative is not *only*
duplication, because nothing ever cleans up the loser. Measured over six consecutive runs on the
real fixtures, the batch grew by **two events and two re-extractions every run, without bound** —
328, 330, 332, 334, 336. The loser was emitted with a fresh uuid and no tags, so it re-extracted at
roughly three minutes and was saved as another row, which then claimed an already-claimed candidate
and became invisible to the next run. After the change: 327 events, stable, and zero re-extraction
from run 2 onward.

**Also fixed:** the index was `candidate -> Event`, so when several stored events claimed one
candidate a plain dict kept only the last. A hidden owner is never reported stale, so it lingered
for the life of the database. It is now `candidate -> list[Event]`.

**Rejected:** raising instead. A legitimate dedup threshold change would then abort a batch.

**Note:** `merge_cluster` choosing its base by fewest-null fields is now load-bearing in a second
place. It was dedup's internal business; it now also decides which fresh event's content survives a
re-merge, so changing that heuristic reaches further than its name suggests.

---

## A dry run persists nothing, which ingestion has to help with

**Decision:** `--dry-run` writes no events, no recommendations, no deletes — and
`IngestionService.run(persist=False)` fetches and filters as normal while touching no table.
`IngestionResult.persisted` became `accepted` and now carries the candidates themselves.

**Rationale:** a flag named dry-run that writes to the database is a trap, and "it is only
candidates" is the reasoning that makes a flag untrustworthy. But ingestion is the only stage that
touches live providers, so a dry run that skips it entirely stops being a rehearsal for the riskiest
moment in this system — the first run where credentials, rate limits, and real payloads meet at
once. Fetching without writing keeps both properties.

**Ingestion writes in four places, not one:** seed sync, candidates, handle discovery, and promotion
evaluation. Gating only `_persist_candidate` would leave a dry run seeding `candidate_entities`,
quietly shaping later real runs. `persist=False` skips all four and never opens a connection, so it
cannot leave a stray zero-byte database either.

**Consequence:** nothing was written, so the candidate loader cannot see what was just fetched. The
orchestrator folds those candidates in from memory on a dry run only, preferring the loaded object
on an id collision and sorting the way `load_candidates` sorts, so both paths hand the pipeline the
same order.

---

## Ranking scope is a local-date window, plus everything stored that nothing claimed

**Decision:** the orchestrator ranks events starting between `run_date` and
`run_date + scraping.horizon_days`, plus undated events created inside `lookback_days`, plus stored
events that no fresh event reconciled against.

**Rationale, three parts:**

**Local date, not UTC.** An event at 11pm local is tomorrow in UTC, so a UTC comparison would
misfile exactly the evening events this system exists to rank. `start_time` converts through
`config.location.timezone` before its date is taken.

**Undated events are kept on discovery age.** The CLI has a labelled `UNDATED` section, and dropping
one would lose a real event to a failed extraction rather than to anything we know about when it
happens.

**Carrying forward unclaimed stored events is load-bearing, not tidiness.** Reconcile only returns
fresh events. Ranking just those would silently stop ranking events whose candidates aged out of the
window, and the settled policy that a wholesale ingestion failure still re-ranks stored events
against tonight's forecast would not work at all — with no candidates there are no fresh events, so
ranking would receive an empty list.

**Wholesale embedding failure stops the batch before ranking.** Every other stage failure is
non-fatal and the batch continues with what it has, but ordering computed without vectors is
meaningless and would be persisted as though it meant something. Everything already saved stays
saved.

---

## The blocklist has no database table

**Decision:** the `blocklist` table is removed from the schema. `data/blocklist.json` is the only
source, read once by the composition root and handed to `IngestionService` and `RankingEngine`.

**Rationale:** the table had **zero readers and zero writers** — not "written but unread", entirely
inert. The intent recorded in CLAUDE.md was that it be "overwritten from file at each batch start",
but no consumer ever appeared, and the composition root has since made the in-memory path the
working one.

Asked the other way round — if we were building this from nothing, would we add it? — the answer is
no. It would be a cache of a file already read at startup, invalidated by nothing and queried by no
one. Dead schema is worse than no schema, because the next person to see it may reasonably query
it, get zero rows, and silently block nothing.

**Not affected:** `venues.blocklisted` is a different column on a different table and is genuinely
written by venue discovery.

**Cheap because the database is empty.** Pre-v1 the schema is changed and the database deleted, so
dropping a table costs nothing today and would cost a migration later.

---

## Extraction and embedding are gated on a hash of their input

**Decision:** `ExtractionStage` runs when `sha256(title + description)` differs from the hash stored
on the event, and `EmbeddingStage` runs when `sha256(tags + summary)` differs from its own. Both are
written **only on success**. Synthetic events are exempt from extraction entirely, by provenance.

**Rationale:** `if event.tags` was doing three jobs and could only tell them apart by accident:

1. *Already extracted* — the incremental case, which it handled.
2. *Authored, never extract* — synthetic activities, which it handled only because their tags happen
   to be non-empty.
3. *Failed, retry* — which it could not distinguish from a valid empty-tag result, so an event the
   model legitimately found no tags for re-extracted every night forever at ~3 minutes a go.

It also could not see an **edited description**: a corrected event kept its stale tags for the life
of the database. A hash separates all three states cleanly — set means done, absent means never ran,
and a failure leaves it absent so the next run retries.

**No cache table.** The event row is already the cache for its own content. A separate
content-addressed table would only buy reuse across *distinct* events with byte-identical text,
which is too rare to pay for. This is the small half of the plan's rejected "Option C"; the identity
half — content-derived `event_id` — stays rejected.

**Embeddings need their own hash, and it was verified before being fixed.** Their input is tags and
summary, both extraction's output, so hashing them makes the chain automatic. Without it, an event
re-extracted from `karaoke` into `punk` kept its `karaoke` vectors — measured, not theorised — and
went on being scored against tags it no longer had. A silent misranking rather than a visible
failure.

**The synthetic exemption is provenance, not state**, so it reads `Event.is_synthetic` rather than
anything a stage stores. A hash rule alone would have been *destructive* here: synthetic events
carry a title and hand-authored tags, so on their first run the hash would mismatch, extraction
would run, and the LLM would overwrite what a person wrote in `config.yaml`.

**Rejected: a persisted `extraction_exempt` boolean.** It reads well at the call site, but it is a
second encoding of a fact `source_type` already carries, and two encodings can disagree. A derived
property costs no schema and cannot fall out of sync. It also reverses CLAUDE.md's recorded "no
special flag", where a property does not.

**Cost of deferring, for the record:** ~16 hours of re-extraction over ~328 events had this landed
after the first live run instead of before it.

## Cinema showtimes come from Veezi's public page, not its API

**Decision:** Showtimes are read from a cinema's **public Veezi ticketing page**
(`ticketing.useast.veezi.com/sessions/?siteToken=<token>`) by `VeeziSessionsSource`. The
API-key adapter specified from the outset — `CinemaVeeziAdapter`, `VEEZI_API_KEY` — is deleted.

**Rationale: the API was never obtainable.** The design assumed a Veezi key could be acquired the
way a TMDb key can. It cannot. Veezi is *exhibitor* software, and its API key is issued from the
cinema's own back office — you get one by operating the cinema, not by asking. The adapter was
therefore dead code from the day it was written, permanently skipped for a missing credential and
warning about it nightly. Nothing surfaced this until someone tried to obtain the key.

**The public page is a better source anyway**, which is the part worth remembering:

- **No credentials at all.** The `siteToken` appears in the cinema's own booking links.
- **Server-rendered.** 200 OK and complete HTML, unlike `cinemasalem.com` itself, which is a
  Next.js shell serving 3.4 KB and an empty `__NEXT_DATA__`.
- **One adapter covers every Veezi cinema.** The token is part of the configured URL, so
  CinemaSalem and Warwick both landed from one implementation, and a third is a config entry.
- **It carries a per-showing id.** `/purchase/38750` is the cinema's own key, measured 1:1 against
  distinct showings with no collisions — a stable candidate id for free, which the API response
  would also have supplied but which scraping usually does not.

**The page lists each showing more than once** — 60 of 144 rows on the measured capture — so the
parser deduplicates on that session id. Counting rows would have overstated the schedule by 70%.

**`source_type` stays `cinema_veezi`.** It is still Veezi, and the label already drives the
`movies` preference domain and TMDb enrichment. Renaming it would have churned config, `domain_map`
and stored rows to record nothing but which door we came in by.

**The general lesson, since it cost a source:** an integration whose credential nobody has tried to
obtain is unvalidated, however well specified. AMC failed the same test at the same time and had no
public fallback — its API is a partner catalog program, and its site answers automated clients with
a 403. Veezi had one, and only because the public page was looked for after the API closed.
