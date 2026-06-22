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
