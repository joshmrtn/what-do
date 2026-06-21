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
