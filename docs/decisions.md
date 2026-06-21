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

## Seed handles → candidate_entities, not venues

**Decision:** Handles in `seeds.yaml` (e.g. `@cinemasalem`) are written to `candidate_entities`
in `probationary` state, not to the `venues` table.

**Rationale:** A handle is not a venue — it's a social account that may or may not correspond
to a physical venue. Resolution happens during event ingestion (Phase 3). Seed venue entries
(name + address) go directly to the `venues` table.

---

## Venue categories: user-configurable in config.yaml

**Decision:** The list of venue category slugs to search for lives in `config.yaml` under
`venue_discovery.categories`.

**Rationale:** Different users care about different types of venues. Making categories
configurable means no code changes to add or remove a category. Default list covers the common
cases (cafe, bar, restaurant, etc.) but users can freely extend or narrow it.
