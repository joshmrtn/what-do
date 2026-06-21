# High Level Design (HLD)

# Project: Local Event Intelligence Hub

Version: 0.1

---

# 1. System Overview

## 1.1 Purpose

The Local Event Intelligence Hub is a local-first recommendation engine that converts messy, unstructured public information into personalized recommendations.

The system is designed around asynchronous batch processing rather than real-time computation.

Heavy processing shall occur during scheduled background jobs. Interactive user requests shall execute using precomputed data.

The architecture prioritizes:

- Deterministic recommendation ranking
- Explainable outputs
- Local-first AI processing
- Provider independence
- Future UI portability

---

# 1.2 High-Level Architecture

The application is divided into six independent layers.

```text
External Sources

↓

Ingestion Layer

↓

Normalization Layer

↓

Enrichment Layer

↓

Semantic Processing Layer

↓

Storage Layer

↓

Presentation Layer
```

Each layer has a single responsibility.

Dependencies flow in one direction only.

No layer may directly depend on a higher layer.

---

# 2. Architecture Principles

## 2.1 Background-First Architecture

Expensive operations shall never execute during interactive user requests.

Examples:

- Web scraping
- LLM processing
- Embedding generation
- Weather retrieval

These operations shall occur during scheduled background runs.

---

## 2.2 Deterministic Ranking

Large Language Models may extract information.

Large Language Models SHALL NOT determine final recommendation ordering.

Final ranking shall be deterministic.

---

## 2.3 Loose Coupling

Components shall communicate exclusively through data contracts.

Components shall not directly access internal state belonging to other components.

---

## 2.4 Provider Independence

External data providers shall be interchangeable.

Provider replacement shall not require modifications to downstream systems.

---

## 2.5 Local-First Operation

Preference matching and semantic processing shall execute locally whenever practical.

Cloud services shall remain optional.

---

## 2.6 Fault Tolerance

Failure of a single provider shall not terminate a batch run.

The system shall continue processing remaining providers.

---

## 2.7 Pluggable Pipeline Stages

Each stage in the processing pipeline shall implement a common interface.

The pipeline runner shall execute registered stages in sequence.

Stages may be inserted, removed, or reordered without modifying adjacent stages.

This pattern applies to both data source adapters and processing stages.

A source adapter interface: `fetch() → List[EventCandidate]`

A pipeline stage interface: `process(events: List[Event]) → List[Event]`

Adding a new source or a new processing step shall require only registering a new implementation — no changes to the orchestrator or adjacent stages.

---

# 3. System Flow

## 3.1 Daily Batch Pipeline

The background pipeline shall execute once per day.

```text
1. Discover venues

↓

2. Discover entities

↓

3. Scrape event sources (social media adapters, movie/theater adapters)
   → extract @handles from post captions → store as probationary in candidate_entities

↓

3a. Disambiguate candidate entities (gemma4:e2b)
    → classify each new probationary handle as venue or person
    → persons → state = discarded (removed from pipeline)
    → venues → remain probationary
    → evaluate promotion: handles with mention_count ≥ threshold from seed sources → active

↓

4. Normalize events

↓

5. Deduplicate events

↓

6. Retrieve weather

↓

7. Retrieve solar data

↓

8. Generate synthetic activities

↓

9. Execute LLM Pass 1

↓

10. Generate embeddings

↓

11. Calculate similarity

↓

12. Execute LLM Pass 2

↓

13. Calculate deterministic scores

↓

14. Persist results

↓

15. Generate logs

END
```

> **Addendum:** Step 12 (LLM Pass 2) is deferred to post-v1. In v1, similarity scores from step 11 feed directly into deterministic scoring at step 13. The pipeline slot is preserved so LLM Pass 2 can be inserted later without restructuring the pipeline. See section 4.5 for detail.

> **Addendum:** Step 3a (Disambiguate candidate entities) was added during Phase 3 design. The original HLD described handle disambiguation within the ingestion layer, but the ingestion layer must not invoke semantic models (section 4.1). Step 3a is a dedicated post-scraping stage that classifies discovered handles using gemma4:e2b and evaluates handle promotion. This keeps ingestion LLM-free while still classifying handles within the same batch run.

## 3.1.1 Per-Event vs Set Operations

Some pipeline steps operate on one event at a time. Others require the full set of events to be in memory simultaneously.

Set operations (must collect all events before proceeding):

- Step 5: Deduplication Pass 1 — compares all events against each other
- Step 2b (after step 10): Deduplication Pass 2 — compares all event embeddings against each other

Per-event operations (can process one at a time, order-independent):

- Steps 4, 6, 7, 9, 10, 11, 13 — normalization, enrichment, LLM extraction, embedding, similarity, scoring

The batch runner shall collect all EventCandidates before beginning normalization. It shall complete deduplication before proceeding to enrichment. This ensures set operations have complete data without requiring the full event set to be held in memory throughout the entire pipeline.

Score retention policy:

Raw scores, embeddings, tags, and all intermediate data shall be stored as computed. Scores shall not be normalized relative to the current batch. This ensures events discovered across different batch runs remain comparable, and allows the scoring formula to be updated and re-applied to historical data without re-running ingestion.

---

## 3.2 Interactive User Flow

```text
User CLI Request

↓

Read Database

↓

Apply Time Filters

↓

Apply User Flags

↓

Render Output

↓

Return Results
```

The interactive flow shall not invoke:

- network requests
- LLM calls
- embedding generation

---

# 4. Component Responsibilities

# 4.1 Ingestion Layer

Purpose:

Discover raw event information.

Responsibilities:

- discover venues
- discover social handles
- retrieve event sources (Instagram/Facebook via Apify, Picuki, Dumpor — failover chain)
- retrieve movie schedules (Cinema Salem via Veezi/Vista API; AMC via AMC Showtime API)
- collect metadata

Output:

EventCandidate objects.

The ingestion layer shall not perform scoring.

The ingestion layer shall not invoke semantic models.

## Source Seeding

The system requires at least one known handle or venue to bootstrap discovery.

Seed sources are defined in `data/seeds.yaml`.

Example structure:

```yaml
handles:
  - "@cinemasalem"
  - "@thevaultlounge"
venues:
  - name: "Cinema Salem"
    address: "95 Washington St, Salem MA"
```

This file is the source of truth for all manually known sources.

New handles may be added via a CLI command:

```
what-do add-source @handle
what-do add-source --venue "Venue Name" --address "123 Main St"
```

The CLI command writes to `seeds.yaml` and is the recommended interface for ongoing additions. Direct file edits are also valid.

The system shall skip seeds that are already present in the active sources list.

## Scraping Time Window

The system shall apply configurable time boundaries when scraping and indexing content.

Look-back window (configurable, default 30 days):

Posts and events older than this window shall be ignored during ingestion.

Look-forward window:

No limit. Events announced far in advance shall be retained regardless of how far ahead they are scheduled.

## Recursive Entity Discovery & Venue Disambiguation

During scraping, post content may reference additional social handles (e.g. `@localvenuename`).

The system shall collect discovered handles as candidate entities and evaluate them for follow-up scraping.

Handle states:

- `active` — scraped on every batch run. Seed handles start here (user-trusted). Promoted probationary handles land here.
- `probationary` — discovered during scraping; not yet scraped. Awaiting classification and promotion.
- `discarded` — classified as a person handle; permanently excluded.

Disambiguation (step 3a — runs after scraping, before normalization):

- The LLM classifies each new `probationary` handle as `venue` or `person` using the surrounding caption as context.
- Default model: `gemma4:e2b` (smaller and faster — simple binary classification task).
- Handles classified as `person` → state set to `discarded`.
- Handles classified as `venue` → remain `probationary`, continue toward promotion.
- Already-classified handles (state ≠ `probationary`) are skipped.

Promotion (also in step 3a, after disambiguation):

- A handle is promoted to `active` when `mention_count ≥ candidate_promotion_threshold` AND at least one entry in `mention_sources` is a seed source.
- Trusted sources (v1) = seed sources only. This prevents two low-quality discovered handles from promoting each other.
- The promotion threshold is configurable (`scraping.candidate_promotion_threshold` in `config.yaml`).

Depth tracking:

- Seed handles: `depth = 0`.
- Handles discovered in posts from depth-0 sources: `depth = 1`.
- A configurable `max_depth` prevents recursive discovery beyond a set number of hops from seed sources.
- Handles at `max_depth` are not stored.

---

# 4.2 Normalization Layer

Purpose:

Convert inconsistent data into canonical structures.

Responsibilities:

- standardize timestamps
- standardize venues
- standardize locations
- normalize text
- remove malformed records

Timezone:

All timestamps shall be converted to the local timezone of the configured coordinates at normalization time.

Timezone shall be derived from lat/lng using a library (e.g. `timezonefinder`) — not hardcoded.

All datetimes stored in the database shall include timezone information.

Malformed record policy:

A record shall be discarded if it is missing both a title and a start_time.

Records missing only one of these fields shall be retained but flagged for review.

Discards shall be logged with the reason and the source.

This layer performs no scoring.

---

# 4.3 Deduplication Layer

Purpose:

Merge duplicate events.

Deduplication may consider:

- title similarity
- venue similarity
- time proximity
- semantic similarity

Duplicate events shall merge into a canonical event.

Canonical record policy:

The most complete record wins (fewest null/empty fields). Fields from the winning record are used as the base. Any non-null fields from secondary records that are null in the winner are merged in.

The `sources` field on the merged event shall contain attribution from all contributing records.

Original source attribution shall be preserved.

Pass 1 (fuzzy, pre-embedding):

- Title similarity above a configurable threshold (default 0.85)
- AND same venue
- AND start times within a configurable window (default 2 hours)

Pass 2 (semantic, post-embedding):

- Cosine similarity between event description vectors above a configurable threshold
- Catches same events described differently across sources

---

# 4.4 Enrichment Layer

Purpose:

Attach contextual information.

Responsibilities:

Weather:

- temperature
- precipitation
- conditions

Astronomical:

- sunrise
- sunset
- dawn
- dusk

Media:

- genres
- summaries
- runtime

Synthetic activities:

- evening walks
- outdoor suggestions

Synthetic activities are defined by rule blocks in `config.yaml` rather than in `likes.txt` / `dislikes.txt`, because their triggers are environmental conditions rather than preference concepts.

Example rule schema:

```yaml
synthetic_activities:
  - name: "Evening walk"
    conditions:
      min_temp_f: 55
      max_temp_f: 85
      weather: [clear, partly_cloudy]
      time_window: sunset_minus_1h to sunset_plus_2h
    tags: [outdoor, walking, low_key]
```

When conditions are satisfied, the enrichment layer injects a synthetic EventCandidate into the pipeline.

Synthetic events are processed identically to discovered events — they receive embedding scores, pass through the ranking engine, and appear in the final output.

Output:

Enriched Event objects.

---

# 4.5 Semantic Processing Layer

Purpose:

Transform raw events into personalized recommendations.

This layer contains four stages.

## Stage 1: LLM Extraction

Default model: `gemma4:e4b` (configurable).

`gemma4:e4b` is multimodal. When an event candidate includes an image URL, the image shall be passed alongside the text. The model extracts signal from both.

Input:

Normalized event text (title + description after normalization and deduplication pass 1). This is not raw scraped HTML — it is cleaned, canonical text produced by the normalization layer. Image URL included when available.

Output:

Structured data.

Fields:

- title
- venue
- start_time
- end_time
- tags
- summary (a single sentence describing the event, used as a holistic embedding signal)

Tag generation:

The LLM shall be prompted to generate a minimum of 5 tags per event. This ensures the embedding layer has enough signal to compute a meaningful score. The minimum tag count shall be configurable.

LLM Pass 1 bypass:

If an event already has `tags` populated when it reaches Stage 1, the LLM extraction step shall be skipped for that event. The existing tags and summary are used as-is.

This handles synthetic activities (which arrive with tags pre-populated from their YAML config) without requiring a special flag. It also handles any future source that delivers pre-tagged events.

---

## Stage 2: Embedding Generation

Embedding provider: Ollama (`nomic-embed-text` model).

The embedding component shall expose a single interface: `embed(text: str) -> List[float]`.

The provider (model name, Ollama host) shall be configurable so the embedding backend can be swapped without modifying callers.

Generate vectors for:

- likes.txt (one embedding per line)
- dislikes.txt (one embedding per line)
- event tags (one embedding per tag)
- event summary (one embedding for the full 1-sentence summary)

Preference embedding cache:

Preference files (likes.txt, dislikes.txt) change rarely. Regenerating their embeddings on every run is wasteful.

The system shall cache preference embeddings as follows:

1. Hash likes.txt and dislikes.txt (SHA-256, one hash per file, one hash per line).
2. Compare hashes against stored values in `preference_embeddings_cache`.
3. If hashes match, load stored vectors — skip re-embedding.
4. If hashes differ, regenerate embeddings and persist new vectors + new hashes.

---

## Stage 2b: Semantic Deduplication

A second deduplication pass shall execute after embeddings exist.

This pass compares event description vectors using cosine similarity to catch duplicates that fuzzy string matching (step 5) missed — e.g. the same event described differently across sources.

Threshold for semantic duplicate classification shall be configurable.

---

## Stage 3: Similarity Mapping

Compute semantic distances using cosine similarity.

Exact string matching is prohibited.

Formula (per event):

For each event tag vector `t`:

```
like_sim    = max(cosine(t, l) for l in like_embeddings)
dislike_sim = max(cosine(t, d) for d in dislike_embeddings)

if like_sim > dislike_sim:
    contribution = +like_sim
else:
    contribution = -dislike_sim
```

Specificity wins: the closer match (like or dislike) determines the direction and magnitude.

```
tag_score     = sum(contributions) / len(tags)   # normalized by tag count
summary_score = same formula applied to the summary embedding
base_score    = tag_score + (summary_weight × summary_score)
```

`summary_weight` is configurable (default 0.3). The summary is a supporting signal, not the primary driver.

Final score after multipliers (applied in scoring layer):

```
final_score = base_score × match_multiplier + weather_bonus
```

---

## Stage 4: Match Classification

Assign a preliminary label:

- yes
- maybe
- no

This classification is advisory.

> **Addendum — LLM Pass 2 (post-v1):** An optional Stage 5 was originally designed to sit between similarity scoring and deterministic ranking. It would use the LLM to resolve conflicts where likes and dislikes produce contradictory scores (e.g. an event tagged both "karaoke" and "pub" when the user likes karaoke but dislikes bars). In v1 this conflict resolution is handled deterministically within the scoring layer. LLM Pass 2 is preserved as a future pipeline stage slot; it shall not be implemented until v1 is complete and the deterministic resolver proves insufficient.

---

# 4.6 Scoring Layer

Purpose:

Compute final ranking.

Responsibilities:

- assign rewards
- assign penalties
- resolve conflicts
- calculate scores

Final ordering shall be deterministic.

LLMs shall not participate.

Scoring factors (v1):

| Factor | Direction | Notes |
|---|---|---|
| Like similarity | + | Cosine similarity between event tag vectors and likes.txt embeddings |
| Dislike similarity | − | Cosine similarity between event tag vectors and dislikes.txt embeddings |
| Conflict resolution | varies | When both like and dislike scores are high, the more specific match (higher raw similarity) takes precedence |
| Match classification | multiplier | `yes` amplifies final score, `no` suppresses it, `maybe` is neutral |
| Weather/outdoor alignment | + | Outdoor-tagged events score higher on days with good weather conditions |
| Blocklist | hard exclude | Blocked venues are removed before ranking, not penalized |

Scoring factors (post-v1 candidates):

| Factor | Direction | Notes |
|---|---|---|
| Prior feedback | + / − | Adjust scores based on user ratings of similar past events |
| Novelty | − | Small penalty for recurring events the user has not attended |

Score range and tiers:

The scoring formula produces an unbounded float. Higher is always better. Negative scores are valid.

Scores shall not be normalized relative to the current batch — this would make events discovered across different runs incomparable (an event found today for a date two weeks away cannot be compared against events not yet discovered).

Tier classification (configurable thresholds in `config.yaml`):

```yaml
scoring:
  tiers:
    top_picks_min: 0.5
    worth_considering_min: 0.1
  summary_weight: 0.3
  match_multipliers:
    yes: 1.5
    maybe: 1.0
    no: 0.5
```

Events below `worth_considering_min` are excluded from the default CLI view. Tune thresholds after observing real score distributions.

Domain-scoped preferences:

Preference files support section headers to scope preferences to a domain.

```
[general]
live music
karaoke
board games

[movies]
horror films
independent cinema

[restaurants]
italian cuisine
sushi
```

Lines before the first section header are treated as `[general]`.

The scoring layer shall apply preferences only within the matching domain. A `[movies]` preference is never applied to a concert. A `[restaurants]` preference is never applied to a film.

New domains require no code changes — only a new section header and entries in the preference files.

Movie events carry a `source_type: movie` tag.

The scoring layer shall apply movie-domain preferences (from a `[movies]` section in preference files) only when scoring events with `source_type: movie`.

This prevents genre terms like "horror films" from cross-contaminating scores for non-movie events.

---

# 4.7 Storage Layer

Purpose:

Persist application state.

Storage technology:

SQLite

The storage layer acts as the system hub.

Responsibilities:

- event persistence
- metadata persistence
- logs
- feedback data
- blocklists

Tables:

- `venues` — name, address, coordinates, category, social handles, blocklist flag, discovery source
- `candidate_entities` — handle, state (active/probationary/discarded), mention_count, mention_sources (JSON array of source handles), LLM classification, discovery context, depth
- `event_candidates` — raw scraped events, source, discovered_at, raw_published_at
- `events` — normalized, deduplicated canonical events with enrichment data attached
- `recommendations` — event_id, score, match label, score reasons, run_date
- `preference_embeddings_cache` — file hash per line, embedding vectors, generated_at
- `weather_cache` — daily weather keyed by date + coordinates
- `run_history` — batch run metadata (start time, duration, steps completed, errors)
- `feedback` — user rating on past recommendations, stored for future training
- `blocklist` — venue names and patterns to skip during discovery

Embedding vector storage:

Embedding vectors shall be stored as BLOBs (raw binary float arrays) rather than JSON text.

Rationale: 768 floats × 4 bytes = ~3KB per event as binary vs ~5KB as JSON text. Across one year of retained event data this is a meaningful difference on a disk-constrained VM.

Serialization/deserialization shall be handled by a single utility function used consistently throughout the codebase:

```python
def encode_vector(v: list[float]) -> bytes: ...
def decode_vector(b: bytes) -> list[float]: ...
```

Blocklist source of truth:

`data/blocklist.json` is the human-editable source of truth.

At the start of each batch run, `blocklist.json` is read and its contents are loaded into the `blocklist` table, overwriting any previous values.

The `blocklist` DB table is derived and ephemeral — it shall never be written to directly by any process other than this load step.

The storage layer performs no business logic.

---

# 4.8 Presentation Layer

Purpose:

Display information.

Current interface:

CLI

Future interfaces:

- FastAPI
- PWA
- desktop UI

Presentation code shall not access ingestion systems directly.

Presentation code shall only consume stored data.

---

# 5. Data Contracts

The following contracts are canonical.

Modules shall communicate exclusively using these structures.

---

## 5.1 EventCandidate

Represents raw discovered information.

Fields:

id

source

source_type (e.g. instagram, facebook, cinema_veezi, amc, synthetic)

url (original post or listing URL)

image_url (optional — populated when source provides an image; passed to multimodal LLM)

raw_published_at (optional — timestamp of the original post; None for movie schedules; used for lookback window filtering)

title

description

venue

location

start_time

end_time

discovered_at

---

## 5.2 Event

Represents a normalized event.

Fields:

event_id

source_event_candidates (list — preserves all sources that contributed to this event)

source_type

url

image_url (optional)

title

venue

description

location

start_time (timezone-aware)

end_time (timezone-aware)

tags (list — extracted by LLM Pass 1, minimum 5)

summary (1-sentence description extracted by LLM Pass 1)

tag_embeddings (list of vectors, one per tag)

summary_embedding (single vector)

weather

astronomical_data

metadata

---

## 5.3 Recommendation

Represents a ranked event.

Fields:

event_id

run_date (which batch produced this recommendation)

score (unbounded float — higher is always better, may be negative)

tier (derived: "top_pick" | "worth_considering" | "excluded")

match ("yes" | "maybe" | "no" — from Stage 4 classification)

reasons[] (list of structured reason objects)

Reason schema:

```json
{
  "factor": "like_similarity",
  "tag": "karaoke",
  "matched_preference": "karaoke night",
  "similarity": 0.87,
  "contribution": 0.87,
  "direction": "positive"
}
```

Supported factor values: `like_similarity`, `dislike_similarity`, `match_classification`, `weather_bonus`, `domain_mismatch`.

---

## 5.4 UserPreference

Fields:

type ("like" | "dislike")

domain ("general" | "movies" | "restaurants" | ... — extensible, no code changes required to add a domain)

text (the preference line as written in the file)

embedding (BLOB — vector representation of text)

Preference file format:

```
[general]
live music
karaoke
board games

[movies]
horror films
independent cinema

[restaurants]
italian cuisine
sushi
```

Lines before the first section header are treated as `[general]`. Each line becomes one UserPreference record. The `domain` field is set from the active section header.

---

## 5.5 EnvironmentalContext

Fields:

temperature

conditions

sunrise

sunset

dawn

dusk

---

# 6. Repository Layout

```text
project_root/

config/
  config.example.yaml     ← checked in (template, no personal data)
  config.yaml             ← gitignored (contains personal location)

data/
  likes.txt               ← gitignored (personal preferences)
  dislikes.txt            ← gitignored (personal preferences)
  blocklist.json          ← gitignored (personal choices)
  seeds.yaml              ← checked in (public social handles only)
  likes.example.txt       ← checked in (example content)
  dislikes.example.txt    ← checked in (example content)
  blocklist.example.json  ← checked in (empty array [])

database/
  event_hub.db            ← gitignored (personal data, year of events)

src/
  __init__.py
  ingestion/
  normalization/
  enrichment/
  processing/
  scoring/
  storage/
  presentation/
  models/
  utils/

tests/

logs/                     ← gitignored

CLAUDE.md
.env.example              ← checked in (template, no real keys)
.env                      ← gitignored (real secrets)
.gitignore
pyproject.toml
README.md
```

.gitignore must include at minimum:

```
.env
config/config.yaml
database/
logs/
data/likes.txt
data/dislikes.txt
data/blocklist.json
__pycache__/
*.pyc
.mypy_cache/
.pytest_cache/
```

## 6.1 Config Schema

`config/config.example.yaml` — known fields at v1:

```yaml
location:
  latitude: 0.0
  longitude: 0.0
  postal_code: "00000"
  search_radius_miles: 10

scraping:
  lookback_days: 30
  max_discovery_depth: 2
  candidate_promotion_threshold: 3   # mentions from distinct trusted sources

scoring:
  tiers:
    top_picks_min: 0.5
    worth_considering_min: 0.1
  summary_weight: 0.3
  match_multipliers:
    yes: 1.5
    maybe: 1.0
    no: 0.5
  min_tags_per_event: 5

deduplication:
  fuzzy_title_threshold: 0.85
  time_window_hours: 2
  semantic_threshold: 0.92

models:
  llm_extraction: "gemma4:e4b"
  llm_disambiguation: "gemma4:e2b"
  embeddings: "nomic-embed-text"

weather:
  provider: "open-meteo"

scheduling:
  run_window_start: "02:00"
  run_window_end: "04:00"
  jitter_minutes: 30

data_retention_days: 365

synthetic_activities:
  - name: "Evening walk"
    conditions:
      min_temp_f: 55
      max_temp_f: 85
      weather: [clear, partly_cloudy]
      time_window: "sunset_minus_1h to sunset_plus_2h"
    tags: [outdoor, walking, low_key]
    summary: "A pleasant evening walk"
```

## 6.2 Blocklist Format

`data/blocklist.json` — flat array of venue names or handles:

```json
["@oneilsbar", "The Sketchy Pub", "@anothervenue"]
```

Handles (`@...`) are matched exactly. Venue names are matched with fuzzy similarity against discovered venue names. Both may appear in the same file.

---

# 7. Scheduling Architecture

The scheduler orchestrates all background processing.

Responsibilities:

- execute daily runs
- apply runtime jitter
- monitor failures
- write logs

Scheduling mechanism:

The scheduler shall be triggered by a system cron job on the always-on VM.

The `what-do-run-batch` CLI command is the entry point for both cron and manual execution.

Example crontab entry:

```
0 2 * * * /path/to/venv/bin/what-do-run-batch >> /path/to/logs/cron.log 2>&1
```

Manual execution: `what-do-run-batch` runs the full pipeline immediately.

Default CLI behaviour:

`what-do` with no arguments displays today's events sorted by score, grouped into Top Picks and Worth Considering sections. Events with tier `excluded` are hidden by default.

Required secrets (`.env.example`):

```
APIFY_API_KEY=
TMDB_API_KEY=
AMC_API_KEY=
OLLAMA_HOST=http://localhost:11434
```

All secrets loaded from environment or `.env` file. No secret shall be hardcoded.

The scheduler shall not contain business logic.

---

# 8. Logging Architecture

The application shall emit structured logs.

Required fields:

timestamp

component

severity

duration_ms

message

Supported levels:

DEBUG

INFO

WARNING

ERROR

---

# 9. Failure Handling

Provider failures shall not terminate a batch run.

Failures shall:

- be logged
- be skipped
- continue processing

Invalid records shall be discarded safely.

Malformed data shall not propagate downstream.

---

# 10. Future Expansion Boundaries

The architecture is intentionally designed to support future capabilities.

Examples:

Additional providers:

- Reddit
- Facebook events
- Meetup
- Eventbrite

Additional interfaces:

- FastAPI
- PWA
- Mobile app

Additional models:

- Recommendation classifiers
- Personalized ranking models

These additions shall not require architectural rewrites.

---

# 11. Agent Operating Rules

These rules exist specifically for AI coding agents.

The agent SHALL:

- write tests before implementation
- implement the minimum code necessary to satisfy tests
- refactor only after tests pass
- stop if a phase's tests fail

The agent SHALL NOT:

- hardcode geographic locations
- hardcode API keys
- perform network operations during CLI requests
- bypass data contracts
- use LLMs for deterministic calculations
- introduce global mutable state
- tightly couple providers to presentation code
- call `datetime.now()` or `date.today()` directly (use injectable `get_now` parameter)
- store embedding vectors as float64 (use float32 via `encode_vector`/`decode_vector`)
- introduce asyncio or concurrency in v1

Every new code module shall:

- have tests
- be independently executable
- expose typed interfaces
- remain replaceable

The agent shall not proceed to the next implementation phase without explicit approval.

## 11.1 Concurrency Boundaries (post-v1 reference)

v1 is deliberately single-threaded. The LLM calls dominate runtime; concurrency adds
complexity with little gain at this scale.

When parallelism is added in a future version, the natural boundaries are:

Parallelizable (independent per-item work):
- Scraping multiple source adapters simultaneously
- Embedding multiple events simultaneously
- Weather/solar enrichment per event

Must remain sequential (set operations):
- Dedup Pass 1 — needs all events in memory
- Dedup Pass 2 — needs all embeddings in memory
- Final tier assignment — needs all scores for threshold comparison

The pipeline stage interface (`process(events) → events`) is already compatible with
future executor-based parallelism. No architectural change required — only the orchestrator
needs updating.

## 11.2 Vector Storage Convention

Embedding vectors are stored as BLOB columns in SQLite.

Precision: float32 (not float64). Cosine similarity does not benefit from float64 precision,
and float32 halves storage: 768 × 4 bytes = 3KB vs 768 × 8 bytes = 6KB per vector.

All vector serialization goes through two utility functions in `src/utils/vectors.py`:

```python
def encode_vector(v: list[float]) -> bytes: ...   # float32 BLOB
def decode_vector(b: bytes) -> list[float]: ...   # back to list[float]
```

These functions shall be the only place in the codebase that touches vector binary format.

