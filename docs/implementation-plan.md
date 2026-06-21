# Implementation Plan

# Project: Local Event Intelligence Hub

Version: 0.1

---

# 1. Development Methodology

The project shall use a Red-Green-Refactor Test Driven Development workflow.

Every phase follows the same process.

```text
PLAN

↓

WRITE TESTS

↓

VERIFY TESTS FAIL (RED)

↓

IMPLEMENT MINIMUM CODE

↓

VERIFY TESTS PASS (GREEN)

↓

REFACTOR

↓

GO / STOP DECISION
```

No subsequent phase may begin until the previous phase reaches GO status.

---

# 2. General Rules

For every phase:

The AI agent SHALL:

- summarize the phase goal
- identify dependencies
- write tests first
- verify tests fail
- implement the minimum code necessary
- run tests
- refactor after tests pass

The AI agent SHALL NOT:

- implement future phases early
- skip writing tests
- bypass failing tests
- modify unrelated modules

## 2.1 Tooling

Language: Python 3.11+

Test framework: pytest

Type checking: mypy

Dependency management: pyproject.toml

Python package name: `what_do` (hyphens not valid in Python package names; the CLI command is `what-do`)

Entry points defined in pyproject.toml:

```toml
[project]
name = "what-do"

[project.scripts]
what-do = "src.presentation.cli:main"
what-do-run-batch = "src.scheduler:run"

[tool.pytest.ini_options]
pythonpath = ["."]
```

The `pythonpath = ["."]` setting is required. Without it, `src.*` imports will fail with `ModuleNotFoundError` when running pytest from the project root. All `src/` subdirectories must contain `__init__.py`.

## 2.2 Test Isolation Rule

Tests shall not make real network calls.

All external services must be mockable at the provider boundary:

- Ollama (LLM and embeddings)
- Open-Meteo (weather)
- Apify / Picuki / Dumpor (scraping)
- Veezi / AMC (movie schedules)
- TMDb (media metadata)

Each provider shall be injectable (passed in, not imported directly) so tests can substitute a fake implementation.

Violation of this rule means a test that passes locally may fail in CI or on a fresh machine — treat it as a bug.

## 2.3 Injectable Time Rule

No module shall call `datetime.now()` or `date.today()` directly.

All time-sensitive logic shall accept a `get_now: Callable[[], datetime]` parameter.

This applies to: lookback window calculations, event filtering, solar data lookup, synthetic activity condition checks, scheduler timing, log timestamps.

Example:

```python
def filter_recent_events(events, get_now=datetime.now):
    cutoff = get_now() - timedelta(days=cfg.lookback_days)
    return [e for e in events if e.discovered_at >= cutoff]
```

Tests pass a fixed `get_now` to control time deterministically. Failure to follow this pattern makes time-dependent tests flaky and untestable — it is expensive to retrofit later.

## 2.3 Integration Gate

At the end of every phase, a smoke test shall verify the handoff to the next phase works end-to-end.

Smoke tests may use real local services (Ollama, SQLite) but shall not make external network calls.

The smoke test is part of the Green Criteria. A phase is not GO until its smoke test passes.

---

# 3. Phase 0 - Project Skeleton

## Goal

Create the foundational repository structure and project scaffolding.

No business logic shall be implemented.

This phase only establishes infrastructure.

---

## Planning Discussion

Every future phase depends on a stable foundation.

This phase intentionally avoids complexity.

The goal is to create a deterministic workspace.

---

## Deliverables

Repository structure:

```text
config/
data/
database/
logs/

src/

  ingestion/
  normalization/
  enrichment/
  processing/
  scoring/
  storage/
  presentation/

  models/
  utils/

scripts/

tests/
```

Files:

```text
.env.example

config.yaml

likes.txt

dislikes.txt

blocklist.json

seeds.yaml

README.md
```

---

## Red Tests

Directory structure:

- all required directories exist
- all `src/` subdirectories contain `__init__.py`

Required files:

- `.env.example` exists and lists all required secret keys: `APIFY_API_KEY`, `TMDB_API_KEY`, `AMC_API_KEY`, `OLLAMA_HOST`
- `config.yaml` exists
- `data/likes.txt` exists
- `data/dislikes.txt` exists
- `data/blocklist.json` exists and is valid JSON
- `data/seeds.yaml` exists and is valid YAML with `handles:` and `venues:` keys
- `README.md` exists
- `pyproject.toml` exists with `what-do` and `what-do-run-batch` entry points defined

Configuration:

- valid config loads without error
- config missing required geographic field raises a clear error
- `OLLAMA_HOST` defaults to `http://localhost:11434` if not set in environment

Environment:

- `.env` file values load into environment
- missing optional keys do not raise errors

---

## Green Criteria

All tests pass.

Project starts successfully.

No runtime exceptions occur.

---

## Smoke Test

```
python -c "from src.config import load_config; cfg = load_config(); print(cfg.location.latitude)"
```

Prints the configured latitude without error.

---

## Go / No-Go

GO if:

- all tests pass
- smoke test passes

STOP otherwise.

---

# 4. Phase 1 - Configuration & Database Foundation

## Goal

Create the core application configuration and SQLite persistence layer.

No ingestion logic shall exist yet.

---

## Planning Discussion

Every component will depend on configuration and storage.

These systems must stabilize first.

---

## Deliverables

Configuration loader.

SQLite initialization.

Schema migrations.

Logging initialization.

---

## Red Tests

Configuration:

- valid config loads
- config missing required field raises a descriptive error
- latitude validated: must be numeric, -90 to 90
- longitude validated: must be numeric, -180 to 180
- search_radius_miles validated: must be positive
- timezone correctly derived from lat/lng (not hardcoded)
- lookback_days defaults to 30 if not set

Database:

- all expected tables exist after init: `venues`, `candidate_entities`, `event_candidates`, `events`, `recommendations`, `preference_embeddings_cache`, `weather_cache`, `run_history`, `feedback`, `blocklist`
- each table has correct column names and types
- migrations are idempotent (running init twice does not error or duplicate)

Logging:

- log file created in `logs/`
- each log entry contains: timestamp, component, severity, duration_ms, message
- DEBUG messages absent from INFO-level log output
- log level configurable from `config.yaml`

Vector utilities:

- `encode_vector(v)` → bytes
- `decode_vector(encode_vector(v))` == v (lossless round-trip)
- round-trip preserves precision for a 768-dimension vector

---

## Green Criteria

Database initializes successfully with all tables.

Configuration loads and validates successfully.

Logs generate with all required fields.

Vector encode/decode is lossless.

---

## Smoke Test

```
python -c "
from src.storage import init_db
from src.utils.logging import get_logger
init_db()
log = get_logger('smoke')
log.info('Phase 1 smoke test', component='smoke', duration_ms=0)
print('OK')
"
```

DB created. Log entry written. No exceptions.

---

## Go / No-Go

GO if all tests pass and smoke test passes.

STOP otherwise.

---

# 5. Phase 2 - Venue Discovery

## Goal

Discover venues within geographic boundaries.

No event scraping yet.

---

## Planning Discussion

Venue discovery establishes the ecosystem.

It is intentionally separated from scraping.

---

## Deliverables

Venue provider abstraction.

Venue discovery service.

Venue persistence.

---

## Red Tests

Discovery:

- venue within configured radius → discovered and persisted
- venue outside configured radius → excluded
- `seeds.yaml` handles and venue entries used as starting points for discovery
- discovered venue stored with all schema fields: name, address, coordinates, category, social handles, blocklist flag, discovery source

Deduplication:

- same venue discovered from two sources → one record in DB
- duplicate venue: existing record updated, not duplicated

Blocklist:

- `blocklist.json` loaded from file at run start (not from DB, not hardcoded)
- blocked venue → skipped during discovery
- blocked venue skip written to log with venue name and reason

Provider abstraction:

- venue provider interface respected: swapping a mock provider requires no changes to the discovery service
- provider failure → logged and skipped, discovery continues with remaining providers

---

## Green Criteria

Venue discovery functions correctly.

No duplicates exist.

Blocklist enforced from file.

---

## Smoke Test

Add one handle from `seeds.yaml` to a mock provider. Run venue discovery. Confirm one venue record exists in DB with all fields populated.

---

## Go / No-Go

GO if all tests pass and smoke test passes.

STOP otherwise.

---

# 6. Phase 3 - Event Ingestion

## Goal

Discover events from providers.

No scoring shall exist.

---

## Planning Discussion

This phase focuses entirely on acquiring raw information.

No semantic processing occurs.

---

## Deliverables

Provider adapters.

Recursive entity discovery.

Raw event persistence.

---

## Red Tests

Provider interface:

- provider returns `EventCandidate` objects with all required fields: `id`, `source`, `source_type`, `url`, `image_url`, `raw_published_at`, `title`, `description`, `venue`, `location`, `start_time`, `end_time`, `discovered_at`
- provider abstraction respected: mock provider substitutable with no changes to ingestion service

Failover chain:

- primary source fails → secondary source attempted → events still returned
- all sources fail → empty result returned, error logged, pipeline continues

Scraping:

- events with `raw_published_at` within lookback window (default 30 days) → ingested
- events with `raw_published_at` outside lookback window → discarded
- events with `raw_published_at = None` (movie schedules) → always ingested regardless of lookback
- lookback window reads from config (not hardcoded)
- uses injected `get_now` (never calls `datetime.now()` directly)

Movie adapters:

- Cinema Salem adapter returns `EventCandidates` with `source_type='cinema_veezi'`
- AMC adapter returns `EventCandidates` with `source_type='amc'`

Schema:

- after `init_db()`, `event_candidates` table has `raw_published_at` column
- after `init_db()`, `candidate_entities` table has `depth` and `mention_sources` columns

Seed loading:

- handles from `seeds.yaml` upserted into `candidate_entities` as `active`, `depth=0`
- seed load is idempotent (running twice produces no duplicates)
- handle in `candidate_entities` as `probationary` that appears in seeds → promoted to `active`

Recursive discovery:

- `@handle` in post caption → added to `candidate_entities` as `probationary`, `depth = parent_depth + 1`
- `@handle` already in `candidate_entities` → `mention_count` incremented, source added to `mention_sources` (if not already present)
- same source mentioning same handle twice → count incremented only once (idempotent per source)
- handle at `max_depth` → not stored; skipped and logged
- blocklisted handle → not stored

Disambiguation step (3a):

- new `probationary` handles classified as `venue` or `person` by injected `DisambiguationProvider`
- handles classified as `person` → state set to `discarded`
- handles classified as `venue` → remain `probationary`
- already-classified handles (state ≠ `probationary`) → skipped (provider not called)
- provider failure → handle stays `probationary`, failure logged, pipeline continues

Promotion:

- handle with `mention_count >= candidate_promotion_threshold` AND a seed source in `mention_sources` → promoted to `active`
- handle meeting count threshold but no seed source in `mention_sources` → remains `probationary`
- threshold read from config, not hardcoded

Seed management CLI:

- `what-do add-source @newhandle` → handle written to `seeds.yaml`
- `what-do add-source @newhandle` run twice → no duplicate in `seeds.yaml`
- `what-do add-source --venue "Name" --address "123 Main St"` → written to `seeds.yaml` venues list

Error handling:

- malformed record (title, description, and start_time all absent) → discarded
- discard written to log with source adapter name and reason
- one malformed record does not stop ingestion of remaining records

---

## Green Criteria

Events persist successfully with all fields.

Failures do not terminate execution.

Lookback window enforced correctly (None raw_published_at bypasses filter).

Handles discovered, classified, and promoted correctly.

---

## Smoke Test

Run ingestion with a mock social adapter returning 3 valid events and one malformed record (all key fields absent). Confirm 3 `EventCandidates` in DB, 1 discard logged. Run with mock adapter raising an exception. Confirm pipeline completes using remaining adapters.

---

## Go / No-Go

GO if all tests pass and smoke test passes.

STOP otherwise.

---

# 7. Phase 4 - Normalization & Deduplication

## Goal

Convert raw events into canonical event objects.

---

## Planning Discussion

This phase prepares data for semantic analysis.

Garbage shall not propagate downstream.

---

## Deliverables

Normalization engine.

Deduplication engine.

Canonical event objects.

---

## Red Tests

Normalization:

- all timestamps timezone-aware after normalization (timezone derived from config lat/lng)
- venue name normalized consistently ("The Vault" and "vault, the" → same canonical form)
- text fields stripped of excess whitespace and encoding artifacts

Deduplication Pass 1 (fuzzy, pre-embedding):

- same title + same venue → one canonical event
- similar title + same venue + start times within 2 hours → duplicate
- same title + different venue → not duplicate
- duplicate merge: most complete record wins (event with venue populated wins over one without)
- duplicate merge: `source_event_candidates` list contains attribution from all contributing records
- dedup similarity threshold reads from config (not hardcoded)

Malformed records:

- record missing both title and start_time → discarded and logged with reason
- record missing only title → flagged in metadata, not discarded
- record missing only start_time → flagged in metadata, not discarded
- discard logged with source and specific missing field

---

## Green Criteria

Canonical events created with all fields.

Duplicates merged with source attribution preserved.

Malformed records handled without halting pipeline.

---

## Smoke Test

Feed 4 mock `EventCandidates`: 2 identical (different sources), 1 unique, 1 missing both title and start_time. Confirm 2 `Events` in DB (merged pair + unique), 1 discard in logs, merged event has both source attributions.

---

## Go / No-Go

GO if all tests pass and smoke test passes.

STOP otherwise.

---

# 8. Phase 5 - Environmental Enrichment

## Goal

Attach environmental context.

---

## Planning Discussion

Environmental context should be isolated from recommendation logic.

---

## Deliverables

Weather integration.

Astronomical calculations.

Synthetic activities.

---

## Red Tests

Weather:

- weather fetched for event date, not batch run date
- event happening tomorrow → tomorrow's forecast attached
- event >16 days ahead → weather field is None (graceful, not an error)
- weather provider failure → event retained with weather=None, failure logged

Astronomical:

- sunrise, sunset, dawn, dusk calculated for event date at configured coordinates
- values are timezone-aware datetimes

Synthetic activities:

- synthetic activity generated when all conditions in config rule are met (temp, weather, time window)
- synthetic activity NOT generated when one condition fails
- generated synthetic event has `source_type='synthetic'`
- generated synthetic event has `tags` and `summary` pre-populated from config rule
- generated synthetic event is injected as a pre-structured `Event` (not `EventCandidate`)
- synthetic activity conditions read from `config.yaml` (not hardcoded)

Movie enrichment:

- movie event → TMDb metadata attached: genre, runtime, summary
- TMDb lookup failure → event retained with metadata=None, failure logged

---

## Green Criteria

Environmental data attached per event date.

Synthetic activities generated and injected correctly.

Failures degrade gracefully.

---

## Smoke Test

Run enrichment on a mock event dated tomorrow. Confirm weather and solar data attached. Update `config.yaml` synthetic rule to match current conditions. Confirm one synthetic event injected with pre-populated tags.

---

## Go / No-Go

GO if all tests pass and smoke test passes.

STOP otherwise.

---

# 9. Phase 6 - LLM Extraction Pipeline

## Goal

Convert messy text into structured data.

No ranking shall occur yet.

---

## Planning Discussion

LLMs are extraction tools.

They are not recommendation engines.

---

## Deliverables

LLM Pass 1.

Tag extraction.

Structured JSON output.

---

## Red Tests

Extraction:

- title extracted from raw event text
- start_time extracted and parsed correctly
- tags extracted as a list of strings
- summary extracted as a single sentence
- at minimum 5 tags generated (output with fewer tags rejected)
- minimum tag count reads from config (not hardcoded)

Multimodal:

- event with `image_url` populated → image passed to LLM alongside text
- event without `image_url` → text-only call made (no error)

Bypass:

- event with `tags` already populated → LLM NOT called (verified by asserting mock LLM receives no call)
- event with `tags` already populated → existing tags and summary pass through unchanged

Schema enforcement:

- malformed LLM output (invalid JSON) → event flagged, pipeline continues, error logged
- LLM output missing required field → treated as malformed
- LLM timeout or error → event flagged, pipeline continues, error logged

---

## Green Criteria

LLM outputs valid structures.

Bypass works for pre-tagged events.

Failures degrade gracefully without halting pipeline.

---

## Smoke Test

Run LLM extraction on one sample Instagram caption using real Ollama (`gemma4:e4b`). Confirm output is valid JSON with title, start_time, ≥5 tags, summary. Run same event again with tags pre-populated. Confirm Ollama not called second time.

---

## Go / No-Go

GO if all tests pass and smoke test passes.

STOP otherwise.

> **Addendum — LLM Pass 2:** A second LLM pass (fuzzy conflict resolution between like/dislike similarity scores) was originally scoped for this phase. It has been deferred to post-v1. In v1 the scoring layer handles conflicts deterministically. The pipeline slot is reserved. Do not implement LLM Pass 2 until v1 is complete.

> **Addendum — OllamaDisambiguationProvider + discovery_context fix:** Phase 3 built `DisambiguationProvider` ABC and `DisambiguationStep` but left the real Ollama implementation as a stub. Phase 6 must implement `OllamaDisambiguationProvider` (using `gemma4:e2b`) and fix the known gap where `HandleExtractor` does not populate `discovery_context`. Specifically: (1) implement `OllamaDisambiguationProvider.classify(handle, context)`; (2) update `HandleExtractor._upsert()` to store the surrounding caption text in `discovery_context` (truncated, e.g. 300 chars); (3) update `HandleExtractor.process()` to pass the source text through. Add red tests for all three before implementing. See `docs/decisions.md` — "Known gap: discovery_context not populated by HandleExtractor".

---

# 10. Phase 7 - Semantic Matching Engine

## Goal

Build the embedding and similarity engine.

---

## Planning Discussion

Preference matching shall be mathematical.

Exact text matching is prohibited.

---

## Deliverables

Embedding engine.

Similarity engine.

Match classifier.

---

## Red Tests

Embedding:

- `embed(text)` returns a list of floats with length 768
- vectors stored as BLOBs in DB
- `encode_vector` / `decode_vector` round-trip is lossless (no precision loss)
- one embedding generated per tag; one generated for summary

Preference cache:

- preference files unchanged → embeddings NOT regenerated (Ollama not called)
- preference file edited → stale hash detected → embeddings regenerated and new hashes stored
- cache correctly handles likes.txt and dislikes.txt independently

Similarity:

- "karaoke night" scores positively against likes containing "karaoke"
- "sports bar" scores negatively against dislikes containing "bars"
- unrelated event scores near zero against both likes and dislikes
- specificity wins: event tag closer to a like than a dislike → positive contribution
- specificity wins: event tag closer to a dislike than a like → negative contribution
- tag score normalized by tag count (event with 10 tags does not automatically outscore event with 3)
- summary embedding weighted at configurable rate (default 0.3) relative to tag score

Domain scoping:

- `[movies]` preference not applied when scoring a `source_type='instagram'` event
- `[movies]` preference applied when scoring a `source_type='cinema_veezi'` event
- `[general]` preferences applied to all events regardless of source_type

Reason objects:

- each scoring contribution produces a `Reason` object with: `factor`, `tag`, `matched_preference`, `similarity`, `contribution`, `direction`
- reasons list attached to event after similarity stage

Semantic deduplication (Pass 2):

- two events with identical meaning but different wording → merged
- semantic dedup threshold reads from config
- events already deduplicated in Pass 1 not re-evaluated

---

## Green Criteria

Similarity scores behave correctly and predictably.

Domain scoping enforced.

Reason objects populated with correct schema.

---

## Smoke Test

Load real `likes.txt` and `dislikes.txt`. Run embedding on 5 sample events. Confirm a "karaoke night" event scores positively, a "trivia at a sports bar" scores negatively. Confirm vectors stored as BLOBs. Clear cache, run again — confirm Ollama called. Run again without changing files — confirm Ollama not called.

---

## Go / No-Go

GO if all tests pass and smoke test passes.

STOP otherwise.

---

# 11. Phase 8 - Deterministic Ranking Engine

## Goal

Calculate final recommendation ordering.

---

## Planning Discussion

This phase intentionally avoids LLM involvement.

Ranking shall always be reproducible.

---

## Deliverables

Scoring engine.

Conflict resolution.

Recommendation ordering.

---

## Red Tests

Scoring:

- event with high like similarity scores higher than event with low like similarity
- event with high dislike similarity scores lower than event with low dislike similarity
- event close to both a like and a dislike → specificity wins (closer match determines direction)
- identical inputs → identical scores on repeated runs (determinism)
- match multiplier applied: `yes` produces higher score than `maybe` from same base; `no` produces lower
- match multipliers read from config (not hardcoded)
- weather bonus applied to event tagged `outdoor` on a clear-weather day
- weather bonus absent on a rainy day for same event
- blocklisted venue → event hard-excluded before ranking (never appears in output)

Tiers:

- score above `top_picks_min` threshold → tier = `top_pick`
- score between `worth_considering_min` and `top_picks_min` → tier = `worth_considering`
- score below `worth_considering_min` → tier = `excluded`
- tier thresholds read from `config.yaml` (not hardcoded)

Output:

- each `Recommendation` has `run_date` stamped with current batch date
- `reasons[]` populated with all contributing factors
- domain-scoped scoring: `[movies]` preferences only applied to movie events

---

## Green Criteria

Ranking is deterministic.

Tiers correctly classified from config thresholds.

All score factors applied correctly.

---

## Smoke Test

Score 10 mock events with known similarity values. Confirm ordering matches expected ranking. Run scorer twice — confirm output is byte-identical. Confirm blocklisted venue absent from output. Confirm tier assignments match configured thresholds.

---

## Go / No-Go

GO if all tests pass and smoke test passes.

STOP otherwise.

---

# 12. Phase 9 - CLI Interface

## Goal

Create the interactive user experience.

---

## Planning Discussion

The CLI shall only read precomputed data.

No heavy operations occur here.

---

## Deliverables

CLI.

Time filters.

Raw mode.

Recommendation view.

---

## Red Tests

Default behaviour:

- `what-do` with no args → shows today's events only, sorted by score
- output displays Top Picks and Worth Considering as separate sections
- score reasons displayed per event
- excluded events do not appear in default output

Filtering:

- `--time 20:30-23:30` → only events overlapping that window returned
- `--after-sunset` → only events starting after today's sunset returned
- `--raw` → all events returned unfiltered, scoring bypassed

Seed management:

- `what-do add-source @handle` → handle written to `seeds.yaml`
- `what-do add-source @handle` run twice → no duplicate in `seeds.yaml`
- `what-do add-source @handle` with existing handle → user-friendly message, no error

Performance:

- `what-do` returns in under 1 second (timed assertion)
- no network calls made during any CLI invocation (verified with mock/intercept)
- no LLM calls made during any CLI invocation

---

## Green Criteria

CLI is instantaneous.

No network or LLM calls occur.

Tiers and reasons visible in output.

---

## Smoke Test

Populate DB with 10 precomputed mock recommendations spanning two days. Run `what-do` — confirm only today's events shown, response time < 1s. Run `what-do --raw` — confirm all events shown. Run `what-do add-source @smoketest` — confirm entry in `seeds.yaml`.

---

## Go / No-Go

GO if all tests pass and smoke test passes.

STOP otherwise.

---

# 13. Phase 10 - Maintenance Utilities

## Goal

Build user maintenance tools.

---

## Planning Discussion

These tools improve long-term recommendation quality.

They are intentionally deferred.

---

## Deliverables

Preference linter.

Feedback logger.

Blocklist manager.

---

## Red Tests

Preference manager:

- domain section headers (`[general]`, `[movies]`, etc.) parsed correctly from preference files
- adding a preference via tool preserves existing sections and entries
- preference added under correct domain header
- preference file valid after edit (no corruption)

Feedback:

- `what-do feedback <event_id> good` → feedback stored with event reference
- `what-do feedback <event_id> bad` → stored correctly
- `what-do feedback <event_id> skip` → stored correctly
- feedback record schema: event_id, rating, submitted_at
- feedback for unknown event_id → user-friendly error, no crash

Blocklist manager:

- `what-do block "Venue Name"` → entry written to `blocklist.json`
- `what-do block "Venue Name"` twice → no duplicate in `blocklist.json`
- `blocklist.json` valid JSON after edit
- blocklist change takes effect on next batch run (file re-read at run start)

---

## Green Criteria

Utilities function independently.

File integrity maintained after edits.

---

## Smoke Test

Add a preference via tool. Confirm `likes.txt` updated with correct section header. Submit feedback for a known event_id. Confirm feedback record in DB. Block a venue. Confirm `blocklist.json` updated. Run batch. Confirm blocked venue excluded.

---

## Go / No-Go

GO if all tests pass and smoke test passes.

STOP otherwise.

---

# 14. Phase 11 - Hardening & Reliability

## Goal

Prepare the system for long-term unattended execution.

---

## Planning Discussion

The system should be able to run for months without intervention.

---

## Deliverables

Retry logic.

Backoff.

Structured logging.

Scheduling.

Performance improvements.

---

## Red Tests

Retry and backoff:

- provider failure → retried with exponential backoff
- second retry waits measurably longer than first retry
- after max retries exceeded → failure logged, pipeline continues with remaining providers
- rate limit response (HTTP 429) → triggers backoff, not immediate failure

Scheduler:

- `what-do-run-batch` command triggers full pipeline manually
- scheduler fires within configured daily window
- scheduler applies configurable jitter (run time varies within window)
- scheduler records start time, end time, and outcome in `run_history`
- scheduler failure logged with full error details

Data retention:

- events older than 365 days pruned on each run
- pruning respects `event.discovered_at` field
- pruning logged with count of records removed
- retention window configurable (not hardcoded to 365)

Logging:

- every log entry contains all required fields: timestamp, component, severity, duration_ms, message
- log rotation configured (logs do not grow unbounded)
- structured log output parseable as JSON

---

## Green Criteria

Daily execution succeeds repeatedly.

Failures degrade gracefully with full logging.

Old data pruned automatically.

---

## Smoke Test

Run `what-do-run-batch` manually. Confirm `run_history` record created with correct timestamps. Inject a mock provider that fails on the first call. Confirm retry logged. Confirm pipeline completes using remaining providers. Insert a mock event with `discovered_at` > 365 days ago. Run batch. Confirm old event pruned.

---

## Go / No-Go

GO if all tests pass and smoke test passes.

STOP otherwise.

---

# 15. Definition of Done

The project is considered complete when:

✓ All implementation phases are GO.

✓ No phase contains failing tests.

✓ The CLI produces recommendations without performing network requests.

✓ Recommendation ordering is deterministic.

✓ All components remain decoupled.

✓ The repository is safe for public GitHub distribution.

✓ The system can execute unattended on a daily schedule.

✓ Future UI layers can be added without architectural rewrites.

