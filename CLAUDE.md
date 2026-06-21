# Local Event Intelligence Hub — Project Context for Claude

## What this is

A local-first batch processing system that discovers, aggregates, and ranks local events and
activities from public social media sources (Instagram, Facebook) and movie/theater schedules.

Runs as an overnight batch job on an always-on VM. Builds and maintains a SQLite database of
enriched, deduplicated, scored events. The CLI answers "what should we do tonight?" instantly
using only precomputed data — no network calls at query time.

See `docs/requirements.md`, `docs/high-level-design.md`, and `docs/implementation-plan.md`
for the full specification.

---

## Tech stack

- Python 3.11+, pytest, mypy, pyproject.toml
- Ollama — LLM extraction (`gemma4:e4b`), disambiguation (`gemma4:e2b`), embeddings (`nomic-embed-text`)
- SQLite — all storage, embedding vectors stored as BLOB (float32)
- Open-Meteo — weather (free, no API key)
- Apify → Picuki → Dumpor — social media failover chain
- Veezi/Vista API — Cinema Salem schedules
- AMC Showtime API — AMC schedules
- TMDb API — movie metadata enrichment

---

## Key architecture decisions

**Background-first.** All heavy work (LLM, embeddings, scraping, enrichment) happens in the
overnight batch. The CLI reads only precomputed data from SQLite. No LLM or network calls
during interactive use.

**LLMs are extraction tools only.** LLM Pass 1 extracts structured data (title, time, tags,
summary) from messy event text. Final ranking is deterministic — LLMs do not determine order.

**Pluggable pipeline.** Two interface contracts:
- Source adapter: `fetch() → List[EventCandidate]`
- Pipeline stage: `process(events: List[Event]) → List[Event]`
Adding a new source or stage = implement the interface and register it. No other changes.

**Specificity wins in scoring.** For each event tag, compute max cosine similarity against
all likes and all dislikes. Whichever is higher (more specific match) determines direction.

**Domain-scoped preferences.** `likes.txt` and `dislikes.txt` support section headers:
`[general]`, `[movies]`, `[restaurants]`. Domain preferences only apply to events with a
matching `source_type`. Lines before the first header are `[general]`.

**Set operations vs per-event.** Dedup passes need all events in memory. Everything else
(normalization, enrichment, LLM, embedding, similarity, scoring) is per-event and can stream.

---

## Scoring formula

```
for each event tag t:
    like_sim    = max(cosine(t, l) for l in like_embeddings)
    dislike_sim = max(cosine(t, d) for d in dislike_embeddings)
    contribution = +like_sim if like_sim > dislike_sim else -dislike_sim

tag_score     = sum(contributions) / len(tags)          # normalized
summary_score = same formula on the 1-sentence summary
base_score    = tag_score + (summary_weight × summary_score)
final_score   = base_score × match_multiplier + weather_bonus
```

Scores are unbounded floats (higher = better, negatives valid). Never normalize relative to
current batch — events must stay comparable across runs.

Tier thresholds, summary_weight, and match_multipliers live in `config.yaml`, not code.

---

## Working agreements

1. **TDD always.** Write failing tests first. Never implement without a red test.
2. **No network calls in tests.** All external services injected as dependencies so tests
   substitute fakes. Violation = bug.
3. **Injectable time.** Never call `datetime.now()` directly. Pass a `get_now` callable as
   a parameter to anything time-sensitive. Critical for testing time filters and lookback windows.
4. **Phase gates.** No phase begins until all previous phase tests are green AND the smoke
   test passes. See `docs/implementation-plan.md` for smoke test per phase.
5. **No hardcoded geography, credentials, or magic numbers.** Everything configurable.

## Test structure

Tests live in `tests/` and mirror the `src/` package structure.

```
tests/
  unit/             ← pure logic, no I/O, no network
    test_config.py
    utils/
      test_vectors.py
    ingestion/
      test_adapters.py
    ...
  integration/      ← cross-module, real SQLite, real Ollama; no external network
    test_smoke.py
  e2e/              ← full CLI invocations against a populated DB
    test_cli.py
```

Rules:
- A module at `src/foo/bar.py` gets its unit tests at `tests/unit/foo/test_bar.py`
- Smoke tests per phase live in `tests/integration/test_smoke.py` and accumulate
- Never write tests that assert directories or files exist — a missing `__init__.py`
  or template file will surface immediately as an import error or runtime failure
- No phase-labelled test names (e.g. `test_phase0_*`, `describe('P0: ...')`) —
  plan structure belongs in the plan file, not in source

---

## Docstrings

Use Google style. Keep them terse — one summary line is enough for simple functions.
Only expand with `Args:` / `Returns:` / `Raises:` sections when the signature alone
isn't self-explanatory.

```python
def load_config(config_path: Path | None = None) -> AppConfig:
    """Load and validate application config from YAML and environment.

    Args:
        config_path: Path to config.yaml. Defaults to config/config.yaml.

    Returns:
        Validated AppConfig instance.

    Raises:
        ConfigError: If required config fields are missing.
    """
```

Every public function, method, and class gets a docstring. Private helpers (`_foo`)
only if the logic is non-obvious.

---

## Commit conventions

Use conventional commits with atomic scope. Subject line ≤ 50 characters.

```
feat: add venue discovery service
fix: correct timezone derivation from lat/lng
chore: add pyproject.toml entry points
test: add normalization edge case coverage
refactor: extract vector encode/decode utilities
docs: update phase status in CLAUDE.md
```

Types: `feat`, `fix`, `chore`, `test`, `refactor`, `docs`, `ci`

One logical change per commit. Tests and implementation for the same unit go in one commit.
Breaking changes get a `!` after the type: `feat!: change EventCandidate schema`.

---

## Key footguns — read before touching anything

| Footgun | Fix |
|---|---|
| `config.yaml` contains personal location data | In `.gitignore`; ship `config/config.example.yaml` |
| `database/`, `logs/`, `data/likes.txt`, `data/dislikes.txt` | All in `.gitignore`; ship `.example` or empty versions |
| Package name can't be `what-do` | Python package is `what_do`; CLI entry point is `what-do` |
| `src/` layout breaks pytest imports | `pyproject.toml` needs `[tool.pytest.ini_options] pythonpath = ["."]` |
| `datetime.now()` called directly | Will cause flaky time-dependent tests; use injected `get_now()` |
| Embedding precision | Store as float32 BLOB, not float64. Use `encode_vector`/`decode_vector` utilities everywhere |
| LLM Pass 1 bypass | If `event.tags` already populated, skip extraction. No special flag. Handles synthetic events |
| Synthetic activities | Enter pipeline as pre-structured `Events` (not `EventCandidates`), after dedup, before Stage 1 |
| Blocklist source of truth | `data/blocklist.json` is authoritative. DB table overwritten from file at each batch start |
| LLM Pass 2 | Deferred to post-v1. Slot reserved between steps 11 and 13. Do not implement in v1 |
| Async in v1 | v1 is deliberately single-threaded. No asyncio. Parallelism is post-v1 only |

---

## Async/sync boundary (future reference)

v1 is single-threaded. When parallelism is added later, natural boundaries are:

- **Parallelizable:** scraping multiple sources, embedding multiple events, enrichment per event
- **Must remain sequential:** dedup pass 1 (needs full set), dedup pass 2 (needs all embeddings),
  final scoring (needs all similarity scores for tier assignment)

Wire these as sequential today using plain lists. The pipeline stage interface
(`process(events) → events`) is already compatible with future executor-based parallelism.

---

## Data files

```
config/config.example.yaml   — copy to config/config.yaml and fill in (gitignored)
data/likes.txt               — user preferences, [section] headers supported (gitignored)
data/dislikes.txt            — user dislikes, same format (gitignored)
data/blocklist.json          — flat array of venue names or @handles (gitignored)
data/seeds.yaml              — starting handles/venues for discovery
.env.example                 — copy to .env and fill in secrets (gitignored)
```

Required secrets (in `.env`):
```
APIFY_API_KEY=
TMDB_API_KEY=
AMC_API_KEY=
OLLAMA_HOST=http://localhost:11434
```

---

## Implementation phases

| Phase | Name | Status |
|---|---|---|
| 0 | Project skeleton | ✅ complete |
| 1 | Config & database foundation | ✅ complete |
| 2 | Venue discovery | ✅ complete |
| 3 | Event ingestion | ⬜ not started |
| 4 | Normalization & deduplication | ⬜ not started |
| 5 | Environmental enrichment | ⬜ not started |
| 6 | LLM extraction pipeline | ⬜ not started |
| 7 | Semantic matching engine | ⬜ not started |
| 8 | Deterministic ranking engine | ⬜ not started |
| 9 | CLI interface | ⬜ not started |
| 10 | Maintenance utilities | ⬜ not started |
| 11 | Hardening & reliability | ⬜ not started |

Update status here as phases complete.
