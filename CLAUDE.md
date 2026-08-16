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
- Veezi public ticketing pages — independent cinema showtimes, no credentials
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
This holds at the classification layer too: an event is `no` only when the best dislike beats
the best like *by a margin*, never on an absolute dislike threshold — measured, an absolute
cutoff force-rejects a karaoke bar the user likes (`bar ↔ bars` = 0.932).

**Tags are weighted by centrality.** LLM Pass 1 assigns each tag a 0.0–1.0 weight for how central
it is to what the event *is*, so incidental venue attributes ("bar", "thursday") recede against
the main activity. Local `gemma4:e4b` judges this better than Gemini flash.

**Domain-scoped preferences.** `likes.txt` and `dislikes.txt` support section headers:
`[general]`, `[movies]`, `[restaurants]`. Domain preferences only apply to events with a
matching `source_type`. Lines before the first header are `[general]`.

**Set operations vs per-event.** Dedup passes need all events in memory. Everything else
(normalization, enrichment, LLM, embedding, similarity, scoring) is per-event and can stream.

**Two timestamp conventions, one boundary.** The **raw layer stores instants**:
`EventCandidate` canonicalises every timestamp to UTC in `__post_init__`, because stored
candidates are compared as *text* by `for_window` and text order only matches chronological
order at a fixed offset. The **domain layer stores local time**: `_normalize_timestamp` puts
`Event` in the configured zone, because that is where `.date()` and `strftime` are called and
they read the zone rather than the text. `_normalize_timestamp` is the boundary. Neither
convention is optional — an `Event` in UTC misfiles every evening event by a day, and a
candidate in local time reintroduces the mixed-offset comparison.

**A naive timestamp is resolved at ingestion, and the assumption is recorded.** `_resolve_naivety`
reads a bare timestamp as local and writes `metadata["assumed_zone"]`, because ingestion is the
single funnel every fetched candidate passes through — no adapter can bypass it and none has to
remember. `EventCandidate` itself stays tolerant: it has no zone to convert with, and ingestion
must never crash on a naive value, which is what killed the first live fetch.

**Compare on a canonical key; store what the source wrote.** Three instances, all landed
2026-08-14 and all the same shape — put both sides into one representation, then compare
*exactly*. This is never a loosening, and a fuzzy comparison is not the same thing:

| what | key | why |
|---|---|---|
| timestamps | UTC | text order only matches chronological order at a fixed offset |
| venues | casefolded, leading article stripped | `The Rhumb Line` and `Rhumb Line` are one venue |
| titles | casefolded, `w/`→`with`, `&`→`and` | `token_sort_ratio` is case-sensitive |

The key is deliberately **not** what gets stored or displayed — stripping the article from a
stored venue would put "House Of The Seven Gables" on screen. Expansions stay short and each is
a judgement: `ft.` is excluded because it genuinely means *feet*, `@` because it is a handle as
often as a preposition, and both have tests pinning the exclusion.

---

## Scoring formula

```
gate(s) = 1 / (1 + exp(-(s - gate_midpoint) / gate_temperature))   # defaults 0.60, 0.04

for each weighted tag (t, w):
    like_sim    = max(cosine(t, l) for l in like_embeddings)
    dislike_sim = max(cosine(t, d) for d in dislike_embeddings)
    contribution = w × (+like_sim × gate(like_sim)  if like_sim > dislike_sim
                        else -dislike_sim × gate(dislike_sim))

tag_score     = mean(positive contributions) - mean(|negative contributions|)
summary_score = same formula on the 1-sentence summary
base_score    = tag_score + (summary_weight × summary_score)

# Symmetric: confidence shrinks magnitude toward zero in BOTH directions,
# because thin evidence means uncertain, not bad. Synthetic activities are
# exempt — their tags are authored, not extracted.
tag_confidence = min(1.0, len(tags) / min_tags_per_event)
confident      = base_score × tag_confidence

# Direction-aware: the multiplier acts on magnitude, sign preserved.
final_score   = confident × match_multiplier + weather_adjustment   if confident >= 0
final_score   = confident ÷ match_multiplier + weather_adjustment   if confident < 0
```

Three parts are load-bearing and were each measured against real embeddings — dropping any one
breaks real cases. See `docs/decisions.md`:

- **The logistic gate** kills the ~0.42 cosine noise floor. Without it, noise decides a tag's sign
  *and* contributes near-full magnitude ("sushi" scored -0.433 against a dislike list containing
  no food terms).
- **Weights multiply the contribution.** Never use them as averaging weights — that normalises
  them away and the suppression silently does nothing.
- **The balanced mean** stops several weak incidental negatives outvoting one strong positive.

The two scaling terms are deliberately different shapes and each looks like a bug from the
other's perspective. The **multiplier** expresses how strong a verdict is, so it preserves sign.
**Confidence** expresses how much evidence exists at all, so it pulls both signs toward zero — a
one-tag event lands mid-ranking, which is where something we know almost nothing about belongs.

Scores are unbounded floats (higher = better, negatives valid). Never normalize relative to
current batch — events must stay comparable across runs.

summary_weight and match_multipliers live in `config.yaml`, not code.

---

## Working agreements

1. **TDD always.** Write failing tests first. Never implement without a red test.
2. **No network calls in tests.** All external services injected as dependencies so tests
   substitute fakes. Violation = bug. The `external` tier is the only exception:
   it is excluded from every default run and may never gate a commit.

   **Fake only at an external boundary — never for code we own.** Dependency injection is
   throughout this codebase so that the *edges* can be substituted: a network call, a model,
   a third-party API, the clock. A pipeline stage, a service, a normalizer is **ours**, and
   a double that restates our logic diverges from it by construction, because nothing checks
   it. The rule that follows:

   > **A double may record, but it may not reimplement.**

   A spy returns its input unchanged and records that it was called — it makes no behavioural
   claim, so it cannot be wrong. A *mirror* restates real behaviour and is always one commit
   from lying. When a test needs a stage's behaviour, **build the real stage and fake the seam
   it already exposes** — `ExtractionStage(provider, …)` and `EmbeddingStage(provider, logger)`
   both take the model boundary as an injected dependency, so the stage fake ceases to exist
   rather than needing to be kept honest.

   Where constructing the real collaborator is genuinely too expensive, it gets a **contract
   suite** instead — one parametrised set of assertions run against every implementation, as
   `InMemory*Repository` has. That is why the repositories have never drifted and the stage
   fakes have.
3. **Injectable time.** Never call `datetime.now()` directly. Pass a `get_now` callable as
   a parameter to anything time-sensitive. Critical for testing time filters and lookback windows.
4. **Phase gates.** No phase begins until all previous phase tests are green AND the smoke
   test passes. See `docs/implementation-plan.md` for smoke test per phase.
5. **No hardcoded geography, credentials, or magic numbers.** Everything configurable.
6. **A config key is two files.** `config/config.example.yaml` is tracked; `config/config.yaml`
   is gitignored and is what actually runs. Adding or changing a key in the example updates
   documentation and nothing else — the live file is untouched, `git status` stays clean, and
   the suite passes either way, because every test builds its own config.

   > **Whenever a config key is added or changed, stop and ask the user about updating the
   > live `config/config.yaml`.** Never edit it unprompted, and never assume the example
   > reached it.

   Measured 2026-08-16: the live `weather:` section held **only `provider`**, so the comfort
   curves never loaded and `weather_adjustment` was **0.0 on every ranking ever stored** —
   phase 8 had contributed nothing to any score since it shipped. `scoring.domain_map` was
   empty the same way, leaving every `[movies]` preference inert against 621 cinema events.
   Both failed *silently*, because an absent key and a configured-empty one are
   indistinguishable to `raw.get("x") or {}` behind a `default_factory=dict`.

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
  model/            ← does a model comply with our prompts; deleted by #2
    test_extraction_compliance.py
  external/         ← third-party API, needs a key; never a gate
    test_gemini_live_path.py
  tier_plugin.py    ← the integration probe and its loud skip
```

Rules:
- A module at `src/foo/bar.py` gets its unit tests at `tests/unit/foo/test_bar.py`
- Smoke tests per phase live in `tests/integration/test_smoke.py` and accumulate
- Never write tests that assert directories or files exist — a missing `__init__.py`
  or template file will surface immediately as an import error or runtime failure
- No phase-labelled test names (e.g. `test_phase0_*`, `describe('P0: ...')`) —
  plan structure belongs in the plan file, not in source

### Markers name the reason to run, not the cost

Cost is why a test gets skipped; *reason to run* is what decides when it runs.
Registered in `pyproject.toml`, enforced with `--strict-markers`, so an
unregistered marker fails collection instead of silently joining the default run.

| marker | what it is | when it runs |
|---|---|---|
| *(unmarked)* | mocked; tests our code | every run |
| `integration` | real SQLite, real Ollama **embeddings**, seconds | every run; skips **loudly** without Ollama |
| `external` | third-party API and a key | by hand. Never a gate |

`model` and `external` are orthogonal and **stack** — a Gemini compliance test
Default selection is `-m "not external"`.

**There is no `model` tier.** Whether a model is any *good* is the bench's
question — `what-do-bench`, issue #2 — and a marker for it made `tests/model/`
a second home for it. `--strict-markers` means a leftover `@pytest.mark.model`
fails collection rather than quietly rejoining the default run.

**The dividing line for what gets mocked:** anything asserting a qualitative or
quantitative property of an LLM belongs in the bench, not the suite. Everything else — transport, contract,
prompt construction, JSON parsing, fence stripping, retry — is our code, is
mocked, and runs every time.

**The commit gate is the default suite**, which now includes `integration`. It
is seconds, which is the point: a gate expensive enough to skip is not a gate.

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
| Embedding precision | Vectors are float32, never float64 — that is the invariant. *How* they are encoded is a storage concern: `encode_vector`/`decode_vector` belong behind the storage layer, not scattered through the core |
| Opening a database with `sqlite3.connect` directly | `foreign_keys` and `busy_timeout` are per-connection and default to off and zero, so a raw connection silently drops referential integrity and fails instantly on a lock. Always use `storage.db.connect()` |
| Copying `event_hub.db` with `cp` | WAL keeps recent commits in a sidecar, so a plain copy captures a torn state. Use `scripts/backup-db.sh`, which does `VACUUM INTO` |
| A tag vector stored per event | A vector is a pure function of `(tag text, model)`. Storing it per event-tag pairing costs 4× the space and re-embeds tags seen on previous nights. The *weight* is per event; the vector is not |
| LLM Pass 1 bypass | If `event.tags` already populated, skip extraction. No special flag. Handles synthetic events |
| Synthetic activities | Enter pipeline as pre-structured `Events` (not `EventCandidates`), after dedup, before Stage 1 |
| Blocklist source of truth | `data/blocklist.json` is the *only* source. The composition root reads it once and hands it to ingestion and ranking; there is no DB table, by decision (#16) |
| LLM Pass 2 | Deferred to post-v1. Slot reserved between steps 11 and 13. Do not implement in v1 |
| Async in v1 | v1 is deliberately single-threaded. No asyncio. Parallelism is post-v1 only |
| `base × match_multiplier` on a negative score | Inverts intent — `no` at 0.5 turns -0.40 into -0.20, *rewarding* the clearest rejections. Divide instead when the base is negative |
| Tag weights used as averaging weights | Normalises the weight away — suppression silently does nothing. Weight must multiply the contribution: `c = w × similarity` |
| Assuming unrelated concepts score ~0 | `nomic-embed-text` puts unrelated pairs at ~0.30–0.47. Every absolute threshold must account for the floor; the logistic gate exists for this |
| Raw cosine magnitude on a near-tie | A 0.03 noise gap becomes a ~0.43 penalty. Always gate before comparing |
| Making tag confidence direction-aware "to match the multiplier" | Inverts its meaning — a thin *negative* would deepen, punishing an event for evidence we never gathered. Confidence multiplies both signs |
| Applying tag confidence to synthetic activities | Their tags come from hand-written `config.yaml` rules, so a low count is authoring, not extraction failure. `source_type == "synthetic"` is exempt |
| Serving a cached forecast without checking its age | An event found a week out would score on the forecast issued that day, forever. `weather.cache_ttl_hours` must stay under 24 so a nightly batch refetches |
| Reintroducing tiers, bands or score labels | Removed entirely on 2026-08-11 and **not** to be re-proposed. They were never asked for, were speculative from the start, and no calibration ever beats ranking by score. The order is the product; a band is a second, worse judgement laid on top |
| Blocklist `@handle` entries at ranking time | An `Event` carries no handle — normalization drops it. Handles are enforced at ingestion; ranking matches venue names only. See #15 |
| Checking a database exists with `Path.exists()` | `sqlite3.connect` creates a zero-byte file for any path, so a stray read leaves one that then fails with `no such table`. Use `has_schema()` |
| Cutting the CLI list without saying what was cut | The count must stay on screen (`+ N more events ranked lower (--all)`). A counted event is one flag away; a silently dropped one is invisible |
| Giving an undated event its own section | It takes the event out of the ranking, which is the one thing the order is for. It ranks inline; the time column reads `time TBC` |
| A default argument that no test ever uses | `get_now=datetime.now` was naive while every test injected an aware clock, so production was the only naive path and the suite stayed green. It killed the first live fetch. Defaults that only production reaches are untested by construction |
| Comparing a datetime without localising it | Sources genuinely differ — Do617 and JSON-LD state an offset, HTML listings do not. Two filters over the same field must read naivety the same way, or one raises on input the other accepted |
| Keying a candidate id on an event URL alone | A recurring programme keeps one page across every date it runs — PEM's 97 listings are 61 URLs — so the id must carry the start too, or a whole season collapses into one candidate |
| Concluding a site has no events because it has no feed | Autodiscovery cannot see `application/ld+json`. PEM publishes 97 events that way and was written off twice. Grep for it before writing any bespoke parser |
| Matching the word `cancelled` anywhere in a title | Throws away `Cancelled Culture: A Comedy Show` and a `Never Cancelled Tour`. A cancellation is the feed's *marker* — asterisk-delimited, or leading with a separator |
| A tier that is deselected by default and never run | It reports green having checked nothing. The FK bug in `event_scores` shipped this way: 1,777 fast tests passed while the one test that would have caught it sat behind a multi-minute marker. Cheap cross-module tests belong in the default run; anything excluded must be excluded for a *reason to run*, not for its cost |
| A model field with no column | Adding a field that round-trips through storage is **four** changes — dataclass, table, writer, reader. Miss any of the last three and nothing throws: the reader returns the default and every test passes. `EventCandidate.timing` was dropped this way for weeks. Worse here because `_merge_candidates` prefers the **loaded** copy, so the stripped reload is what the pipeline uses |
| A round-trip test that leaves a field at its default | It passes against a column that does not exist. Assert every field at a **non-default** value — see `tests/unit/storage/test_candidate_repository.py` |
| A cache whose caller must remember to check freshness | Pass the freshness bound *in* (`WeatherCache.get(..., fresh_since=…)`) so a stale read has no API. A rule remembered at every call site is a rule that will be forgotten |
| Asserting a substring that spans a wrapped line | `assert "more (--all)" not in out` was **vacuously true** — the real line is `+ 4 more events ranked lower (--all)`. Collapse whitespace, or assert on text that is actually contiguous |
| A test that pins a value the code re-derives | It passes against a mutation that changes the arithmetic. The `extraction_input` guard asserted the *reason string's* character count — and the reason string calls `extraction_input` a second time, so the number matched while the calculation read something else. Assert the **behaviour** the value feeds, not a value recomputed alongside it |
| A package `__init__` that imports a submodule | Every import of the package pays for it, and no import statement in the suffering module shows why. A convenience re-export in `composition/__init__.py` made importing `composition.storage` execute `batch`, taking the CLI from 41 loaded modules to 105 — every adapter and the Ollama client — with the whole suite green. Guard on the **loaded module graph**, not on imports |
| Reading `tag_confidence` as extraction *quality* | It measures conformity to what the model typically does at that input length, not correctness. An event the extractor reliably mis-tags looks perfectly confident. The curve is a heuristic over a weak proxy (R² 0.57); only issue #2's bench can speak to whether a tag is right |
| A fake that reimplements a stage we own | It drifts the moment the real one changes, silently, because nothing checks it against the original. `_FakeExtraction` says *"Mirrors ExtractionStage"* and was wrong within one commit. Build the **real** stage with a faked provider — the model boundary is already injected — or give it a contract suite. Fakes belong at network, model and third-party edges only |
| Testing a marker with `item.keywords` | `keywords` holds the name of every parent node, so `"integration" in item.keywords` matches the `tests/integration/` **directory** and skips the whole file. Use `item.get_closest_marker(...)` |
| Trusting the suite to notice a schema change reached the live database | `_SCHEMA` is all `CREATE TABLE IF NOT EXISTS`, so adding a column serves every *fresh* database — every test — and leaves `event_hub.db` untouched until a hand migration runs. The suite is green either way and **no test can see the difference**. A DDL change is two artefacts, and the only check that closes the gap is comparing the live schema against a freshly initialised one, column for column |
| Assuming a column-order difference between live and fresh is harmless | It is harmless *here* only because every SELECT names its columns and there is no `SELECT *` in `src/`, so positional reads index the SELECT list rather than the table. `event_scores` has been column-order-drifted since the C2 hand migration for exactly that reason. Introduce one `SELECT *` and the drift becomes a live-only bug the suite cannot reach |
| `ALTER TABLE x RENAME TO x_old` as the first step of a table rebuild | SQLite **rewrites the `REFERENCES` clause of every table pointing at `x`**, so they now point at `x_old` — which the rebuild then drops. Measured: renaming `event_scores` left `score_reasons` with **7,297 dangling references**, and every column in every table still matched a fresh build exactly. Build the replacement under a *new* name, copy, drop the old, rename into place: nothing references the new name, so nothing is rewritten |
| Verifying a migration after `COMMIT` | The check finds the damage and it is already written. `foreign_key_check`, row counts and any FK target belong **inside the transaction**, with a rollback on failure. This is how the `event_scores` rebuild shipped a broken database on its first attempt |
| A schema check that only compares columns | It cannot see referential integrity. A database whose foreign keys all dangle matches `_SCHEMA` column for column. Compare the shape **and** assert `PRAGMA foreign_key_check` is empty |
| Adding an `ExtractionResult` field with a default | A default lets a construction site record nothing, and its rows are then indistinguishable from honest ones — the failure the provenance columns exist to prevent. Required with no default makes `mypy --strict` enumerate the sites instead |
| A mutable default on a `NamedTuple` | `merges: dict[str, str] = {}` is **one dict shared by every instance that omits it**, so a caller who mutates it leaks into the next. `field(default_factory=...)` does not exist here. Make it required and let `mypy --strict` name the sites |
| Spending a scarce resource before checking the work is wanted | Extraction is soonest-first, and "soonest" can be *behind* us. Two whole 480-minute budgets went on events that had already happened. The budget did not cause it — it revealed a cost that was free when nothing was scarce. Any queue with a bound needs the scope check *inside* it, at the seam that sees every item whichever door it came in by |
| A recording double that pins a signature | `_DedupSpy.deduplicate(self, events, config)` broke the moment the real method grew a `now`. A spy exists to record and forward; the instant it declares the shape of what it forwards, it has started reimplementing. Use `**kwargs`, or `__getattr__`, as `_StageSpy` does |
| Reasoning from the shapes the code already has | "Losers are destroyed, so provenance goes on the survivor, so Pass 1 can never be explained" — every step true, conclusion wrong, because the premise was the thing to change. Ask what this would look like with no existing code, then decide what to keep. It turned a lossy row-shaped design into `dedup_decisions`, which can record the label a row *cannot*: "compared, and judged different" |
| Assuming O(n²) means unaffordable | The dedup guards cut 1.39M possible pairs to **1,784 actually scored**. The expensive-looking thing was already cheap and nobody had measured it. Measure the population before designing around its size |
| Assuming a fuzzy string library folds case | **`fuzz.token_sort_ratio` is case-sensitive** — rapidfuzz applies no processor by default. Measured: `HEADLANDS` vs `Headlands` scored **0.111**, so a venue writing its listings in caps was invisible to dedup Pass 1 entirely, and a `w/` versus `with` pair lost twice as much to casing as to the abbreviation. Any rapidfuzz comparison needs a canonical key |
| A structural guard compared on raw strings | A failed guard is *no comparison*, not a low score, so a venue written two ways silently removes the pair from dedup altogether. Measured: 3 of 154 venues collided, costing 4 pairs and 2 real duplicates. Canonicalise both sides and keep the guard exact — loosening it to a *fuzzy* venue match would admit pairs to Pass 2, whose summary vectors cannot tell two events at one venue apart (#13) |
| Reading a stored span as one continuous occurrence | A workshop published as "10–14 August, 09:00–12:00" is stored as a single row spanning **96 unbroken hours**, so it overlaps every `--time` window that can be typed. Beyond `LONG_SPAN_HOURS` the endpoints are read as *daily* hours. Equal times-of-day mean whole days and genuine continuity — a 12:00→12:00 mooring rental is not a programme that runs at noon |
| Anchoring a time-of-day window to the event's own date | For anything that began earlier the window is built on that earlier date and the comparison stops meaning anything — a month-long exhibition cleared every window unconditionally. Anchor to the **night being asked about**, which the caller passes |
| A sentinel taken from inside the value domain | It is a collision waiting for someone to type it. A bare `--upcoming` needs a placeholder until config is read; `0` made `--upcoming 0` mean "use the default", and `-1` did the same once it replaced it — argparse only rejects negatives when an option string looks like one. The `const` is an **object** now: `type=` applies only to strings, so a non-string passes through untouched |
| `args.x or default` for a numeric flag | `0` is falsy, so asking for none silently gets the default — and a negative reaches a slice. `--limit -5` rendered 91 rows because `pairs[:-5]` drops the *last* five, so the listing looked plausible and was wrong, with the "+ N more" count no longer describing it. Use `is None`, and validate **ahead of every dispatch** rather than beside the code that consumes the value |
| Comparing ISO timestamp strings across mixed offsets | Meaningless, and it looks fine. Google Calendar's `basic.ics` emits `DTSTART:…Z` for some events and `DTSTART;TZID=…` for others **in one file**, so `event_candidates` held three offsets and `for_window`'s text comparison wrongly kept events that were already over. **Both sides must share the offset** — canonicalise the *bound* too, or a local-form floor disagrees with the truth (measured: 15 candidates) against a table that is entirely UTC |
| Assuming a row absent from a query's result is gone | A retrieval filter is not a delete. There is no `DELETE` against `event_candidates` anywhere in `src/`, so a narrower bound is reversible: widen it and the row is reloaded, re-derived, and rematched onto its event by candidate id. Reasoning about "eviction" from a layer that is never destroyed nearly justified the right decision for an invented reason |
| Making two filters over one field "agree" | Name the question each asks first. `_scope_filter` asks *is this worth ranking* (product); `for_window` asks *is this record still live* (currency). They share a **floor**, because a finished event is both — and must not share a **ceiling**, because a horizon says nothing about staleness. Sharing the whole predicate bakes the confusion in |
| A shared bound that no test distinguishes from the obvious alternative | `_scope_floor` versus `now` differ only for events between local midnight and the batch's 02:00 start — so every test passed with the floor reverted. The one decision deliberately made was the one nothing checked. Write the case that separates them, then mutate to prove it |
| A tunable whose default is empty | It is a feature that ships switched off, and nothing says so. `weather.comfort` defaults to `{}`, so a missing section meant `compute_comfort` iterated nothing, returned no factors, and every weather adjustment was 0.0 — for every run ever stored. The other weather numbers have real code defaults; the curves were the odd ones out. Prefer a working default, or refuse to start |
| Editing `config.example.yaml` and stopping there | The example is tracked and documents; `config.yaml` is gitignored and *runs*. A key added to one never reaches the other, `git status` stays clean, and the suite passes because every test builds its own config. Ask before touching the live file — but always ask |
| Attributing an LLM transcript prompt to one event | A recurring series produces **byte-identical `extraction_input` across dates** — the same title, venue and category on every occurrence. Two events matched one prompt and a whole diagnosis was built on the wrong one. Key on the stored hash, never on prompt text |

---

## Async/sync boundary (future reference)

v1 is single-threaded. When parallelism is added later, natural boundaries are:

- **Parallelizable:** scraping multiple sources, embedding multiple events, enrichment per event
- **Must remain sequential:** dedup pass 1 (needs full set), dedup pass 2 (needs all embeddings),
  final scoring (needs all similarity scores to assign rank)

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

Secrets (in `.env`):
```
APIFY_API_KEY=
TMDB_API_KEY=
AMC_API_KEY=
OLLAMA_HOST=http://localhost:11434
```

Only `OLLAMA_HOST` is genuinely required. Every API key is optional: the composition root skips
a source whose key is absent, warning with the exact variable name it looked for, and records
the skip in the run summary. A skip never sets `outcome = partial` — that is for stage failures.

---

## Implementation phases

| Phase | Name | Status |
|---|---|---|
| 0 | Project skeleton | ✅ complete |
| 1 | Config & database foundation | ✅ complete |
| 2 | Venue discovery | ✅ complete |
| 3 | Event ingestion | ✅ complete |
| 4 | Normalization & deduplication | ✅ complete |
| 5 | Environmental enrichment | ✅ complete |
| 6 | LLM extraction pipeline | ✅ complete |
| 7 | Semantic matching engine | ✅ complete |
| 8 | Weather comfort enrichment | ✅ complete |
| 9 | Deterministic ranking engine | ✅ complete |
| 10 | CLI interface | ✅ complete |
| — | Batch orchestrator (issue #12) | ✅ complete |
| 11 | Maintenance utilities | ⬜ not started |
| 12 | Hardening & reliability | ⬜ not started |

Update status here as phases complete.

The batch orchestrator is not one of the numbered phases — it is the sequencer and composition
root that makes phases 3–10 actually run. `src/scheduler.py` holds `run_batch` and the
`what-do-run-batch` entry point; `src/composition.py` is the only place real providers are built.
Note that issue **#12** is the orchestrator, unrelated to **phase 12** above.

## Proving the sources without paying for a full run

`what-do-run-batch --ingest-only` fetches, filters and stops — seconds, against the hours a full
run costs at ~3 min/event of local LLM time. `--dry-run` is *not* a cheap alternative: it persists
nothing but still runs every stage.

Read the per-source table it prints rather than the total, and note that it counts fetched *and*
kept. `0 kept of 0 fetched` means broken or empty; `0 kept of 213 fetched` means the source works
and its dates are landing outside the window. Add `--raw [PATH]` to dump every candidate **as
fetched, before filtering**, as JSON Lines with the discard reason.

Baseline as of 2026-08-14 evening: **1275 ingested from 18 sources** at `horizon_days: 90` (1234
earlier the same day; was 1091 at
45 days on 2026-08-13 — raising the horizon a whole quarter added only ~143, because most sources
simply do not publish that far out). Expected zeroes are `do617_koto`, `do617_bit_bar` and `moon`;
anything else at zero is a regression.
