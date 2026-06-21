# what-do

A local-first recommendation engine that answers "what should we do tonight?"

> **Status: under active development — not yet functional.**

Discovers local events from social media, movie theaters, and venue calendars. Runs an
overnight batch job to scrape, normalize, enrich, and score everything against your
personal preferences. When you ask, it responds instantly from precomputed data — no
network calls, no waiting.

## How it works

1. A nightly batch job scrapes configured sources (Instagram accounts, theater schedules, etc.)
2. Events are normalized, deduplicated, and enriched with weather and astronomical context
3. A local LLM extracts structured tags and summaries from raw event text
4. Events are scored against your `likes.txt` / `dislikes.txt` using semantic similarity
5. `what-do` reads the precomputed scores and renders your recommendations instantly

## Setup

```bash
# 1. Copy config templates
cp config/config.example.yaml config/config.yaml   # fill in your coordinates
cp .env.example .env                               # fill in API keys
cp data/likes.example.txt data/likes.txt           # add your preferences
cp data/dislikes.example.txt data/dislikes.txt

# 2. Install
pip install -e .

# 3. Run a batch
what-do-run-batch

# 4. See recommendations
what-do
```

## Requirements

- Python 3.11+
- [Ollama](https://ollama.com) running locally with `gemma4:e4b` and `nomic-embed-text`

## Design

See [`docs/`](docs/) for the full requirements, architecture, and implementation plan.
