"""Composition roots: the only place real providers and stores are constructed.

- `storage.py` builds persistence, and is the one module naming a concrete
  implementation. Both roots go through it.
- `batch.py` builds a whole pipeline, for `what-do-run-batch`.
- The view root lives in `presentation/cli.py` and needs only `storage`.

**Deliberately empty.** Re-exporting `build_dependencies` here for convenience
made importing `composition.storage` execute `batch`, which pulls in every
adapter and the Ollama client — measured at 41 → 105 modules on the CLI's import
path, the one path whose promise is no LLM at query time. Import from the
specific module instead.
"""
