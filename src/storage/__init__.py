"""Storage layer.

Deliberately empty: a package `__init__` that imports a submodule makes every
import of this package pay for it, and the re-export of `init_db` that used to
live here had no callers at all — all 58 importers name
`storage.sqlite.connection` directly.
"""
