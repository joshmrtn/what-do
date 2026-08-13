"""Model evaluation bench — issue #2.

Deliberately empty. A package `__init__` that imports its submodules makes every
import of the package pay for all of them, with no import statement in the
suffering module to show why: a convenience re-export in `composition/__init__`
once took the CLI from 41 loaded modules to 105.
"""
