"""Shared test infrastructure.

Deliberately small. The rule this package lives under: it may **build real
collaborators and fake a boundary**, and it may not restate behaviour we own.
A helper that reimplemented a stage would drift from the real one silently,
because nothing checks it — which is why the stage fakes rotted and the
repository contract suites did not.
"""
