"""Model-compliance tests — a waiting room for the bench in issue #2.

Every test here asks the same kind of question: *does this model comply with
our prompts?* That is a measurement of a model, not an assertion about our
code, so it is run when **selecting** a model and never as a commit gate.
Transport, contract and prompt construction are our code, and stay in the fast
suite where they are mocked.

**This directory is temporary.** Issue #2 builds a multi-model bench, where a
compliance result becomes a measurement against a named model rather than a
pass/fail assertion — which is what these tests actually want to be. When it
lands, these move into it and the `model` marker and this package are deleted.
Do not add to it in the meantime; it exists only because these tests needed
somewhere to live that was not the commit gate.

Run with `pytest -m model`. Tests carrying `external` as well need a
third-party key; `-m "model and not external"` is the local-only subset.
"""
