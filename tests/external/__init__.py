"""Tests that call a third-party API. Run by hand; never a gate.

These cost money, need a key, and can fail for reasons that have nothing to do
with our code — Gemini returned `503 UNAVAILABLE — this model is currently
experiencing high demand` twice in one session while this tier was being
designed, once blocking a commit. A gate a third party can block is a gate we
learn to bypass.

Keep this tier small. Anything asserting how *well* a model performs belongs in
`tests/model/` (and ultimately the bench in #2); anything about our own code
belongs in the fast suite, mocked. What is left here is the narrow claim that
the live path works at all.

Run with `pytest -m external`.
"""
