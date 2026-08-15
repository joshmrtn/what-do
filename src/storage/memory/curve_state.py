"""In-memory `CurveStateRepository` — the single official fake."""

from __future__ import annotations

from src.storage.curve_state import CurveState


class InMemoryCurveStateRepository:
    """Holds one curve state."""

    def __init__(self) -> None:
        self._state: CurveState | None = None

    def load(self) -> CurveState | None:
        """The curve in force, or None when the config defaults still stand."""
        return self._state

    def save(self, state: CurveState) -> None:
        """Replace the curve in force."""
        self._state = state
