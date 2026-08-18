"""A real call to the real Gemini API returns something our provider can read.

Deliberately the cheapest claim available — one short disambiguation, about two
seconds. It is not here to judge Gemini's answers; that is the bench's question
(#2). It is here so that a change to `GeminiClient` can be checked against the
live API without inventing a new harness, and it stays small so that checking is
never expensive enough to skip.

**It is also the only place the politeness path is exercised end to end.**
Nothing in `src/` constructs a `GeminiClient` — Ollama is what both composition
roots build — so the unit tests fake the SDK and this is the one run where a real
throttle, a real timeout and a real SDK meet each other. It is never a gate.
"""

from __future__ import annotations

import os
import random
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
import pytest

from src.config import load_config
from src.ingestion.disambiguation import OllamaDisambiguationProvider
from src.network.policy import RequestPolicy
from src.network.throttle import InMemoryThrottle
from src.utils.gemini_client import GEMINI_HOST, GeminiClient


def _require_gemini() -> tuple[str, str]:
    """Load the key/model or skip the test if no key is configured."""
    load_dotenv()
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        pytest.skip("GEMINI_API_KEY not set")
    return key, os.environ.get("GEMINI_MODEL", "gemini-flash-latest")


def _live_policy() -> RequestPolicy:
    """The real policy over the real config — including its real sleeps.

    The configured `network.policies.gemini` is what a batch would use, so a
    missing host assignment fails here rather than at 02:00.
    """
    return RequestPolicy(
        network=load_config().network,
        throttle=InMemoryThrottle(
            get_now=lambda: datetime.now(timezone.utc), sleep=time.sleep
        ),
        sleep=time.sleep,
        random=random.random,
    )


@pytest.mark.external
def test_real_gemini_disambiguation():
    """Real Gemini classifies an obvious venue handle as 'venue'."""

    key, model = _require_gemini()
    client = GeminiClient(
        api_key=key,
        policy=_live_policy(),
        get_now=lambda: datetime.now(timezone.utc),
    )
    provider = OllamaDisambiguationProvider(client=client, model=model)

    result = provider.classify(
        handle="@thevaultlounge",
        context="Come enjoy live jazz at @thevaultlounge this Saturday — doors open at 7pm!",
    )
    assert result == "venue"


@pytest.mark.external
def test_the_configured_policy_covers_the_host_gemini_actually_calls():
    """The assignment and the caller cannot drift: one names the host, the other
    is the constant the call site uses."""
    assert load_config().network.for_host(GEMINI_HOST).max_attempts >= 1
