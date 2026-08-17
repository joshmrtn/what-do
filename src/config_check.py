"""Report configuration that leaves a feature switched off.

Measured 2026-08-16: the live `config.yaml` held a `weather:` section containing
only `provider`, so `weather.comfort` loaded as `{}`, `compute_comfort` iterated
nothing, and **every weather adjustment was 0.0 on every ranking ever stored** —
phase 8 had never contributed to a score. `scoring.domain_map` was empty the same
way, leaving every `[movies]` preference inert against 621 cinema events.

Neither warned. `raw.get("x") or {}` behind a `field(default_factory=dict)`
cannot tell an absent key from a configured-empty one, `git status` stays clean
because the live file is gitignored, and the suite passes throughout because
every test builds its own config — so the un-configured path is reachable **only
in production**.

**This checks the loaded config, not `config.example.yaml`.** The example is a
third artefact that drifts, and a checker built on it would rebuild the trap it
exists to close.

**It reports features that are off, not values that are defaulted.** Dataclass
field names are not YAML keys — nine fields are flattenings or renames
(`weather.air_quality_enabled` is `air_quality.enabled`, the three
`scoring.match_multiplier_*` are `match_multipliers.*`) — so a per-key inventory
would need a field-to-path mapping, which is a fourth drifting artefact. An empty
collection needs no mapping: it is legible from the loaded object alone, and it
is the shape that means *this feature does nothing*.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import yaml
from typing import Any

from src.config import AppConfig

#: A feature is off and nothing else says so.
WARNING = "WARNING"

#: A section the file never mentioned. Quieter, because every key in it has a
#: default and most of those defaults are correct.
INFO = "INFO"


@dataclass(frozen=True)
class Finding:
    """One switched-off feature.

    `path` is dotted and `detail` names the value in force, because "say what it
    fell back to" is the actionable half — a path alone sends the reader back to
    the file to work out what empty meant.
    """

    level: str
    path: str
    detail: str


def check_config(config: Any) -> list[Finding]:
    """Every collection in the config that is empty, deepest path first.

    Args:
        config: The loaded `AppConfig`.

    Returns:
        One finding per empty `dict` or `list`, in declaration order. Empty when
        nothing is switched off — which is the result a correct config must
        produce, or the one real finding arrives in a crowd.
    """
    return list(_walk(config, prefix=""))


def _walk(node: Any, *, prefix: str) -> list[Finding]:
    findings: list[Finding] = []
    for field in dataclasses.fields(node):
        value = getattr(node, field.name)
        path = f"{prefix}{field.name}"

        # Recurse before testing: a nested config is itself a dataclass, never a
        # collection, so the two branches cannot both apply.
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            findings.extend(_walk(value, prefix=f"{path}."))
        elif isinstance(value, (dict, list)) and not value:
            findings.append(
                Finding(
                    level=WARNING,
                    path=path,
                    detail=f"empty ({value!r}) — this feature does nothing",
                )
            )
    return findings


def check_sections(raw: Any) -> list[Finding]:
    """Top-level sections the file never mentioned.

    Separate from `check_config` because it answers a different question from a
    different input: `check_config` reads the *loaded* object and asks "what is
    switched off", while this reads the *raw mapping* and asks "what did the
    file never say". Only the raw form can tell absent from defaulted.

    Section names are the one place dataclass fields and YAML keys do match
    1:1, which is why this is section-level and not key-level — nine fields are
    flattenings or renames, so a per-key version would need a mapping table that
    would drift exactly as `config.example.yaml` does.

    Args:
        raw: The parsed YAML mapping, before any defaults are applied.

    Returns:
        One `INFO` per absent top-level section, in declaration order.
    """
    named = set(raw or {})
    return [
        Finding(
            level=INFO,
            path=field.name,
            detail="not in the file — running entirely on defaults",
        )
        for field in dataclasses.fields(AppConfig)
        if field.name not in named and _is_section(field)
    ]


def _is_section(field: dataclasses.Field[Any]) -> bool:
    """Whether a top-level field is a *section* rather than a bare value.

    A section is a nested config block, or a list of them. A bare scalar is not
    one — and three of them (`ollama_host`, `gemini_api_key`, `gemini_model`)
    are read from `.env` and never appear in YAML at all, so reporting absent
    scalars would produce three permanent false positives against every config
    that has ever existed, the example included. A check that is wrong three
    times on a correct file teaches people to skip it.

    Decided by the field's *default*, which the dataclass already carries, so
    there is no list of names to keep in step with `config.py`.
    """
    factory = field.default_factory
    if factory is dataclasses.MISSING:
        # No factory means a plain default or a required field. Required
        # sections (`location`, `scraping`) have no default at all and must
        # still be reported when the file omits them.
        return field.default is dataclasses.MISSING
    produced = factory()
    return dataclasses.is_dataclass(produced) or isinstance(produced, list)


def check_config_file(config: Any, path: Any) -> list[Finding]:
    """Both checks, as the batch wants them: switched-off features and absent sections.

    One function rather than two seams. The batch injects this whole thing in
    tests, and a second, un-injected half would read the machine's real
    `config.yaml` in every test that did not pass `--config` — which is the
    environmental dependency `de50499` removed from the config tests once
    already.

    An unreadable file yields no section findings: the loader has already failed
    loudly by the time anything reaches here, and a second complaint about the
    same file is noise.

    Args:
        config: The loaded `AppConfig`.
        path: Where the raw YAML lives.

    Returns:
        Switched-off features first, then absent sections.
    """
    try:
        with open(path) as handle:
            raw = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError):
        return check_config(config)
    return check_config(config) + check_sections(raw)
