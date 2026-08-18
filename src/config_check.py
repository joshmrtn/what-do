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
from urllib.parse import urlsplit

import yaml
from typing import Any

from src.config import AppConfig
from src.enrichment.air_quality import AIR_QUALITY_HOST
from src.enrichment.movies import TMDB_HOST
from src.enrichment.weather import OPEN_METEO_HOST
from src.ingestion.geocoder import NOMINATIM_HOST
from src.ingestion.movies.amc import AMC_HOST
from src.ingestion.social.apify import APIFY_HOST
from src.ingestion.social.dumpor import DUMPOR_HOST
from src.ingestion.social.picuki import PICUKI_HOST
from src.processing.image_fetcher import DATA_DERIVED_POLICY
from src.utils.gemini_client import GEMINI_HOST

#: The config cannot run as it stands. Today that means exactly one thing: a
#: host this config will call with no politeness policy assigned to it. `for_host`
#: already refuses one — but only the first that happens to be reached, at the
#: moment it is reached, which for a seasonal source is months away.
ERROR = "ERROR"

#: A feature is off and nothing else says so.
WARNING = "WARNING"

#: A section the file never mentioned. Quieter, because every key in it has a
#: default and most of those defaults are correct.
INFO = "INFO"

#: Hosts a provider module hardcodes, which `sources:` therefore never mentions.
#: **Imported, not spelled again** — each provider already declares its host as
#: the constant its own call site uses, so there is one artefact per host rather
#: than two that can disagree. What is left to drift is membership of this tuple,
#: and `TestTheRegistryCannotDrift` compares it against what `src/` names.
PROVIDER_HOSTS = (
    AIR_QUALITY_HOST,
    AMC_HOST,
    APIFY_HOST,
    DUMPOR_HOST,
    GEMINI_HOST,
    NOMINATIM_HOST,
    OPEN_METEO_HOST,
    PICUKI_HOST,
    TMDB_HOST,
)

#: Hosts written into `src/` that are never fetched from, with the reason each
#: is not a gap. Short by design: an entry here exempts a host from the coverage
#: check, so it is the one place this check can go quietly wrong.
NEVER_FETCHED = {
    # A vocabulary URI. It identifies a schema.org type in markup we parse and is
    # never dereferenced.
    "schema.org": "a vocabulary URI in parsed markup, never fetched",
    # Ollama, and the bench's default pointing at the same place. Localhost is
    # the exemption — not "the model client", which would survive a provider swap
    # and silently exempt a hosted API.
    "localhost": "localhost: Ollama runs on this machine",
}

#: Policies named at a call site rather than assigned to a host, because their
#: hosts arrive from fetched data. Imported from the call site that names each.
CALL_SITE_POLICIES = (DATA_DERIVED_POLICY,)

#: Collections whose **empty state is the working one**, with the reason each is
#: not a switched-off feature. Short by design, on the same terms as
#: `NEVER_FETCHED`: an entry here is a finding this check agrees not to make.
#:
#: The rule everywhere else is that an empty collection means the feature does
#: nothing — `weather.comfort` empty meant no comfort curves loaded and every
#: adjustment was 0.0. An entry belongs here only when the *opposite* is true:
#: the feature is fully operative with nothing configured, and populating it is
#: the exception rather than the point.
WORKING_WHEN_EMPTY = {
    # Every source defaults to `auto` — measure the publisher's ids, latch to
    # content if they churn. That is the feature, and it needs no assignments;
    # an entry only pins a source out of the measurement. A config naming none
    # is the ordinary case, not one somebody forgot to fill in.
    "sources.identity": "unassigned means auto, which is the working default",
}


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
        elif isinstance(value, (dict, list)) and not value and path not in WORKING_WHEN_EMPTY:
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


def check_hosts(config: Any) -> list[Finding]:
    """Every host this config will call, and whether it has a policy.

    Two sources, because a host arrives two ways: `sources:` names the feeds,
    and a provider module hardcodes its own service. Neither half sees the
    other, and a check built on `sources:` alone would have declared a config
    complete while TMDb, Nominatim and Gemini went unassigned.

    Args:
        config: The loaded `AppConfig`.

    Returns:
        One `ERROR` per host with no policy and per name that resolves to
        nothing, then one `WARNING` per declared policy nothing uses. Empty when
        every host is covered.
    """
    declared = set(config.network.policies)
    assigned = config.network.hosts

    findings = [
        Finding(
            level=ERROR,
            path=f"network.hosts.{host}",
            detail=f"no politeness policy assigned — asked for by {source}",
        )
        for host, source in sorted(_hosts_called(config).items())
        if host not in assigned
    ]

    findings.extend(
        Finding(
            level=ERROR,
            path=f"network.hosts.{host}",
            detail=(
                f"names policy {name!r}, which is not declared under "
                f"network.policies ({', '.join(sorted(declared)) or 'none'})"
            ),
        )
        for host, name in sorted(assigned.items())
        if name not in declared
    )

    findings.extend(
        Finding(
            level=ERROR,
            path=f"network.policies.{name}",
            detail="named at a call site for hosts that arrive from fetched data, "
            "and not declared — the fetch that needs it will be refused",
        )
        for name in CALL_SITE_POLICIES
        if name not in declared
    )

    used = set(assigned.values()) | set(CALL_SITE_POLICIES)
    findings.extend(
        Finding(
            level=WARNING,
            path=f"network.policies.{name}",
            detail="declared but no host uses it — dead config",
        )
        for name in sorted(declared - used)
    )

    return findings


def _hosts_called(config: Any) -> dict[str, str]:
    """Every host this config will call, mapped to what asks for it.

    A feed's own name rather than its URL, because that is what the reader has
    to go and edit; a provider's host names itself, since no config line
    produced it.
    """
    called = {host: "a provider module" for host in PROVIDER_HOSTS}
    for feed in config.sources.all_feeds():
        host = urlsplit(feed.url).hostname
        if host is None:
            continue
        # First feed wins: ten venue pages on one host would otherwise report
        # the last one read, which is not the one anybody wrote first.
        called.setdefault(host, f"source {feed.name!r}")
    return called


def exit_code(findings: list[Finding]) -> int:
    """What `what-do-check-config` returns.

    An `ERROR` fails. A `WARNING` does not: a switched-off feature is worth
    saying and never worth refusing over, which has been true of this module
    since it shipped and does not change because a stricter level now exists.
    """
    return 1 if any(finding.level == ERROR for finding in findings) else 0


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
        Switched-off features first, then unassigned hosts, then absent sections.
    """
    try:
        with open(path) as handle:
            raw = yaml.safe_load(handle)
    except (OSError, yaml.YAMLError):
        return check_config(config) + check_hosts(config)
    return check_config(config) + check_hosts(config) + check_sections(raw)


def format_findings(path: Any, findings: list[Finding]) -> str:
    """The report `what-do-check-config` prints."""
    if not findings:
        return f"{path}: nothing switched off, every host covered."

    width = max(len(finding.path) for finding in findings)
    lines = [f"{path}: {len(findings)} finding(s)"]
    lines.extend(
        f"  {finding.level:<7} {finding.path.ljust(width)}  {finding.detail}"
        for finding in findings
    )
    return "\n".join(lines)


def run() -> int:
    """Entry point for `what-do-check-config`."""
    import argparse

    from src.config import DEFAULT_CONFIG_PATH, load_config

    parser = argparse.ArgumentParser(
        prog="what-do-check-config",
        description="Report configuration that leaves a feature off or a host uncovered.",
    )
    parser.add_argument("config", nargs="?", default=str(DEFAULT_CONFIG_PATH))
    args = parser.parse_args()

    findings = check_config_file(load_config(args.config), args.config)
    print(format_findings(args.config, findings))
    return exit_code(findings)
