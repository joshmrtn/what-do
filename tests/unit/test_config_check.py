"""Unit tests for the config completeness check."""

import ast
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlsplit

from src.config import (
    AppConfig,
    ComfortCurve,
    FeedConfig,
    NetworkPolicy,
    LocationConfig,
    NetworkConfig,
    ScoringConfig,
    ScrapingConfig,
    SourcesConfig,
    SyntheticActivityRule,
    SyntheticConditions,
    VenueDiscoveryConfig,
    WeatherConfig,
)
from src.config_check import (
    ERROR,
    INFO,
    NEVER_FETCHED,
    PROVIDER_HOSTS,
    WARNING,
    CALL_SITE_POLICIES,
    Finding,
    check_config,
    check_config_file,
    check_hosts,
    check_sections,
    exit_code,
)

SRC = Path(__file__).resolve().parents[2] / "src"


def _config(**overrides) -> AppConfig:
    """A config with **every** collection populated.

    Deliberately *not* the dataclass defaults, which are empty: a sparse fixture
    makes every test a finding and leaves none able to show the absence of one.
    The first draft populated only weather and scoring and reported ten
    findings — the fixture was the bug, not the check, and fixing it is what
    lets this stay judgement-free. Every empty collection is reported, full
    stop; there is no allowlist, on the same terms as `schema_check`.
    """
    feed = FeedConfig(name="x", url="https://example.test/feed.ics", source_type="x")
    fields = dict(
        location=LocationConfig(
            latitude=42.5,
            longitude=-70.9,
            postal_code="01970",
            search_radius_miles=25.0,
            timezone="America/New_York",
        ),
        scraping=ScrapingConfig(),
        venue_discovery=VenueDiscoveryConfig(),
        weather=WeatherConfig(
            comfort={"temperature_f": ComfortCurve(
                ideal=(20.0, 65.0), zero=(-15.0, 78.0), floor=(-40.0, 95.0)
            )},
            condition_penalty={"rain": -0.4},
        ),
        scoring=ScoringConfig(domain_map={"cinema_veezi": "movies"}),
        network=NetworkConfig(
            policies={
                # Declared but assigned to no host, on purpose: it is named at
                # the image fetcher's call site, because those hosts arrive from
                # fetched data and cannot be listed.
                "data_derived": NetworkPolicy(
                    min_interval_seconds=1.0,
                    timeout_seconds=15.0,
                    max_attempts=2,
                    backoff_base_seconds=1.0,
                    backoff_max_seconds=30.0,
                    cache_ttl=timedelta(days=7),
                ),
                "tmdb": NetworkPolicy(
                    min_interval_seconds=0.05,
                    timeout_seconds=30.0,
                    max_attempts=3,
                    backoff_base_seconds=1.0,
                    backoff_max_seconds=60.0,
                    cache_ttl=timedelta(days=7),
                )
            },
            # Every host this config will call, because that is now part of
            # being a correct config: the feed above, and every host a provider
            # module hardcodes. One policy serves them all here — the assignment
            # is what is being tested, not the numbers.
            hosts={host: "tmdb" for host in [*PROVIDER_HOSTS, "example.test"]},
        ),
        sources=SourcesConfig(
            ics_calendars=[feed], html_calendars=[feed], veezi_cinemas=[feed],
            cabot_listings=[feed], tribe_calendars=[feed], do617_venues=[feed],
            moon_feeds=[feed], assabet_feeds=[feed], jsonld_pages=[feed],
        ),
        synthetic_activities=[
            SyntheticActivityRule(
                name="Evening walk",
                conditions=SyntheticConditions(),
                tags=["walking"],
                summary="A pleasant evening walk",
            )
        ],
    )
    fields.update(overrides)
    return AppConfig(**fields)


def _paths(findings) -> list[str]:
    return [f.path for f in findings]


def test_a_fully_populated_config_reports_nothing():
    """The whole point. A check that cries wolf on a correct config is worse
    than no check, because the one real finding arrives in a crowd."""
    assert check_config(_config()) == []


def test_empty_comfort_curves_are_reported():
    """The 2026-08-16 defect: the live weather section held only `provider`, so
    every weather adjustment was 0.0 on every ranking ever stored."""
    findings = check_config(_config(weather=WeatherConfig(condition_penalty={"rain": -0.4})))

    assert "weather.comfort" in _paths(findings)


def test_an_empty_domain_map_is_reported():
    """The same defect's quieter half: every [movies] preference inert against
    621 cinema events."""
    findings = check_config(_config(scoring=ScoringConfig()))

    assert "scoring.domain_map" in _paths(findings)


def test_a_switched_off_feature_is_a_warning():
    findings = check_config(_config(scoring=ScoringConfig()))

    assert [f.level for f in findings] == [WARNING]


def test_the_finding_says_what_is_in_force():
    """"Say what it fell back to" — the value in force is the actionable half."""
    findings = check_config(_config(scoring=ScoringConfig()))

    assert "{}" in findings[0].detail


def test_a_populated_collection_is_never_a_finding():
    findings = check_config(_config())

    assert not any("condition_penalty" in p for p in _paths(findings))


def test_a_scalar_default_is_never_a_finding():
    """`gate_midpoint` 0.6 is a default working as intended. This check has no
    opinion on values — only on features that are off."""
    findings = check_config(_config(scoring=ScoringConfig(domain_map={"a": "movies"})))

    assert _paths(findings) == []


def test_no_identity_assignments_is_not_a_finding():
    """Empty here means every source is `auto`, which is the working default —
    not a feature that is off. The one exemption to "every empty collection is
    reported", and it earns it by being the only collection whose empty state is
    the intended one."""
    findings = check_config(_config(sources=SourcesConfig()))

    assert "sources.identity" not in _paths(findings)


def test_the_identity_exemption_does_not_cover_its_neighbours():
    """An exemption that swallowed the rest of `sources` would hide a genuinely
    empty feed list, which is the shape that means a source silently never runs."""
    findings = check_config(_config(sources=SourcesConfig()))

    assert "sources.ics_calendars" in _paths(findings)


def test_nested_paths_are_dotted():
    findings = check_config(_config(weather=WeatherConfig()))

    assert all("." in path for path in _paths(findings))


def test_a_top_level_empty_collection_is_reported_too():
    """`synthetic_activities` lives on AppConfig itself, not in a section."""
    findings = check_config(_config(synthetic_activities=[]))

    assert "synthetic_activities" in _paths(findings)


class TestAbsentSections:
    """A whole section missing from the file, which the loaded object cannot show.

    Distinct from an empty collection: those are legible from the config alone,
    but "the file never mentioned `view`" is only knowable from the raw mapping.
    Reported quietly — every key in an absent section has a default, and most of
    those defaults are correct.
    """

    def test_an_absent_section_is_reported(self):
        findings = check_sections({"location": {}, "scoring": {}})

        assert "view" in [f.path for f in findings]

    def test_an_absent_section_is_only_an_info(self):
        findings = check_sections({"location": {}})

        assert {f.level for f in findings} == {INFO}

    def test_a_present_section_is_not_reported(self):
        findings = check_sections({"view": {"limit": 25}})

        assert "view" not in [f.path for f in findings]

    def test_a_partial_section_is_not_reported(self):
        """Its keys have defaults, and this check does not inventory them —
        that is what needs the field-to-YAML mapping which does not exist."""
        findings = check_sections({"weather": {"provider": "open-meteo"}})

        assert "weather" not in [f.path for f in findings]

    def test_a_file_naming_every_section_reports_nothing(self):
        import dataclasses

        from src.config import AppConfig

        every = {f.name: {} for f in dataclasses.fields(AppConfig)}

        assert check_sections(every) == []

    def test_a_value_that_comes_from_the_environment_is_never_reported(self):
        """`ollama_host`, `gemini_api_key` and `gemini_model` are read from
        `.env`, never from YAML, so a file-absence check would report all three
        against every config that has ever existed — including the example.
        Three permanent false positives is how a check teaches people to ignore
        it.

        Told apart without a list of names: a *section* is a nested config block
        or a list of them. A bare scalar on `AppConfig` is not one.
        """
        findings = check_sections({})

        assert {"ollama_host", "gemini_api_key", "gemini_model"}.isdisjoint(
            f.path for f in findings
        )

    def test_a_scalar_section_is_not_reported_either(self):
        """`day_starts_at` is a YAML key, but it is a bare time with a sane
        default — the per-key inventory this check deliberately does not do."""
        findings = check_sections({})

        assert "day_starts_at" not in [f.path for f in findings]

    def test_the_real_sections_are_still_reported(self):
        findings = check_sections({})
        paths = [f.path for f in findings]

        assert "view" in paths and "weather" in paths
        assert "synthetic_activities" in paths


class TestBothChecksTogether:
    """What the batch actually calls: one function, both questions.

    Two seams meant the section half read the machine's real `config.yaml` in
    every scheduler test that did not pass `--config` — the environmental
    dependency `de50499` removed from the config tests once already.
    """

    def test_it_reports_absent_sections_from_the_file(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("location: {}\n")

        findings = check_config_file(_config(), path)

        assert "view" in [f.path for f in findings]

    def test_it_reports_switched_off_features_from_the_config(self, tmp_path):
        path = tmp_path / "config.yaml"
        path.write_text("location: {}\n")

        findings = check_config_file(_config(scoring=ScoringConfig()), path)

        assert "scoring.domain_map" in [f.path for f in findings]

    def test_an_unreadable_file_still_reports_the_loaded_config(self, tmp_path):
        """The loader has already failed loudly about the file by the time
        anything reaches here; a second complaint is noise. What was loaded is
        still worth checking."""
        findings = check_config_file(_config(scoring=ScoringConfig()), tmp_path / "gone.yaml")

        assert [f.path for f in findings] == ["scoring.domain_map"]


def _network(hosts: dict[str, str], policies: list[str] | None = None) -> NetworkConfig:
    """A network config naming `policies`.

    Defaults to the ones `hosts` uses **plus the call-site policies**, so a test
    about host assignment is not also a test about `data_derived`. A test that
    means to leave one out names the whole declared set itself.
    """
    declared = (
        policies
        if policies is not None
        else sorted(set(hosts.values()) | set(CALL_SITE_POLICIES))
    )
    return NetworkConfig(
        policies={
            name: NetworkPolicy(
                min_interval_seconds=1.0,
                timeout_seconds=30.0,
                max_attempts=3,
                backoff_base_seconds=2.0,
                backoff_max_seconds=60.0,
                cache_ttl=timedelta(hours=6),
            )
            for name in declared
        },
        hosts=hosts,
    )


def _covered(extra_hosts: tuple[str, ...] = ()) -> dict[str, str]:
    """Every host that must be assigned, plus any the test adds."""
    return {host: "everything" for host in [*PROVIDER_HOSTS, "example.test", *extra_hosts]}


class TestHostCoverage:
    """Every host this config will call, and whether it has a policy.

    `for_host` already refuses an unassigned host — but only the *first* one
    reached, at the moment it is reached, which for a seasonal source is months
    away. This answers the same question about all of them, before anything is
    fetched.
    """

    def test_a_complete_config_reports_nothing(self):
        """The whole point, again: a check that fires on a correct config is one
        people learn to skip."""
        assert check_hosts(_config()) == []

    def test_a_feed_host_with_no_policy_is_an_error(self):
        feed = FeedConfig(name="new", url="https://unassigned.test/x.ics", source_type="new")
        config = _config(sources=SourcesConfig(ics_calendars=[feed]))

        findings = check_hosts(config)

        assert [(f.level, f.path) for f in findings] == [(ERROR, "network.hosts.unassigned.test")]

    def test_every_unassigned_host_is_named_not_just_the_first(self):
        """The reason this exists rather than leaving it to `for_host`: that one
        raises on whichever host is reached first and says nothing about the
        rest, so a config with four holes takes four runs to fix."""
        feeds = [
            FeedConfig(name="a", url="https://one.test/a.ics", source_type="a"),
            FeedConfig(name="b", url="https://two.test/b.ics", source_type="b"),
        ]
        config = _config(sources=SourcesConfig(ics_calendars=feeds))

        paths = _paths(check_hosts(config))

        assert paths == ["network.hosts.one.test", "network.hosts.two.test"]

    def test_a_provider_host_no_feed_mentions_is_checked_too(self):
        """The half config cannot see. TMDb's host is hardcoded in `movies.py`,
        so nothing in `sources:` names it and an unassigned one would surface
        only when a film was next looked up."""
        hosts = _covered()
        del hosts["api.themoviedb.org"]

        paths = _paths(check_hosts(_config(network=_network(hosts))))

        assert paths == ["network.hosts.api.themoviedb.org"]

    def test_the_finding_says_where_the_host_came_from(self):
        """A hostname alone sends the reader hunting for who asks for it."""
        feed = FeedConfig(name="new_venue", url="https://unassigned.test/x", source_type="v")
        config = _config(sources=SourcesConfig(html_calendars=[feed]))

        detail = check_hosts(config)[0].detail

        assert "new_venue" in detail

    def test_a_host_assigned_to_a_policy_that_does_not_exist_is_an_error(self):
        """A typo that would otherwise surface when that host is next fetched."""
        hosts = _covered()
        hosts["example.test"] = "web_listings"

        findings = check_hosts(
            _config(network=_network(hosts, policies=["everything", "data_derived"]))
        )

        assert [(f.level, f.path) for f in findings] == [(ERROR, "network.hosts.example.test")]
        assert "web_listings" in findings[0].detail

    def test_a_policy_named_at_a_call_site_must_be_declared(self):
        """`data_derived` is assigned to no host by design — an image URL points
        at whatever CDN a venue uses — so nothing else in this check would
        notice it missing until an image was fetched."""
        network = _network(_covered(), policies=["everything"])

        findings = check_hosts(_config(network=network))

        assert [(f.level, f.path) for f in findings] == [(ERROR, "network.policies.data_derived")]

    def test_a_policy_nothing_uses_is_reported_as_dead_config(self):
        network = _network(_covered(), policies=["everything", "data_derived", "nobody_uses_me"])

        findings = check_hosts(_config(network=network))

        assert [(f.level, f.path) for f in findings] == [
            (WARNING, "network.policies.nobody_uses_me")
        ]

    def test_a_call_site_policy_is_not_dead_config(self):
        """It is used, just not by a host — which is the one thing a host-based
        reading of "used" cannot see."""
        network = _network(_covered(), policies=["everything", "data_derived"])

        assert check_hosts(_config(network=network)) == []

    def test_the_batch_sees_host_findings_too(self, tmp_path):
        """One function, both questions — the batch calls `check_config_file`."""
        path = tmp_path / "config.yaml"
        path.write_text("location: {}\n")
        feed = FeedConfig(name="a", url="https://unassigned.test/a", source_type="a")

        findings = check_config_file(_config(sources=SourcesConfig(ics_calendars=[feed])), path)

        assert "network.hosts.unassigned.test" in _paths(findings)


class TestExitCode:
    """What `what-do-check-config` returns, which is the part a script reads."""

    def test_an_error_fails(self):
        assert exit_code([Finding(level=ERROR, path="x", detail="d")]) == 1

    def test_a_warning_does_not(self):
        """A switched-off feature is worth saying and not worth refusing over —
        that has been true since this module shipped, and stays true."""
        assert exit_code([Finding(level=WARNING, path="x", detail="d")]) == 0

    def test_nothing_at_all_passes(self):
        assert exit_code([]) == 0


class TestTheRegistryCannotDrift:
    """`PROVIDER_HOSTS` is a second artefact, and the trap that comes with one.

    It is the same shape as `_SCHEMA` against the live database: a list somebody
    has to remember to update is a list that will be wrong, and wrong here means
    a host silently exempt from the coverage check — the exact hole the check
    exists to close. So it is compared against what `src/` actually names.
    """

    def _host_constants(self) -> dict[str, str]:
        """Every module-level `*_HOST = "..."` in `src/`."""
        found: dict[str, str] = {}
        for path in sorted(SRC.rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Constant):
                    continue
                if not isinstance(node.value.value, str):
                    continue
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id.endswith("_HOST"):
                        found[target.id] = node.value.value
        return found

    def _literal_hosts(self) -> set[str]:
        """Every host appearing in a URL literal in `src/`."""
        found: set[str] = set()
        for path in sorted(SRC.rglob("*.py")):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if node.value.startswith(("http://", "https://")):
                        host = urlsplit(node.value).hostname
                        if host:
                            found.add(host)
        return found

    def test_every_host_constant_is_in_the_registry(self):
        """A provider that declares its host and is not registered would be
        checked by nothing."""
        assert set(self._host_constants().values()) <= set(PROVIDER_HOSTS)

    def test_every_host_written_into_a_url_is_accounted_for(self):
        """The other way in: a module that hardcodes a URL rather than naming a
        constant. Each one is either a host we call — and so registered — or one
        we demonstrably do not, recorded with its reason."""
        unaccounted = self._literal_hosts() - set(PROVIDER_HOSTS) - set(NEVER_FETCHED)

        assert unaccounted == set(), f"unregistered host(s) in src/: {sorted(unaccounted)}"

    def test_nothing_is_excused_that_is_actually_called(self):
        """The exclusion list is where this check would go quietly wrong."""
        assert set(NEVER_FETCHED).isdisjoint(PROVIDER_HOSTS)
