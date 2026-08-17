"""Unit tests for the config completeness check."""

from src.config import (
    AppConfig,
    ComfortCurve,
    FeedConfig,
    LocationConfig,
    ScoringConfig,
    ScrapingConfig,
    SourcesConfig,
    SyntheticActivityRule,
    SyntheticConditions,
    VenueDiscoveryConfig,
    WeatherConfig,
)
from src.config_check import INFO, WARNING, check_config, check_sections


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
