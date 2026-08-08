"""Unit tests for the RSS feed parser."""

from datetime import datetime, timezone

import pytest

from src.ingestion.rss import parse_rss


def _feed(*items: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0" xmlns:content="http://purl.org/rss/1.0/modules/content/">
      <channel>
        <title>A Venue</title>
        <link>https://example.org</link>
        {''.join(items)}
      </channel>
    </rss>"""


def _item(
    title: str = "6/27/26: Viraya at Felt Fanatic",
    link: str = "https://example.org/shows/62726-viraya",
    description: str = "Catch Viraya on 6/27/26 at 6pm at Felt Fanatic",
    pub_date: str | None = "Wed, 20 May 2026 19:12:56 +0000",
    guid: str | None = "5f2a1b",
    cdata: bool = False,
) -> str:
    body = f"<![CDATA[{description}]]>" if cdata else description
    date_tag = f"<pubDate>{pub_date}</pubDate>" if pub_date else ""
    guid_tag = f"<guid isPermaLink='false'>{guid}</guid>" if guid else ""
    return f"""<item>
      <title>{title}</title>
      <link>{link}</link>
      <description>{body}</description>
      {date_tag}{guid_tag}
    </item>"""


class TestParsing:
    def test_reads_one_item(self) -> None:
        items = parse_rss(_feed(_item()))

        assert len(items) == 1
        item = items[0]
        assert item.title == "6/27/26: Viraya at Felt Fanatic"
        assert item.link == "https://example.org/shows/62726-viraya"
        assert item.description == "Catch Viraya on 6/27/26 at 6pm at Felt Fanatic"
        assert item.guid == "5f2a1b"

    def test_reads_every_item_in_feed_order(self) -> None:
        items = parse_rss(_feed(_item(title="first"), _item(title="second")))

        assert [item.title for item in items] == ["first", "second"]

    def test_reads_the_publication_date(self) -> None:
        items = parse_rss(_feed(_item()))

        assert items[0].published_at == datetime(2026, 5, 20, 19, 12, 56, tzinfo=timezone.utc)

    def test_a_cdata_description_arrives_unwrapped(self) -> None:
        items = parse_rss(_feed(_item(cdata=True)))

        assert items[0].description == "Catch Viraya on 6/27/26 at 6pm at Felt Fanatic"

    def test_a_cdata_description_keeps_its_markup(self) -> None:
        """Some items are an image and nothing else; the markup is the content."""
        items = parse_rss(_feed(_item(description='<figure class="sqs"><img/></figure>', cdata=True)))

        assert items[0].description == '<figure class="sqs"><img/></figure>'

    def test_entities_are_resolved(self) -> None:
        items = parse_rss(_feed(_item(title="Fat Randy &amp; Friends")))

        assert items[0].title == "Fat Randy & Friends"

    def test_reads_categories(self) -> None:
        feed = _feed(
            _item().replace("</item>", "<category>Music</category><category>Live</category></item>")
        )

        assert parse_rss(feed)[0].categories == ["Music", "Live"]


class TestDegrading:
    def test_an_empty_feed_yields_nothing(self) -> None:
        assert parse_rss(_feed()) == []

    def test_an_item_without_a_publication_date_is_kept(self) -> None:
        items = parse_rss(_feed(_item(pub_date=None)))

        assert len(items) == 1
        assert items[0].published_at is None

    def test_an_unreadable_publication_date_does_not_lose_the_item(self) -> None:
        items = parse_rss(_feed(_item(pub_date="sometime last spring")))

        assert len(items) == 1
        assert items[0].published_at is None

    def test_an_item_without_a_guid_falls_back_to_its_link(self) -> None:
        items = parse_rss(_feed(_item(guid=None)))

        assert items[0].guid == "https://example.org/shows/62726-viraya"

    def test_an_item_without_a_title_is_dropped(self) -> None:
        # Nothing downstream can identify or extract from a titleless item.
        feed = _feed(_item().replace("<title>6/27/26: Viraya at Felt Fanatic</title>", ""))

        assert parse_rss(feed) == []

    def test_malformed_xml_raises(self) -> None:
        # A source returning HTML where a feed should be is a real failure, and
        # the ingestion service already treats a raising source as one.
        with pytest.raises(ValueError):
            parse_rss("<html><body>Not a feed</body></html>")

    def test_a_feed_with_no_channel_raises(self) -> None:
        with pytest.raises(ValueError):
            parse_rss("<?xml version='1.0'?><rss version='2.0'></rss>")
