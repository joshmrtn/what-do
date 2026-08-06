"""Unit tests for HTML-to-text conversion of source descriptions."""

from __future__ import annotations

from src.utils.html import html_to_text


def test_plain_text_passes_through_unchanged():

    assert html_to_text("Join us in the taproom for trivia!") == (
        "Join us in the taproom for trivia!"
    )


def test_empty_string():

    assert html_to_text("") == ""


def test_inline_tags_vanish_without_inserting_space():
    """<u> and <b> carry no text of their own, so removing them joins nothing."""

    assert html_to_text("<u></u>Daniel Luberto dazzles") == "Daniel Luberto dazzles"
    assert html_to_text("a <b>bold</b> claim") == "a bold claim"


def test_br_becomes_a_newline_and_never_welds_words():
    """The real hazard: deleting <br> outright runs two sentences together."""

    result = html_to_text("Welcome to the real Last Call Band of Boston!<br>You will")

    assert "Boston!\nYou will" in result
    assert "Boston!You" not in result


def test_br_self_closing_variants():

    assert html_to_text("a<br/>b") == "a\nb"
    assert html_to_text("a<br />b") == "a\nb"


def test_paragraph_close_becomes_a_newline():

    result = html_to_text("<p>LOVER, The Unofficial ERAS Tour. </p><p>Taylor Swift</p>")

    assert "LOVER, The Unofficial ERAS Tour." in result
    assert "Taylor Swift" in result
    assert "Tour. Taylor" not in result


def test_nested_tags_keep_their_text():

    assert html_to_text("<p><b>Trivia</b> night</p>").strip() == "Trivia night"


def test_anchor_keeps_the_url():
    """A link is high-value information at the end of the pipeline; never drop the href."""

    assert (
        html_to_text('Tickets <a href="https://example.com/x">here</a>')
        == "Tickets here (https://example.com/x)"
    )


def test_anchor_with_single_quoted_href():

    assert (
        html_to_text("<a href='https://example.com/x'>here</a>")
        == "here (https://example.com/x)"
    )


def test_anchor_whose_text_is_already_the_url_is_not_duplicated():

    assert (
        html_to_text('<a href="https://example.com/x">https://example.com/x</a>')
        == "https://example.com/x"
    )


def test_anchor_with_no_href_keeps_its_text():

    assert html_to_text("<a>plain</a>") == "plain"


def test_bare_url_outside_markup_survives_verbatim():

    text = "Learn more at https://www.northshorenightout.com/ today"

    assert html_to_text(text) == text


def test_entities_are_unescaped():

    assert html_to_text("Fish &amp; Chips") == "Fish & Chips"
    assert html_to_text("It&#39;s on") == "It's on"


def test_nbsp_becomes_a_space_not_a_literal():

    assert html_to_text("Doors&nbsp;at 7") == "Doors at 7"


def test_raw_nbsp_is_normalised_even_without_any_markup():
    """Real feed descriptions carry U+00A0 directly, with no entity and no tags."""

    assert html_to_text("Come and enjoy.\n\xa0\nAlan Whitney") == (
        "Come and enjoy.\n \nAlan Whitney"
    )


def test_non_ascii_is_preserved():
    """En-dashes and curly quotes appear throughout the feed and carry meaning.

    Markup is included deliberately so this exercises the conversion path rather
    than the no-markup fast path.
    """
    result = html_to_text("<p>The Actors Studio – Crossroads Plaza</p><p>“pro’s”</p>")

    assert "–" in result
    assert "“pro’s”" in result


def test_runs_of_blank_lines_are_collapsed_but_paragraphs_survive():

    result = html_to_text("<p>A</p><br><br><br><p>B</p>")

    assert "A" in result and "B" in result
    assert "\n\n\n" not in result
