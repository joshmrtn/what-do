"""Which listing section headings are evidence, and which are noise.

A heading is the listing site's taxonomy, not a claim about the event. Some
headings say something the title does not — a bare performer name under `Music`
is genuinely a music event — and some say nothing at all.

This lives in its own module because two adapters read the same listing. The
HTML page and the ICS feed publish the same sections in different shapes, and
they disagreed for months: the HTML path filtered headings and passed the
survivor as structured metadata, while the ICS path welded every heading,
filtered or not, onto the front of the description as prose. One rule, one
place, or the next source added restates it a third way.
"""

from __future__ import annotations

#: Headings that add information a title may lack.
#:
#: `Karaoke & trivia` names two activities and every event beneath it already
#: says which one it is, so the heading only ever contradicted the title.
#: `Other` is a null bucket. Both are dropped rather than passed to a model
#: that will treat them as evidence.
CARRIED_CATEGORIES = frozenset({"Music", "Sports"})


def category_metadata(category: str | None) -> dict[str, str]:
    """The section heading as structured data, when it is worth carrying.

    Structured rather than prose so the extraction prompt can name the field and
    say what it is. Folded into the description it reads as event copy, and
    normalization collapses the blank line that separated them, leaving the
    model `Category: Music The Brethren is what happens when...`.

    Args:
        category: The section heading the listing filed this event under.

    Returns:
        A `listing_category` entry, or an empty dict when the heading says
        nothing worth telling a model about.
    """
    if category in CARRIED_CATEGORIES:
        return {"listing_category": category}
    return {}
