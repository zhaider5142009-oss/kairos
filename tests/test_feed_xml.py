"""A feed is a document from somebody else's server, parsed inside this process.

Two defects this pins down. The first is entity expansion: `xml.etree` expands
internal entities, so a two-kilobyte document can declare ten nested entities
and ask the parser for gigabytes of text — a headline feed is not a trusted
config file. Both feed parsers now go through `_feed_xml`, which refuses
entities while still accepting the plain `<!DOCTYPE rss …>` that real feeds
carry. The second is the read: `_http_get` used to hold whatever a host sent,
so a redirect to a huge file was a memory exhaustion. A feed now has a ceiling,
and a body past it is refused without being retried (the host would send the
same bytes again).
"""

from __future__ import annotations

import urllib.request

import pytest

from kairos import data as D
from kairos import news as N
from kairos.netio import PayloadTooLarge, retryable

ENTITY_FEED = (b'<?xml version="1.0"?>'
               b'<!DOCTYPE rss [<!ENTITY a "aaaaaaaaaa">]>'
               b'<rss version="2.0"><channel><item>'
               b'<title>&a;</title><description>&a;</description>'
               b'</item></channel></rss>')

DOCTYPE_FEED = (b'<?xml version="1.0"?>'
                b'<!DOCTYPE rss PUBLIC "-//Netscape Communications//DTD RSS 0.91//EN" '
                b'"http://my.netscape.com/publish/formats/rss-0.91.dtd">'
                b'<rss version="2.0"><channel><item>'
                b'<title>gold steadies as dollar slips</title>'
                b'<pubDate>Tue, 21 Oct 2026 00:00:00 GMT</pubDate>'
                b'<link>https://example.test/gold</link>'
                b'</item></channel></rss>')


class _Body:
    """A response that records how much was asked of it."""

    def __init__(self, body: bytes) -> None:
        self.body = body
        self.asked: list[int] = []

    def read(self, n: int = -1) -> bytes:
        self.asked.append(n)
        return self.body if n < 0 else self.body[:n]

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False


# ------------------------------------------------------------ entity refusal
def test_a_feed_that_declares_entities_is_refused_whole(monkeypatch):
    bodies = {"https://hostile.test/rss": ENTITY_FEED,
              "https://honest.test/rss": DOCTYPE_FEED}
    monkeypatch.setattr(D, "_http_get", lambda url, **kw: bodies[url])
    svc = D.NewsService()
    svc.feeds = [{"name": "hostile", "url": "https://hostile.test/rss"},
                 {"name": "honest", "url": "https://honest.test/rss"}]

    items = svc.fetch(limit_per_feed=10)

    # the hostile feed contributes nothing, and the honest one is still
    # attributed to itself — refusing a feed must not shift the pairing
    assert [i.source for i in items] == ["honest"]
    assert items[0].title == "gold steadies as dollar slips"
    assert all("aaaaaaaaaa" not in i.title + i.summary for i in items)


def test_a_plain_doctype_still_parses():
    # some real feeds ship a DTD reference; that is not an entity declaration
    # and must not cost the feed
    items = D._feed_xml(DOCTYPE_FEED)
    assert items is not None
    assert items.findtext(".//title") == "gold steadies as dollar slips"


def test_the_google_news_parser_refuses_entities_too():
    assert N._parse_rss(ENTITY_FEED) == []
    heads = N._parse_rss(DOCTYPE_FEED)
    assert [h.title for h in heads] == ["gold steadies as dollar slips"]
    assert heads[0].source == ""


# ------------------------------------------------------------------ size cap
def test_an_oversize_feed_is_cut_off_and_not_retried(monkeypatch):
    body = _Body(b"x" * (D.FEED_MAX_BYTES + 10))
    calls: list[str] = []

    def fake_urlopen(req, timeout=None):
        calls.append(req.full_url)
        return body

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    with pytest.raises(PayloadTooLarge):
        D._http_get("https://hostile.test/big", retries=2,
                    max_bytes=D.FEED_MAX_BYTES)

    assert len(calls) == 1, "an oversized body was downloaded more than once"
    assert body.asked == [D.FEED_MAX_BYTES + 1]
    assert retryable(PayloadTooLarge("https://hostile.test/big", 8)) is False


def test_a_feed_under_the_cap_is_returned_whole(monkeypatch):
    body = _Body(DOCTYPE_FEED)
    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda req, timeout=None: body)

    out = D._http_get("https://honest.test/rss", retries=0,
                      max_bytes=D.FEED_MAX_BYTES)

    assert out == DOCTYPE_FEED
