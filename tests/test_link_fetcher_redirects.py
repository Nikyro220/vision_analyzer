"""link_fetcher: короткие ссылки (pin.it), служебные адреса Pinterest и страница-заглушка.

Сети нет: _fetch, yt-dlp и скачивание картинок подменены."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

pytest.importorskip("aiohttp")
pytest.importorskip("yt_dlp")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inference"))

import link_fetcher as lf  # noqa: E402

PIN = "https://www.pinterest.com/pin/199425089743462631/"
FEEDBACK = PIN + "feedback/?invite_code=67bf&sender_id=1149"

GENERIC_HTML = (
    '<html><head><title>Pinterest</title><meta property="og:title" content="Pinterest">'
    '<meta property="og:description" content="Discover recipes, home ideas, style inspiration and other ideas to try.">'
    '<meta property="og:image" content="https://s.pinimg.com/webapp/stub.png"></head></html>'
)
REAL_HTML = (
    '<html><head><title>x</title><meta property="og:title" content="Man with swords reference">'
    '<meta property="og:site_name" content="Pinterest">'
    '<meta property="og:image" content="https://i.pinimg.com/736x/real.jpg"></head></html>'
)
NEWS_HTML = (
    '<html><head><meta property="og:title" content="Article">'
    '<meta property="og:image" content="https://example.com/a.jpg"></head></html>'
)


@pytest.fixture()
def net(monkeypatch):
    """pages: url -> (конечный_url, html); calls фиксирует запросы страниц и вызовы yt-dlp."""
    state = {"pages": {}, "fetches": [], "ytdlp": [], "ytdlp_result": None}

    async def fake_fetch(session, url, *, referer=None):
        state["fetches"].append(url)
        final, html, *rest = state["pages"][url]
        chain = tuple(rest[0]) if rest else tuple(dict.fromkeys((url, final)))
        return lf._Fetched(final, "text/html", "utf-8", html.encode(), chain)

    async def fake_ytdlp(url, ie_key):
        state["ytdlp"].append(url)
        if isinstance(state["ytdlp_result"], Exception):
            raise state["ytdlp_result"]
        return state["ytdlp_result"]

    async def noop(*a, **k):
        return None

    async def fake_download(session, urls, referer):
        return [(b"img", u) for u in urls], None

    monkeypatch.setattr(lf, "_fetch", fake_fetch)
    monkeypatch.setattr(lf, "_ytdlp_extract", fake_ytdlp)
    monkeypatch.setattr(lf, "_assert_public_host", noop)
    monkeypatch.setattr(lf, "_download_images", fake_download)
    return state


def _resolve(url):
    return asyncio.run(lf._resolve(None, url, None, "ru"))


@pytest.mark.parametrize("url, expected", [
    (FEEDBACK, PIN),
    ("https://pinterest.ru/pin/199425089743462631/feedback/", PIN),
    ("https://www.pinterest.com/pin/some-slug--199425089743462631/", PIN),
    ("https://pin.it/6c9iyYquH", "https://pin.it/6c9iyYquH"),
    ("https://www.pinterest.com/someone/board/", "https://www.pinterest.com/someone/board/"),
    ("https://example.com/pin/123/", "https://example.com/pin/123/"),
])
def test_canonical_url(url, expected):
    assert lf._canonical_url(url) == expected


def test_pin_it_feedback_redirect_uses_ytdlp_on_canonical_pin(net):
    net["pages"]["https://pin.it/x"] = (FEEDBACK, GENERIC_HTML)
    net["ytdlp_result"] = {"title": "Man with swords reference", "thumbnail": "https://i.pinimg.com/a.jpg"}
    link = _resolve("https://pin.it/x")
    assert net["ytdlp"] == [PIN]
    assert link.source["method"] == "yt-dlp" and link.source["title"] == "Man with swords reference"
    assert link.source["url"] == "https://pin.it/x"
    assert [u for _, u in link.images] == ["https://i.pinimg.com/a.jpg"]


def test_ytdlp_failure_falls_back_to_opengraph_of_canonical_pin_not_feedback_page(net):
    net["pages"]["https://pin.it/x"] = (FEEDBACK, GENERIC_HTML)
    net["pages"][PIN] = (PIN, REAL_HTML)
    net["ytdlp_result"] = RuntimeError("login required")
    link = _resolve("https://pin.it/x")
    assert net["fetches"] == ["https://pin.it/x", PIN]
    assert link.source["method"] == "opengraph" and link.source["title"] == "Man with swords reference"
    assert "login required" in link.source["fallback_reason"]
    assert [u for _, u in link.images] == ["https://i.pinimg.com/736x/real.jpg"]


def test_generic_pinterest_page_is_rejected_not_analyzed(net):
    net["pages"]["https://pin.it/x"] = (FEEDBACK, GENERIC_HTML)
    net["pages"][PIN] = (PIN, GENERIC_HTML)
    net["ytdlp_result"] = RuntimeError("login required")
    with pytest.raises(lf.LinkError) as exc:
        _resolve("https://pin.it/x")
    assert exc.value.code == "placeholder" and exc.value.status == 422
    assert "https://pin.it/x" in exc.value.message("ru") and "заглушк" in exc.value.message("ru")
    assert "placeholder page" in exc.value.message("en")


def test_direct_feedback_link_is_canonicalized_before_opengraph(net):
    net[("pages")][PIN] = (PIN, REAL_HTML)
    net["ytdlp_result"] = RuntimeError("boom")
    link = _resolve(FEEDBACK)
    assert net["ytdlp"] == [PIN] and net["fetches"] == [PIN]
    assert link.source["title"] == "Man with swords reference"


def test_pin_it_to_normal_pin_page_still_works(net):
    net["pages"]["https://pin.it/y"] = (PIN, REAL_HTML)
    net["ytdlp_result"] = RuntimeError("no extractor result")
    link = _resolve("https://pin.it/y")
    assert link.source["title"] == "Man with swords reference"
    assert net["fetches"] == ["https://pin.it/y"]  # страница уже скачана при выборе экстрактора — второй раз не нужна


def test_unknown_short_link_does_not_fetch_page_twice(net):
    net["pages"]["https://bit.ly/z"] = ("https://example.com/news/1", NEWS_HTML)
    link = _resolve("https://bit.ly/z")
    assert net["fetches"] == ["https://bit.ly/z"] and net["ytdlp"] == []
    assert link.source["method"] == "opengraph" and link.source["title"] == "Article"


LOGIN = "https://www.pinterest.com/login/?next=%2Fpin%2F199425089743462631%2Ffeedback%2F%3Finvite_code%3D67bf"
REFRESH_HTML = (
    '<html><head><meta http-equiv="refresh" content="0; url=' + PIN + 'sent/?invite_code=67bf"></head></html>'
)
CANONICAL_HTML = '<html><head><link rel="canonical" href="' + PIN + '"><title>Pinterest</title></head></html>'
FOREIGN_HTML = (
    '<html><head><meta property="og:url" content="https://evil.example/pin/199425089743462631/">'
    '<meta property="og:image" content="https://evil.example/a.jpg"></head></html>'
)


def test_pin_id_taken_from_login_redirect_next_param(net):
    """pin.it отправил на страницу входа: адрес пина виден только в параметре next."""
    net["pages"]["https://pin.it/x"] = (LOGIN, GENERIC_HTML, ("https://pin.it/x", LOGIN))
    net["ytdlp_result"] = {"title": "Real", "thumbnail": "https://i.pinimg.com/a.jpg"}
    link = _resolve("https://pin.it/x")
    assert net["ytdlp"] == [PIN] and link.source["method"] == "yt-dlp"


def test_pin_id_taken_from_intermediate_hop(net):
    """Пин был в промежуточном редиректе, а конечная страница — уже общая."""
    hops = ("https://pin.it/x", FEEDBACK, "https://www.pinterest.com/")
    net["pages"]["https://pin.it/x"] = ("https://www.pinterest.com/", GENERIC_HTML, hops)
    net["ytdlp_result"] = {"title": "Real", "thumbnail": "https://i.pinimg.com/a.jpg"}
    _resolve("https://pin.it/x")
    assert net["ytdlp"] == [PIN]


def test_meta_refresh_and_canonical_hints(net):
    for html in (REFRESH_HTML, CANONICAL_HTML):
        net["ytdlp"].clear()
        net["pages"]["https://pin.it/x"] = ("https://pin.it/x", html)
        net["ytdlp_result"] = {"title": "Real", "thumbnail": "https://i.pinimg.com/a.jpg"}
        _resolve("https://pin.it/x")
        assert net["ytdlp"] == [PIN], html


def test_foreign_site_cannot_point_us_to_a_pin(net):
    net["pages"]["https://short.example/x"] = ("https://short.example/x", FOREIGN_HTML)
    link = _resolve("https://short.example/x")
    assert net["ytdlp"] == [] and link.source["method"] == "opengraph"


def test_placeholder_error_carries_redirect_chain_and_ytdlp_reason(net):
    net["pages"]["https://pin.it/x"] = ("https://www.pinterest.com/", GENERIC_HTML, ("https://pin.it/x", "https://www.pinterest.com/"))
    with pytest.raises(lf.LinkError) as exc:
        _resolve("https://pin.it/x")
    detail = exc.value.params["detail"]
    assert exc.value.code == "placeholder"
    assert "https://pin.it/x -> https://www.pinterest.com/" in detail


class _Stream:
    def __init__(self, body):
        self.body = body

    async def iter_chunked(self, n):
        yield self.body


class _Resp:
    def __init__(self, status, headers, body=b""):
        self.status, self.headers, self.content_length, self.charset = status, headers, None, "utf-8"
        self.content = _Stream(body)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _Session:
    def __init__(self, routes):
        self.routes, self.asked = routes, []

    def get(self, url, allow_redirects, headers):
        self.asked.append(url)
        return self.routes[url]


def test_fetch_records_redirect_chain():
    routes = {
        "https://pin.it/x": _Resp(302, {"Location": "https://www.pinterest.com/pin/1/sent/"}),
        "https://www.pinterest.com/pin/1/sent/": _Resp(301, {"Location": "/pin/1/"}),
        "https://www.pinterest.com/pin/1/": _Resp(200, {"Content-Type": "text/html"}, b"<html></html>"),
    }
    fetched = asyncio.run(lf._fetch(_Session(routes), "https://pin.it/x"))
    assert fetched.url == "https://www.pinterest.com/pin/1/"
    assert fetched.chain == ("https://pin.it/x", "https://www.pinterest.com/pin/1/sent/", "https://www.pinterest.com/pin/1/")
