"""
link_fetcher.py — превращает ссылку на пост в то, что понимает /analyze:
картинки + caption с контекстом поста (платформа, автор, дата, заголовок, текст).

Порядок для каждой ссылки:

  1. Если URL узнаёт экстрактор конкретной платформы yt-dlp (Instagram, X/Twitter,
     TikTok, Reddit, VK, YouTube, Facebook, Bluesky, Pinterest, ...) — метаданные и
     картинки берём оттуда. Видео НЕ скачивается: анализируется обложка (превью).
     Универсальный экстрактор Generic сознательно не используется: он ходит по
     произвольным адресам и редиректам внутри yt-dlp, где наша защита от SSRF не действует.
  2. Иначе (платформа неизвестна ИЛИ yt-dlp не справился/не нашёл картинок) — свой
     разбор страницы: прямая ссылка на картинку либо Open Graph (og:image, og:title,
     og:description). Так читаются публичные посты t.me, новостные сайты, Threads и т.п.

     Осторожно: если yt-dlp упал на платформе с логином (Instagram/Facebook без cookies),
     og:image страницы может оказаться заглушкой. Такой результат помечается в ответе
     (source.method="opengraph" + source.fallback_reason), чтобы его можно было отличить.
     Заглушку Pinterest (общая страница вместо пина) мы узнаём сами и отклоняем ошибкой
     placeholder — анализировать её как «пост» нельзя: результат был бы «безопасно».

  Короткие ссылки (pin.it и т. п.): платформа видна только после редиректов, поэтому для
  адреса, который не знает ни один экстрактор, страница скачивается сразу, а экстрактор
  подбирается по конечному адресу. Pinterest отдаёт из pin.it адрес вида
  /pin/<id>/feedback/?invite_code=… — это страница «приглашения», без входа она показывает
  общую заглушку; поэтому адрес сводится к каноническому /pin/<id>/ (_canonical_url).

Защита от SSRF (сервер ходит по ссылкам клиента): допускаются только http/https на
публичные адреса. Адреса-литералы проверяются на каждом шаге редиректа, имена хостов —
в момент подключения (см. _PublicOnlyResolver), так что подмена DNS между проверкой и
подключением ничего не даёт. Картинки, на которые ссылается страница (og:image и т.п.),
качаются тем же защищённым путём. Остаточный риск: сам yt-dlp ходит в сеть своим кодом
(но только для адресов, которые узнал экстрактор платформы; имя хоста перед этим
дополнительно проверяется на публичность).

Модуль возвращает сырые байты картинок — детект mime и апскейл делает analyze.py
(_prepare_image), как для обычных загрузок.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import re
import shutil
import socket
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import NamedTuple
from urllib.parse import unquote, urljoin, urlsplit

import aiohttp
from aiohttp import abc as aiohttp_abc

import config

try:
    import yt_dlp
    from yt_dlp.extractor import gen_extractor_classes
except ImportError:  # опциональная зависимость: без неё остаются прямые ссылки и Open Graph
    yt_dlp = None
    gen_extractor_classes = None
    logging.warning(
        "link_fetcher: пакет yt-dlp не установлен — ссылки на посты соцсетей будут читаться "
        "только через Open Graph (pip install yt-dlp)"
    )


if config.LINKS_COOKIES_FILE and not os.path.isfile(config.LINKS_COOKIES_FILE):
    logging.warning(
        "link_fetcher: VISION_ANALYZER_LINKS_COOKIES указывает на несуществующий файл %r — "
        "yt-dlp будет падать, пока его не создадут", config.LINKS_COOKIES_FILE,
    )

_USER_AGENT = "Mozilla/5.0 (compatible; VisionAnalyzer/1.0)"
_MAX_REDIRECTS = 5
_HTML_MAX_BYTES = 2 * 1024 * 1024  # og-теги лежат в <head>, больше читать незачем
_HTML_TYPES = {"", "text/html", "application/xhtml+xml", "text/plain"}
_IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "gif", "bmp", "avif"}
_DEFAULT_CAPTION_LIMIT = 2000  # как prompt._CAPTION_MAX_CHARS, если prompt недоступен
_PINTEREST_HOST_RE = re.compile(r"(?:[\w-]+\.)*pinterest\.[a-z]{2,3}(?:\.[a-z]{2})?")
_PINTEREST_PIN_RE = re.compile(r"/pin/(?:[\w-]+--)?(\d+)")
_PLACEHOLDER_TITLES = {"", "pinterest"}  # og:title общей страницы-заглушки


# ---------------------------------------------------------------------------
# Результаты и ошибки
# ---------------------------------------------------------------------------

class LinkError(Exception):
    """Ошибка обработки одной ссылки. code → ключ локали error.link_<code>,
    status — HTTP-статус, с которым ответит /analyze, если не удалось ни одной ссылки."""

    def __init__(self, code: str, status: int = 400, **params):
        super().__init__(f"{code}: {params}")
        self.code = code
        self.status = status
        self.params = params

    def message(self, lang: str | None = None) -> str:
        return config._t(f"error.link_{self.code}", lang=lang, **self.params)


@dataclass
class LinkFailure:
    url: str
    code: str
    message: str
    status: int

    def as_dict(self) -> dict:
        return {"url": self.url, "code": self.code, "error": self.message}


@dataclass
class ResolvedLink:
    url: str                                   # ссылка как её прислал клиент
    caption: str                               # контекст поста (+ caption клиента), может быть ""
    images: list[tuple[bytes, str | None]]     # (байты, откуда скачано)
    source: dict                               # описание источника для ответа /analyze


@dataclass
class _Post:
    url: str
    method: str                                # "yt-dlp" | "opengraph" | "direct"
    platform: str | None = None
    author: str | None = None
    title: str | None = None
    text: str | None = None
    date: str | None = None
    image_urls: list[str] = field(default_factory=list)
    inline_images: list[tuple[bytes, str | None]] = field(default_factory=list)
    image_kind: str = "image"                  # "image" — сама картинка, "thumbnail" — превью/обложка
    fallback_reason: str | None = None         # почему вместо yt-dlp сработал Open Graph


# ---------------------------------------------------------------------------
# Защита от SSRF
# ---------------------------------------------------------------------------

class _BlockedAddress(OSError):
    """Имя хоста указывает на непубличный адрес."""


def _is_public_ip(value: str) -> bool:
    try:
        ip = ipaddress.ip_address(value)
    except ValueError:
        return False
    if ip.version == 6 and ip.ipv4_mapped:  # ::ffff:127.0.0.1 и т.п.
        ip = ip.ipv4_mapped
    return ip.is_global and not ip.is_multicast


def _literal_ip(host: str) -> str | None:
    """IP-адрес, если host — адрес-литерал, иначе None. Понимает и «старые» формы IPv4
    (десятичную 2130706433, 0x7f000001, 127.1, восьмеричную): getaddrinfo превращает их
    в обычный адрес, так что проверять надо именно то, во что они разворачиваются."""
    try:
        return str(ipaddress.ip_address(host))
    except ValueError:
        pass
    try:
        return socket.inet_ntoa(socket.inet_aton(host))
    except OSError:
        return None


class _PublicOnlyResolver(aiohttp_abc.AbstractResolver):
    """DNS-резолвер aiohttp, отбрасывающий непубличные адреса. Соединение идёт ровно по тем
    IP, которые вернул этот резолвер, поэтому проверка и подключение не расходятся."""

    def __init__(self) -> None:
        self._inner = aiohttp.ThreadedResolver()

    async def resolve(self, host, port=0, family=socket.AF_INET):
        infos = await self._inner.resolve(host, port, family)
        public = [i for i in infos if _is_public_ip(i["host"])]
        if not public:
            raise _BlockedAddress(f"{host} resolves to a non-public address")
        return public

    async def close(self) -> None:
        await self._inner.close()


def _normalize_url(url: str) -> str:
    """Проверяет схему и адрес-литерал; возвращает URL без фрагмента.
    Ссылку без схемы (instagram.com/p/...) принимаем как https."""
    url = (url or "").strip()
    if "://" not in url:
        url = "https://" + url
    try:
        parts = urlsplit(url)
        parts.port  # noqa: B018 — ValueError на мусорном порту
    except ValueError:
        raise LinkError("invalid", url=url) from None
    if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
        raise LinkError("invalid", url=url)
    if not config.LINKS_ALLOW_PRIVATE:
        literal = _literal_ip(parts.hostname)  # None → имя хоста, его проверит резолвер при подключении
        if literal is not None and not _is_public_ip(literal):
            raise LinkError("blocked", url=url)
    return parts._replace(fragment="").geturl()


async def _assert_public_host(host: str, url: str) -> None:
    """Для пути через yt-dlp (у него свой сетевой код): имя хоста должно резолвиться
    только в публичные адреса."""
    if config.LINKS_ALLOW_PRIVATE:
        return
    try:
        infos = await asyncio.get_running_loop().getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise LinkError("fetch_failed", 502, url=url, detail=f"DNS: {e.strerror or e}") from None
    if not infos or not all(_is_public_ip(i[4][0]) for i in infos):
        raise LinkError("blocked", url=url)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class _Fetched(NamedTuple):
    url: str            # конечный URL после редиректов
    content_type: str   # без параметров, в нижнем регистре ("" если заголовка нет)
    charset: str | None
    data: bytes
    chain: tuple[str, ...] = ()  # все адреса по пути (первый — запрошенный, последний — конечный)


def _make_session() -> aiohttp.ClientSession:
    resolver = None if config.LINKS_ALLOW_PRIVATE else _PublicOnlyResolver()
    return aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(resolver=resolver),
        timeout=aiohttp.ClientTimeout(total=config.LINKS_TIMEOUT),
        headers={
            "User-Agent": _USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,image/*;q=0.9,*/*;q=0.5",
        },
    )


def _too_large(url: str) -> LinkError:
    return LinkError("too_large", 422, url=url, max_mb=config.LINKS_MAX_IMAGE_BYTES // (1024 * 1024))


async def _fetch(session: aiohttp.ClientSession, url: str, *, referer: str | None = None) -> _Fetched:
    """GET с ручным разбором редиректов (каждый шаг проверяется заново) и лимитом размера.
    HTML обрезается по _HTML_MAX_BYTES (нам нужен только <head>); всё остальное считается
    картинкой и при превышении LINKS_MAX_IMAGE_BYTES отклоняется."""
    current = url
    hops: list[str] = []
    for _ in range(_MAX_REDIRECTS + 1):
        current = _normalize_url(current)
        hops.append(current)
        headers = {"Referer": referer} if referer else {}
        try:
            async with session.get(current, allow_redirects=False, headers=headers) as resp:
                if resp.status in (301, 302, 303, 307, 308):
                    location = resp.headers.get("Location")
                    if not location:
                        raise LinkError("fetch_failed", 502, url=url, detail=f"HTTP {resp.status} without Location")
                    nxt = urljoin(current, location)
                    logging.info("link_fetcher: редирект HTTP %s: %s -> %s", resp.status, current[:200], nxt[:200])
                    current = nxt
                    continue
                if resp.status >= 400:
                    raise LinkError(
                        "fetch_failed", 502 if resp.status >= 500 else 422, url=url, detail=f"HTTP {resp.status}",
                    )

                ctype = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
                is_html = ctype in _HTML_TYPES
                if not is_html and resp.content_length and resp.content_length > config.LINKS_MAX_IMAGE_BYTES:
                    raise _too_large(url)
                cap = _HTML_MAX_BYTES if is_html else config.LINKS_MAX_IMAGE_BYTES

                data = bytearray()
                async for chunk in resp.content.iter_chunked(64 * 1024):
                    data += chunk
                    if len(data) > cap:
                        if not is_html:
                            raise _too_large(url)
                        break
                return _Fetched(current, ctype, resp.charset, bytes(data), tuple(hops))
        except LinkError:
            raise
        except aiohttp.ClientConnectorError as e:
            if isinstance(e.os_error, _BlockedAddress):
                raise LinkError("blocked", url=url) from None
            raise LinkError("fetch_failed", 502, url=url, detail=str(e.os_error or e)) from None
        except asyncio.TimeoutError:
            raise LinkError("fetch_failed", 504, url=url, detail="timeout") from None
        except aiohttp.ClientError as e:
            raise LinkError("fetch_failed", 502, url=url, detail=str(e) or type(e).__name__) from None
    raise LinkError("fetch_failed", 502, url=url, detail="too many redirects")


async def _download_images(
    session: aiohttp.ClientSession, urls: list[str], referer: str,
) -> tuple[list[tuple[bytes, str | None]], LinkError | None]:
    """Скачивает картинки параллельно. Недоступная картинка не валит весь пост —
    пропускаем; первая ошибка возвращается, чтобы показать её, если не скачалось ничего."""
    async def one(u: str):
        try:
            fetched = await _fetch(session, u, referer=referer)
        except LinkError as e:
            logging.warning("link_fetcher: не удалось скачать картинку %s: %s", u, e.params.get("detail") or e.code)
            return e
        if fetched.content_type in _HTML_TYPES:  # вместо картинки отдали страницу (заглушка/логин)
            logging.warning("link_fetcher: по %s пришёл HTML, а не картинка", u)
            return None
        return fetched.data, fetched.url

    results = await asyncio.gather(*(one(u) for u in urls))
    images = [r for r in results if isinstance(r, tuple)]
    first_error = next((r for r in results if isinstance(r, LinkError)), None)
    return images, first_error


# ---------------------------------------------------------------------------
# Open Graph / прямые ссылки на картинки
# ---------------------------------------------------------------------------

class _MetaParser(HTMLParser):
    """Собирает <meta property|name=... content=...> и <title>."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.meta: dict[str, list[str]] = {}
        self.title = ""
        self.refresh_url: str | None = None  # <meta http-equiv="refresh" content="0; url=...">
        self.canonical: str | None = None    # <link rel="canonical" href="...">
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag == "meta":
            a = {k.lower(): (v or "") for k, v in attrs}
            key = (a.get("property") or a.get("name") or "").strip().lower()
            content = a.get("content", "").strip()
            if key and content:
                self.meta.setdefault(key, []).append(content)
            if a.get("http-equiv", "").strip().lower() == "refresh" and not self.refresh_url:
                m = re.search(r"url\s*=\s*['\"]?([^'\";]+)", a.get("content", ""), re.I)
                if m:
                    self.refresh_url = m.group(1).strip()
        elif tag == "link":
            a = {k.lower(): (v or "") for k, v in attrs}
            if "canonical" in a.get("rel", "").lower().split() and a.get("href") and not self.canonical:
                self.canonical = a["href"].strip()
        elif tag == "title":
            self._in_title = True

    def handle_endtag(self, tag):
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title:
            self.title += data


_META_CHARSET_RE = re.compile(rb"""<meta[^>]+charset=["']?([\w-]+)""", re.I)


def _decode_html(data: bytes, charset: str | None) -> str:
    candidates = [charset] if charset else []
    m = _META_CHARSET_RE.search(data[:4096])
    if m:
        candidates.append(m.group(1).decode("ascii", "ignore"))
    candidates.append("utf-8")
    for enc in candidates:
        try:
            return data.decode(enc, errors="replace")
        except LookupError:  # неизвестная кодировка в заголовке/мете
            continue
    return data.decode("utf-8", errors="replace")


def _dedupe(items) -> list:
    return list(dict.fromkeys(items))


def _post_from_html(fetched: _Fetched, url: str) -> _Post:
    parser = _MetaParser()
    try:
        parser.feed(_decode_html(fetched.data, fetched.charset))
    except Exception as e:  # битая разметка не повод падать — берём, что успели разобрать
        logging.warning("link_fetcher: ошибка разбора HTML %s: %s", url, e)
    meta = parser.meta

    def first(*keys):
        for k in keys:
            if meta.get(k):
                return meta[k][0]
        return None

    host = (urlsplit(fetched.url).hostname or "").removeprefix("www.")
    author = first("article:author", "author", "twitter:creator")
    if author and author.lower().startswith("http"):  # article:author бывает ссылкой на профиль
        author = None
    date = first("article:published_time", "og:published_time")
    date = date[:10] if date and re.match(r"\d{4}-\d{2}-\d{2}", date) else None

    raw_images = (
        meta.get("og:image:secure_url", []) + meta.get("og:image", []) + meta.get("og:image:url", [])
        + meta.get("twitter:image", []) + meta.get("twitter:image:src", [])
    )
    image_urls = _dedupe(urljoin(fetched.url, u) for u in raw_images)

    return _Post(
        url=url, method="opengraph",
        platform=first("og:site_name") or host or None,
        author=author,
        title=first("og:title", "twitter:title") or parser.title.strip() or None,
        text=first("og:description", "twitter:description", "description"),
        date=date,
        image_urls=image_urls[: config.LINKS_MAX_IMAGES_PER_POST],
        image_kind="thumbnail",  # og:image — это превью, не обязательно сам снимок
    )


def _post_from_fetched(fetched: _Fetched, url: str) -> _Post:
    if fetched.content_type not in _HTML_TYPES:  # прямая ссылка на картинку
        return _Post(url=url, method="direct", inline_images=[(fetched.data, fetched.url)])
    return _post_from_html(fetched, url)


async def _post_from_page(session: aiohttp.ClientSession, url: str) -> _Post:
    return _post_from_fetched(await _fetch(session, url), url)


def _canonical_url(url: str) -> str:
    """Приводит адрес поста к виду, который показывает сам пост, а не служебную страницу.

    Pinterest: pin.it ведёт на /pin/<id>/feedback/?invite_code=…&sender_id=… — страницу «приглашения»,
    которая без входа отдаёт общую заглушку (og:title «Pinterest»), а не пин. Каноническая
    https://www.pinterest.com/pin/<id>/ — сам пин (его же понимает экстрактор yt-dlp).
    Остальные адреса возвращаются как есть.
    """
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
    except ValueError:
        return url
    if _PINTEREST_HOST_RE.fullmatch(host):
        match = _PINTEREST_PIN_RE.match(parts.path)
        if match:
            return f"https://www.pinterest.com/pin/{match.group(1)}/"
    return url


_PIN_ID_RE = re.compile(r"/pin/(?:[\w-]+--)?(\d{6,})")


def _html_hints(fetched: _Fetched) -> list[str]:
    """Адреса, на которые страница намекает сама: meta refresh, canonical, og:url."""
    if fetched.content_type not in _HTML_TYPES:
        return []
    parser = _MetaParser()
    try:
        parser.feed(_decode_html(fetched.data, fetched.charset))
    except Exception:  # noqa: BLE001 — подсказки необязательны
        return []
    hints = [parser.refresh_url, parser.canonical, *parser.meta.get("og:url", [])]
    return [urljoin(fetched.url, h) for h in hints if h]


def _find_pinterest_pin(urls: list[str]) -> str | None:
    """Канонический адрес пина, если id пина виден в одном из адресов (путь или параметры вроде
    login/?next=%2Fpin%2F<id>%2F). Смотрим только адреса самого Pinterest: чужой сайт не должен
    подсунуть нам пин. Первый подходящий адрес выигрывает — вызывающий кладёт самые надёжные первыми."""
    for url in urls:
        try:
            host = (urlsplit(url).hostname or "").lower()
        except ValueError:
            continue
        if not _PINTEREST_HOST_RE.fullmatch(host):
            continue
        match = _PIN_ID_RE.search(unquote(unquote(url)))
        if match:
            return f"https://www.pinterest.com/pin/{match.group(1)}/"
    return None


def _expanded_target(fetched: _Fetched) -> str:
    """Полный адрес поста после раскрытия короткой ссылки: пин Pinterest из цепочки редиректов
    (конечный адрес — первым) или подсказок страницы, иначе — конечный адрес редиректа."""
    pin = _find_pinterest_pin([fetched.url, *reversed(fetched.chain), *_html_hints(fetched)])
    return pin or _normalize_url(_canonical_url(fetched.url))


def _is_placeholder(post: _Post, final_url: str) -> bool:
    """Open Graph-разбор вернул общую страницу платформы, а не пост (сейчас — Pinterest)."""
    if post.method != "opengraph":
        return False
    host = (urlsplit(final_url).hostname or "").lower()
    if not _PINTEREST_HOST_RE.fullmatch(host):
        return False
    return (post.title or "").strip().lower() in _PLACEHOLDER_TITLES


# ---------------------------------------------------------------------------
# yt-dlp
# ---------------------------------------------------------------------------

class _YtdlpLogger:
    """Заглушка: yt-dlp не должен писать в stdout/stderr сервера, ошибки мы логируем сами."""

    def debug(self, msg):
        pass

    def info(self, msg):
        pass

    def warning(self, msg):
        logging.debug("yt-dlp: %s", msg)

    def error(self, msg):
        logging.debug("yt-dlp: %s", msg)


def _pick_extractor_key(url: str) -> str | None:
    """Ключ платформенного экстрактора yt-dlp для этого URL, либо None, если URL знает
    только Generic (или yt-dlp не установлен)."""
    if gen_extractor_classes is None:
        return None
    try:
        for ie in gen_extractor_classes():
            if ie.suitable(url):
                key = ie.ie_key()
                return None if key == "Generic" else key
    except Exception:
        logging.exception("link_fetcher: ошибка подбора экстрактора yt-dlp для %s", url)
    return None


def _ytdlp_extract_sync(url: str, ie_key: str) -> dict:
    opts = {
        "quiet": True,
        "no_warnings": True,
        "no_color": True,
        "logger": _YtdlpLogger(),
        "skip_download": True,                 # только метаданные, видео не качаем
        "noplaylist": True,                    # watch?v=...&list=... → одно видео
        "playlistend": config.LINKS_MAX_IMAGES_PER_POST,
        "ignore_no_formats_error": True,       # у фото-постов нет видеоформатов — это нормально
        "socket_timeout": 20,
        "retries": 1,
        "extractor_retries": 1,
        "cachedir": False,
    }
    # yt-dlp при закрытии сохраняет cookie-jar обратно в файл — параллельные запросы
    # гонялись бы за один файл, а исходный мог бы быть перезаписан. Даём ему временную копию.
    tmp_cookies = None
    try:
        if config.LINKS_COOKIES_FILE:
            fd, tmp_cookies = tempfile.mkstemp(prefix="vision_cookies_", suffix=".txt")  # права 0600
            os.close(fd)
            shutil.copyfile(config.LINKS_COOKIES_FILE, tmp_cookies)
            opts["cookiefile"] = tmp_cookies
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False, ie_key=ie_key)
            return ydl.sanitize_info(info)     # JSON-безопасный dict, entries → список
    finally:
        if tmp_cookies:
            try:
                os.unlink(tmp_cookies)
            except OSError:
                pass


async def _ytdlp_extract(url: str, ie_key: str) -> dict:
    loop = asyncio.get_running_loop()
    return await asyncio.wait_for(
        loop.run_in_executor(None, _ytdlp_extract_sync, url, ie_key), config.LINKS_TIMEOUT,
    )


def _short_error(e: BaseException) -> str:
    msg = re.sub(r"\x1b\[[0-9;]*m", "", str(e)).strip().removeprefix("ERROR: ")
    return (msg.splitlines()[0] if msg else type(e).__name__)[:300]


def _first_str(*values) -> str | None:
    for v in values:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _pick_image(item: dict) -> tuple[str, bool] | None:
    """(url, это_сама_картинка) для одного элемента yt-dlp: сначала фото-формат,
    затем лучшее превью (в sanitize_info превью отсортированы от худшего к лучшему)."""
    direct = item.get("url")
    if (item.get("ext") or "").lower() in _IMAGE_EXTS and isinstance(direct, str) and direct.startswith("http"):
        return direct, True
    thumb = item.get("thumbnail")
    if isinstance(thumb, str) and thumb.startswith("http"):
        return thumb, False
    thumbs = [t["url"] for t in item.get("thumbnails") or [] if isinstance(t, dict) and t.get("url")]
    if thumbs:
        return thumbs[-1], False
    return None


def _date_from_info(info: dict) -> str | None:
    ud = info.get("upload_date")
    if isinstance(ud, str) and re.fullmatch(r"\d{8}", ud):
        return f"{ud[:4]}-{ud[4:6]}-{ud[6:]}"
    ts = info.get("timestamp")
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%d")
        except (OverflowError, OSError, ValueError):
            return None
    return None


def _post_from_ytdlp(info: dict, url: str) -> _Post:
    entries = [e for e in (info.get("entries") or []) if isinstance(e, dict)]
    items = entries or [info]

    chosen: list[tuple[str, bool]] = []
    for item in items:
        picked = _pick_image(item)
        if picked and picked[0] not in {c[0] for c in chosen}:
            chosen.append(picked)
    if not chosen and entries:  # картинка только у «обёртки» плейлиста
        picked = _pick_image(info)
        if picked:
            chosen.append(picked)
    chosen = chosen[: config.LINKS_MAX_IMAGES_PER_POST]

    sources = [info, *entries[:1]]  # метаданные — с поста, при пустоте — с первого элемента

    def pick(*keys) -> str | None:
        for src in sources:
            value = _first_str(*(src.get(k) for k in keys))
            if value:
                return value
        return None

    name = pick("uploader", "channel", "creator", "artist")
    handle = pick("uploader_id")
    if handle and (handle.isdigit() or len(handle) > 40):  # числовой id — не для людей
        handle = None
    if handle:
        handle = handle.lstrip("@")
    if name and handle and handle.lower() not in name.lower():
        author = f"{name} (@{handle})"
    else:
        author = name or (f"@{handle}" if handle else None)

    return _Post(
        url=url, method="yt-dlp",
        platform=pick("webpage_url_domain") or (urlsplit(url).hostname or "").removeprefix("www.") or None,
        author=author,
        title=pick("title", "fulltitle"),
        text=pick("description"),
        date=next((d for d in (_date_from_info(s) for s in sources) if d), None),
        image_urls=[c[0] for c in chosen],
        image_kind="image" if chosen and all(c[1] for c in chosen) else "thumbnail",
    )


# ---------------------------------------------------------------------------
# caption
# ---------------------------------------------------------------------------

def _clean_text(s: str) -> str:
    """Многострочный текст из поста. Текст поста — недоверенные данные (его пишет тот, кого
    проверяют), а prompt.py ограждает caption строками '---': серию дефисов заменяем длинным
    тире, чтобы пост не мог «закрыть» блок контекста и дописать свои инструкции."""
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)
    s = re.sub(r"-{3,}", "—", s)
    s = re.sub(r"[ \t]+\n", "\n", s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def _clean_line(s: str | None) -> str:
    """Однострочное поле (платформа, автор, заголовок): без переводов строк, чтобы значение
    не могло подделать соседние метки ("Платформа: ...")."""
    return " ".join(_clean_text(s).split()) if s else ""


def _squash(s: str) -> str:
    return " ".join(s.split())


def _title_is_redundant(title: str, text: str, author: str) -> bool:
    """Заголовок поста часто просто повторяет текст: "Автор - начало текста…" (X/Twitter),
    сам текст (TikTok) и т.п. Такой заголовок в caption только шумит."""
    t, body = _squash(title), _squash(text)
    head, sep, rest = t.partition(" - ")  # префикс "Автор - " снимаем, если это имя автора
    if sep and author and (head.lower() in author.lower() or author.lower() in head.lower()):
        t = rest
    t = t.rstrip("….").strip()  # хвост обрезанного заголовка
    return bool(t) and (t in body or body in t)


def build_caption(post: _Post, lang: str | None, explicit: str | None = None) -> str:
    """Собирает caption: [caption клиента] + строки платформа/автор/дата/заголовок + текст поста.
    Текст идёт последним и обрезается так, чтобы всё влезло в лимит caption промпта
    (prompt._CAPTION_MAX_CHARS) — иначе промпт отрежет хвост «как придётся»."""
    explicit = (explicit or "").strip()

    def label(key: str) -> str:
        return config._t(key, lang=lang)

    title = _clean_line(post.title)
    text = _clean_text(post.text) if post.text else ""
    author = _clean_line(post.author)
    platform = _clean_line(post.platform)

    if title and text and _title_is_redundant(title, text, author):
        title = ""

    if not (title or text or author):
        return explicit  # одних «платформа/дата» мало — не засоряем caption

    header = []
    if platform:
        header.append(f"{label('link.label_platform')}: {platform}")
    if author:
        header.append(f"{label('link.label_author')}: {author}")
    if post.date:
        header.append(f"{label('link.label_date')}: {post.date}")
    if title:
        header.append(f"{label('link.label_title')}: {title}")

    lines = list(header)
    if text:
        limit = getattr(getattr(config, "prompt", None), "_CAPTION_MAX_CHARS", _DEFAULT_CAPTION_LIMIT)
        text_label = f"{label('link.label_text')}:\n"
        used = len(explicit) + (2 if explicit else 0) + len("\n".join(header)) + 1 + len(text_label)
        room = max(limit - used, 200)
        if len(text) > room:
            text = text[:room].rstrip() + "…"
        lines.append(text_label + text)

    body = "\n".join(lines)
    return f"{explicit}\n\n{body}" if explicit else body


# ---------------------------------------------------------------------------
# Публичный вход
# ---------------------------------------------------------------------------

async def _resolve(
    session: aiohttp.ClientSession, raw_url: str, explicit_caption: str | None, lang: str | None,
) -> ResolvedLink:
    raw_url = raw_url.strip()
    url = _normalize_url(raw_url)
    loop = asyncio.get_running_loop()

    # Адрес, по которому работаем дальше: служебные адреса Pinterest сводятся к самому пину.
    work_url = _normalize_url(_canonical_url(url))
    trail: list[str] = []            # все адреса, по которым прошли (для диагностики в логе)
    fetched: _Fetched | None = None  # страница, уже скачанная для Open Graph (чтобы не качать дважды)
    fetched_for: str | None = None   # work_url, с которого она скачана

    ie_key = await loop.run_in_executor(None, _pick_extractor_key, work_url)
    if not ie_key:
        # Платформа может быть видна только после редиректов (pin.it и другие короткие ссылки):
        # идём по ним и подбираем экстрактор по конечному адресу. Страница всё равно нужна
        # для Open Graph, если экстрактора не окажется.
        fetched, fetched_for = await _fetch(session, work_url), work_url
        trail.extend(fetched.chain)
        logging.info("link_fetcher: %s -> %s (цепочка: %s)", work_url[:200], fetched.url[:200],
                     " -> ".join(u[:200] for u in fetched.chain))
        if fetched.content_type in _HTML_TYPES:
            target = _expanded_target(fetched)
            if target != work_url:
                target_key = await loop.run_in_executor(None, _pick_extractor_key, target)
                if target_key:
                    ie_key, work_url = target_key, target

    post: _Post | None = None
    ytdlp_error: str | None = None
    if ie_key:
        await _assert_public_host(urlsplit(work_url).hostname, work_url)
        try:
            info = await _ytdlp_extract(work_url, ie_key)
            post = _post_from_ytdlp(info, work_url)
            if not post.image_urls:
                ytdlp_error = "no images in extracted data"
                post = None
        except Exception as e:  # DownloadError/ExtractorError, таймаут и т.п. — идём в запасной путь
            ytdlp_error = _short_error(e)
        if post is None:
            logging.info("link_fetcher: yt-dlp (%s) не дал картинок для %s: %s — пробую Open Graph",
                         ie_key, work_url, ytdlp_error)

    if post is None:
        try:
            # Уже скачанную страницу берём повторно, если она и есть страница work_url
            # (либо редирект привёл ровно на него) — иначе качаем канонический адрес.
            if fetched is None or (fetched_for != work_url and _normalize_url(fetched.url) != work_url):
                fetched = await _fetch(session, work_url)
                trail.extend(fetched.chain)
        except LinkError as e:
            if ytdlp_error and e.code == "fetch_failed":
                e.params["detail"] = f"{e.params.get('detail')}; yt-dlp: {ytdlp_error}"
            raise
        post = _post_from_fetched(fetched, url)
        post.fallback_reason = ytdlp_error
        if _is_placeholder(post, fetched.url):
            detail = "цепочка: " + " -> ".join(dict.fromkeys(trail or [fetched.url]))
            if ytdlp_error:
                detail += f"; yt-dlp: {ytdlp_error}"
            raise LinkError("placeholder", 422, url=raw_url, detail=detail)

    images = list(post.inline_images)
    if post.image_urls:
        downloaded, first_error = await _download_images(session, post.image_urls, referer=url)
        images += downloaded
        if not images and first_error:
            raise first_error
    if not images:
        raise LinkError("no_image", 422, url=url)

    caption = build_caption(post, lang, explicit_caption)
    source = {
        "url": raw_url,
        "method": post.method,
        "image_kind": post.image_kind,
        "platform": post.platform,
        "author": post.author,
        "title": post.title,
        "date": post.date,
        "caption": caption,
        "fallback_reason": post.fallback_reason,
    }
    return ResolvedLink(
        url=raw_url, caption=caption, images=images,
        source={k: v for k, v in source.items() if v},
    )


async def _resolve_one(
    session: aiohttp.ClientSession, url: str, explicit_caption: str | None, lang: str | None,
) -> ResolvedLink | LinkFailure:
    try:
        return await _resolve(session, url, explicit_caption, lang)
    except LinkError as e:
        logging.warning("link_fetcher: %s — %s (%s)", url, e.code, e.params.get("detail", ""))
        return LinkFailure(url=url, code=e.code, message=e.message(lang), status=e.status)
    except Exception as e:
        logging.exception("link_fetcher: непредвиденная ошибка при обработке %s", url)
        err = LinkError("fetch_failed", 502, url=url, detail=type(e).__name__)
        return LinkFailure(url=url, code=err.code, message=err.message(lang), status=err.status)


def make_failure(code: str, status: int, lang: str | None, **params) -> LinkFailure:
    """Ошибка, найденная уже за пределами модуля (например, скачанное не оказалось картинкой)."""
    err = LinkError(code, status, **params)
    return LinkFailure(url=params.get("url", ""), code=code, message=err.message(lang), status=status)


async def resolve_links(
    items: list[tuple[str, str | None]], lang: str | None = None,
) -> list[ResolvedLink | LinkFailure]:
    """items — [(url, caption_клиента_или_None)]. Результат — по одному элементу на ссылку,
    в том же порядке: ResolvedLink либо LinkFailure. Исключений наружу не бросает."""
    async with _make_session() as session:
        return list(await asyncio.gather(*(_resolve_one(session, url, cap, lang) for url, cap in items)))
