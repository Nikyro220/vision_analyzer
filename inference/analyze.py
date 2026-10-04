"""
analyze.py — всё, что относится к POST /analyze: подготовка изображений
(детект mime + апскейл), разбор тела запроса в трёх поддерживаемых
форматах (сырые байты, JSON, multipart) и сам хендлер, который сводит
их к общему пути — вызову backends._analyze_image и сборке ответа.
Помимо самих изображений запрос может содержать ссылки на посты (поле/параметр
"url", JSON-поле "urls"): link_fetcher.py достаёт из поста картинки и контекст,
контекст уходит в caption.

Вынесено из server.py отдельным модулем, потому что это самый сложный
путь сервера, а server.py должен остаться тонким HTTP-роутингом.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
from typing import Any

import aiohttp
from aiohttp import web
from PIL import Image

import backends
import categories
import config
import link_fetcher
import providers
from config import image_upscaler, locales

# ---------------------------------------------------------------------------
# Подготовка одного изображения: детект mime + апскейл + base64
# ---------------------------------------------------------------------------

def _detect_image_mime_sync(data: bytes) -> str | None:
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        mime = Image.MIME.get(img.format)
        logging.info(
            "Pillow определил изображение: format=%s mime=%s size=%d байт",
            img.format, mime, len(data),
        )
        return mime
    except Exception as e:
        logging.warning(
            "Pillow не смог распознать содержимое как изображение (%d байт): %s",
            len(data), e,
        )


async def _detect_image_mime(data: bytes) -> str | None:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _detect_image_mime_sync, data)


async def _prepare_image(data: bytes, source_name: str) -> tuple[str, str] | None:
    """Детектит mime, при необходимости апскейлит, кодирует в base64.

    Возвращает (image_b64, mime) либо None, если data — не изображение.
    Общая точка для всех трёх веток handle_analyze, чтобы не повторять
    один и тот же порядок действий три раза.
    """
    real_mime = await _detect_image_mime(data)
    if real_mime is None:
        return None

    if image_upscaler is not None:
        data, new_mime = await image_upscaler.upscale_if_needed(data, source_name=source_name)
        if new_mime:
            real_mime = new_mime

    return base64.b64encode(data).decode("utf-8"), real_mime


# ---------------------------------------------------------------------------
# Разбор тела запроса — три формата (+ только ссылки в query), один и тот же результат:
# (tasks, names, captions, url_items) либо _BodyError с готовым web.Response.
# url_items — [(url, caption_или_None)]: ссылки на посты, они превращаются в картинки
# позже, в handle_analyze (после проверки lang/backend/categories), см. _tasks_from_links.
# ---------------------------------------------------------------------------

class _BodyError(Exception):
    """Внутренний сигнал "верни этот JSON-ответ с ошибкой и не ходи дальше"."""

    def __init__(self, response: web.Response):
        self.response = response


def _query_overrides(request: web.Request) -> dict[str, Any]:
    return {
        "backend": request.query.get("backend"),
        "model": request.query.get("model"),
        "lang": request.query.get("lang"),
        "caption": request.query.get("caption"),
        # Разовые категории на этот вызов (см. categories.build_overlay).
        # В отличие от остальных полей — список, а не строка: через
        # query/multipart можно передать НЕСКОЛЬКО категорий, каждую
        # отдельным значением/файлом (backends._parse_categories_json
        # разбирает каждый элемент как один объект категории ИЛИ как
        # JSON-массив категорий). Пустой список ("параметра нет")
        # приводим к None, чтобы работала общая логика
        # "overrides[key] = overrides[key] or body.get(key)" в
        # _parse_json_body — иначе пустой список (falsy и так, но для
        # ясности) не перекрывал бы JSON-тело.
        "categories": request.query.getall("categories", []) or None,
    }


def _query_url_items(request: web.Request, caption: str | None) -> list[tuple[str, str | None]]:
    """?url=... (можно повторять) — доступно для любого формата тела."""
    return [(u.strip(), caption) for u in request.query.getall("url", []) if u.strip()]


def _json_url_items(body: dict, default_caption: str | None) -> list[tuple[str, str | None]]:
    """"url": "..." и/или "urls": ["...", {"url": "...", "caption": "..."}] из JSON-тела."""
    raw: list[Any] = []
    if body.get("url"):
        raw.append(body["url"])
    urls = body.get("urls")
    if isinstance(urls, list):
        raw.extend(urls)
    elif isinstance(urls, str):
        raw.append(urls)

    items = []
    for item in raw:
        if isinstance(item, dict):
            url, cap = item.get("url"), item.get("caption") or default_caption
        else:
            url, cap = item, default_caption
        if isinstance(url, str) and url.strip():
            items.append((url.strip(), cap))
    return items


async def _parse_raw_image_body(request: web.Request, overrides: dict) -> tuple[list, list, list, list]:
    """Content-Type: image/* — сырые байты картинки прямо в теле."""
    data = await request.read()
    if not data:
        raise _BodyError(_json({"error": config._t("error.empty_body", lang=overrides["lang"])}, status=400))

    prepared = await _prepare_image(data, source_name="body")
    if prepared is None:
        raise _BodyError(_json({"error": config._t("error.not_image", lang=overrides["lang"])}, status=400))

    return [prepared], ["body"], [overrides["caption"]], _query_url_items(request, overrides["caption"])


async def _parse_json_body(request: web.Request, overrides: dict) -> tuple[list, list, list, list]:
    """Content-Type: application/json — удобно для UI/ботов, поддерживает
    batch с caption на каждую картинку отдельно.

    'images' — список, каждый элемент либо строка с картинкой (caption
    для неё общий, из overrides['caption']), либо объект
    {"image": "...", "caption": "..."} — свой caption на эту картинку.

    'url' / 'urls' — ссылки на посты (строка либо {"url": "...", "caption": "..."});
    можно вместе с 'images'. Контекст поста дописывается к caption автоматически.
    """
    body = await request.json() or {}
    for key in overrides:
        overrides[key] = overrides[key] or body.get(key)

    raw_images = body.get("images")
    if not raw_images:
        single = body.get("image")
        raw_images = [single] if single else []

    url_items = _json_url_items(body, overrides["caption"]) + _query_url_items(request, overrides["caption"])

    if not raw_images and not url_items:
        raise _BodyError(_json({"error": config._t("error.empty_body", lang=overrides["lang"])}, status=400))

    tasks, names, captions = [], [], []
    for idx, item in enumerate(raw_images):
        if isinstance(item, dict):
            img, item_caption = item.get("image"), item.get("caption") or overrides["caption"]
        else:
            img, item_caption = item, overrides["caption"]

        try:
            data = base64.b64decode(providers.strip_data_url(img))
        except Exception:
            raise _BodyError(_json({"error": config._t("error.not_image", lang=overrides["lang"])}, status=400))

        source_name = f"json_{idx + 1}"
        prepared = await _prepare_image(data, source_name=source_name)
        if prepared is None:
            raise _BodyError(_json({"error": config._t("error.not_image", lang=overrides["lang"])}, status=400))

        tasks.append(prepared)
        names.append(source_name)
        captions.append(item_caption)

    return tasks, names, captions, url_items


# Текстовые поля-переопределения в multipart-запросе → куда класть
# значение в overrides. True — значение стрипается (лишние пробелы по
# краям не нужны), False — не стрипается (в тексте поста пробелы могут
# быть значимыми). 'categories' сюда не входит — она может повторяться
# (несколько файлов за один вызов), см. _parse_multipart_body.
_MULTIPART_TEXT_FIELDS = {
    "backend": True, "model": True, "lang": True, "caption": False,
}


async def _parse_multipart_body(request: web.Request, overrides: dict) -> tuple[list, list, list, list]:
    """Content-Type: multipart/form-data — старый путь, одно или несколько
    полей 'images'/'image' плюс текстовые поля-переопределения.

    'categories' — необязательно, можно прикрепить НЕСКОЛЬКО раз за один
    запрос (каждая часть — файл с одним объектом категории, как
    inference/categories/<имя>.json на сервере; допускается и часть с
    JSON-массивом сразу нескольких — backends._parse_categories_json
    разберёт оба варианта). Клиенту не нужно ничего парсить самому —
    файл просто прикрепляется как есть.

    'url' — ссылка на пост (текстовое поле), тоже можно повторять; вместо 'images' или вместе с ними.
    """
    reader = await request.multipart()
    tasks, names = [], []
    raw_categories: list[str] = []
    raw_urls: list[str] = []

    async for part in reader:
        if part.name == "url":
            raw = (await part.read(decode=True)).decode("utf-8").strip()
            if raw:
                raw_urls.append(raw)
            continue
        if part.name == "categories":
            raw = (await part.read(decode=True)).decode("utf-8").strip()
            if raw:
                raw_categories.append(raw)
            continue
        if part.name in _MULTIPART_TEXT_FIELDS:
            raw = (await part.read(decode=True)).decode("utf-8")
            overrides[part.name] = raw.strip() if _MULTIPART_TEXT_FIELDS[part.name] else raw
            continue
        if part.name not in ("images", "image"):
            continue

        data = await part.read(decode=True)
        source_name = part.filename or f"image_{len(names) + 1}"
        prepared = await _prepare_image(data, source_name=source_name)
        if prepared is None:
            logging.warning("Пропускаю не-изображение: %s", part.filename)
            continue

        tasks.append(prepared)
        names.append(source_name)

    if raw_categories:
        overrides["categories"] = raw_categories

    url_items = [(u, overrides["caption"]) for u in raw_urls] + _query_url_items(request, overrides["caption"])

    if not tasks and not url_items:
        raise _BodyError(_json(
            {"error": config._t("error.no_images_multipart", lang=overrides["lang"])}, status=400,
        ))

    return tasks, names, [overrides["caption"]] * len(tasks), url_items


async def _parse_links_only_body(request: web.Request, overrides: dict) -> tuple[list, list, list, list]:
    """Тела нет (или тип не распознан), но есть ?url=... — просто ссылки:
    curl -X POST "http://host:6769/analyze?url=https://..."."""
    return [], [], [], _query_url_items(request, overrides["caption"])


_BODY_PARSERS = (
    ("image/", _parse_raw_image_body),
    ("application/json", _parse_json_body),
    ("multipart/", _parse_multipart_body),
)


def _json(data, status: int = 200) -> web.Response:
    """web.json_response с ensure_ascii=False и indent=2 по умолчанию.

    Лежит здесь (а не в server.py), потому что server.py импортирует
    handle_analyze из этого модуля — если бы _json была в server.py,
    получился бы циклический импорт. server.py переиспользует эту же
    функцию для остальных хендлеров через `from analyze import _json`.
    """
    return web.json_response(
        data, status=status, dumps=lambda obj: json.dumps(obj, ensure_ascii=False, indent=2),
    )


# ---------------------------------------------------------------------------
# Ссылки на посты → картинки + caption
# ---------------------------------------------------------------------------

async def _tasks_from_links(
    url_items: list[tuple[str, str | None]], lang: str | None,
) -> tuple[list, list, list, list, list[link_fetcher.LinkFailure]]:
    """Разворачивает ссылки в (tasks, names, captions, sources, failures) — те же структуры,
    что дают парсеры тела, плюс sources (описание источника для ответа) и ошибки по ссылкам.

    Пост с несколькими картинками даёт несколько записей с одним и тем же caption;
    имя записи — сама ссылка (для нескольких картинок — со скрепой "#N")."""
    tasks, names, captions, sources, failures = [], [], [], [], []

    for res in await link_fetcher.resolve_links(url_items, lang):
        if isinstance(res, link_fetcher.LinkFailure):
            failures.append(res)
            continue

        accepted = 0
        for idx, (data, image_url) in enumerate(res.images, 1):
            name = res.url if len(res.images) == 1 else f"{res.url}#{idx}"
            prepared = await _prepare_image(data, source_name=name)
            if prepared is None:
                logging.warning("Ссылка %s: %s — не изображение, пропускаю", res.url, image_url)
                continue
            tasks.append(prepared)
            names.append(name)
            captions.append(res.caption or None)
            sources.append({**res.source, "image_url": image_url} if image_url else dict(res.source))
            accepted += 1

        if not accepted:
            failures.append(link_fetcher.make_failure("no_image", 422, lang, url=res.url))

    return tasks, names, captions, sources, failures


# ---------------------------------------------------------------------------
# Хендлер
# ---------------------------------------------------------------------------

async def handle_analyze(request: web.Request) -> web.Response:
    overrides = _query_overrides(request)
    content_type = request.content_type

    parser = next((fn for prefix, fn in _BODY_PARSERS if content_type.startswith(prefix)), None)
    if parser is None and request.query.getall("url", []):
        parser = _parse_links_only_body
    if parser is None:
        return _json({"error": config._t("error.unsupported_content_type", lang=overrides["lang"])}, status=400)

    try:
        tasks, names, captions, url_items = await parser(request, overrides)
    except _BodyError as e:
        return e.response

    # Разовый (per-request) язык ответа/промпта — не трогает глобальный
    # locales.DEFAULT_LANG, поэтому параллельные запросы с разными lang
    # не мешают друг другу.
    resolved_lang = None
    if overrides["lang"]:
        resolved_lang = overrides["lang"].strip().lower()
        if locales is not None and resolved_lang not in locales.LOCALES:
            return _json(
                {
                    "error": config._t("error.lang_not_loaded", requested_lang=resolved_lang),
                    "available": list(locales.LOCALES.keys()),
                },
                status=400,
            )

    backend = (overrides["backend"] or config.BACKEND).strip().lower()
    if not providers.is_known(backend):
        return _json(
            {"error": config._t("error.unknown_backend", backend=backend, lang=resolved_lang)}, status=400,
        )

    try:
        extra_categories = backends._parse_categories_json(overrides["categories"])
    except ValueError:
        return _json({"error": config._t("error.invalid_categories", lang=resolved_lang)}, status=400)

    try:
        category_overlay = categories.build_overlay(extra_categories)
    except categories.CategoryError as e:
        return _json({"error": str(e)}, status=400)

    sources: list[dict | None] = [None] * len(tasks)
    link_failures: list[link_fetcher.LinkFailure] = []
    if url_items:
        if not config.LINKS_ENABLED:
            return _json({"error": config._t("error.link_disabled", lang=resolved_lang)}, status=503)
        if len(url_items) > config.LINKS_MAX_PER_REQUEST:
            return _json(
                {"error": config._t("error.link_too_many", lang=resolved_lang, max=config.LINKS_MAX_PER_REQUEST)},
                status=400,
            )
        l_tasks, l_names, l_captions, l_sources, link_failures = await _tasks_from_links(url_items, resolved_lang)
        tasks += l_tasks
        names += l_names
        captions += l_captions
        sources += l_sources

    if not tasks:
        if not link_failures:  # например, ?url= из одних пробелов
            return _json({"error": config._t("error.empty_body", lang=resolved_lang)}, status=400)
        # Ни картинок из тела, ни одной удавшейся ссылки: отвечаем ошибкой первой ссылки.
        first = link_failures[0]
        return _json(
            {"error": first.message, "link_errors": [f.as_dict() for f in link_failures]},
            status=first.status,
        )

    backend_was_explicit = bool(overrides["backend"])

    logging.info(
        "Получено изображений в запросе: %d (%s) | backend=%s model=%s lang=%s "
        "categories=%d(разовых)",
        len(tasks), ", ".join(names), backend, overrides["model"] or "auto",
        resolved_lang or "default", len(extra_categories or []),
    )

    try:
        results_raw = await asyncio.gather(*[
            backends._analyze_image(
                img_b64, img_mime,
                backend=backend, model=overrides["model"],
                allow_fallback=not backend_was_explicit,
                lang=resolved_lang,
                caption=cap, overlay=category_overlay,
            )
            for (img_b64, img_mime), cap in zip(tasks, captions)
        ])
    except aiohttp.ClientConnectorError:
        endpoint = providers.get(backend).endpoint
        logging.error("Не удалось подключиться к бэкенду %s (%s)", backend, endpoint)
        return _json(
            {"error": config._t("error.backend_unavailable", backend=backend, endpoint=endpoint, lang=resolved_lang)},
            status=502,
        )
    except asyncio.TimeoutError:
        logging.warning(
            "Таймаут при обращении к backend=%s (не уложились в %.0fс) — "
            "проверь num_ctx (/sampling), бэкенд может быть перегружен",
            backend, config.REQUEST_TIMEOUT.total,
        )
        return _json({"error": config._t("error.backend_timeout", lang=resolved_lang)}, status=504)
    except (ValueError, RuntimeError) as e:
        return _json({"error": str(e)}, status=400)
    except aiohttp.ClientResponseError as e:
        # Ошибка апстрима (503 у Gemini и т.п.) — ожидаемая, трейсбек не нужен.
        logging.error(
            "Бэкенд %s вернул %s %s (%s)", backend, e.status, e.message, e.request_info.real_url.host,
        )
        return _json(
            {"error": f"{backend}: {e.status} {e.message}"},
            status=502 if e.status >= 500 else 400,
        )
    except Exception as e:
        logging.exception("Ошибка при анализе изображений: %s", e)
        return _json({"error": config._t("error.analyze_failed", lang=resolved_lang)}, status=500)

    results = []
    for name, source, (report, actual_backend) in zip(names, sources, results_raw):
        if "_raw" in report:
            logging.info(
                "Анализ %r завершён (backend=%s), ответ модели не по JSON-схеме", name, actual_backend,
            )
        else:
            logging.info("Анализ %r завершён (backend=%s)", name, actual_backend)
        entry = {"file": name}
        if source:
            entry["source"] = source
        entry.update(backend=actual_backend, report=report)
        results.append(entry)

    response = {"count": len(results), "requested_backend": backend, "results": results}
    if link_failures:
        response["link_errors"] = [f.as_dict() for f in link_failures]
    return _json(response)