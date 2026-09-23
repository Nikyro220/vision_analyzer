"""
analyze.py — всё, что относится к POST /analyze: подготовка изображений
(детект mime + апскейл), разбор тела запроса в трёх поддерживаемых
форматах (сырые байты, JSON, multipart) и сам хендлер, который сводит
их к общему пути — вызову backends._analyze_image и сборке ответа.

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
import config
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
# Разбор тела запроса — три формата, один и тот же результат:
# (tasks, names, captions, overrides) либо готовый web.Response с ошибкой.
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
        "history": request.query.get("history"),
        "caption": request.query.get("caption"),
    }


async def _parse_raw_image_body(request: web.Request, overrides: dict) -> tuple[list, list, list]:
    """Content-Type: image/* — сырые байты картинки прямо в теле."""
    data = await request.read()
    if not data:
        raise _BodyError(_json({"error": config._t("error.empty_body", lang=overrides["lang"])}, status=400))

    prepared = await _prepare_image(data, source_name="body")
    if prepared is None:
        raise _BodyError(_json({"error": config._t("error.not_image", lang=overrides["lang"])}, status=400))

    return [prepared], ["body"], [overrides["caption"]]


async def _parse_json_body(request: web.Request, overrides: dict) -> tuple[list, list, list]:
    """Content-Type: application/json — удобно для UI/ботов, поддерживает
    историю диалога и batch с caption на каждую картинку отдельно.

    'images' — список, каждый элемент либо строка с картинкой (caption
    для неё общий, из overrides['caption']), либо объект
    {"image": "...", "caption": "..."} — свой caption на эту картинку.
    """
    body = await request.json() or {}
    for key in overrides:
        overrides[key] = overrides[key] or body.get(key)

    raw_images = body.get("images")
    if not raw_images:
        single = body.get("image")
        raw_images = [single] if single else []

    if not raw_images:
        raise _BodyError(_json({"error": config._t("error.empty_body", lang=overrides["lang"])}, status=400))

    tasks, names, captions = [], [], []
    for idx, item in enumerate(raw_images):
        if isinstance(item, dict):
            img, item_caption = item.get("image"), item.get("caption") or overrides["caption"]
        else:
            img, item_caption = item, overrides["caption"]

        try:
            data = base64.b64decode(backends._strip_data_url(img))
        except Exception:
            raise _BodyError(_json({"error": config._t("error.not_image", lang=overrides["lang"])}, status=400))

        source_name = f"json_{idx + 1}"
        prepared = await _prepare_image(data, source_name=source_name)
        if prepared is None:
            raise _BodyError(_json({"error": config._t("error.not_image", lang=overrides["lang"])}, status=400))

        tasks.append(prepared)
        names.append(source_name)
        captions.append(item_caption)

    return tasks, names, captions


# Текстовые поля-переопределения в multipart-запросе → куда класть
# значение в overrides. True — значение стрипается (лишние пробелы по
# краям не нужны), False — не стрипается (в тексте поста пробелы могут
# быть значимыми).
_MULTIPART_TEXT_FIELDS = {
    "backend": True, "model": True, "lang": True, "history": True, "caption": False,
}


async def _parse_multipart_body(request: web.Request, overrides: dict) -> tuple[list, list, list]:
    """Content-Type: multipart/form-data — старый путь, одно или несколько
    полей 'images'/'image' плюс текстовые поля-переопределения."""
    reader = await request.multipart()
    tasks, names = [], []

    async for part in reader:
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

    if not tasks:
        raise _BodyError(_json(
            {"error": config._t("error.no_images_multipart", lang=overrides["lang"])}, status=400,
        ))

    return tasks, names, [overrides["caption"]] * len(tasks)


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
# Хендлер
# ---------------------------------------------------------------------------

async def handle_analyze(request: web.Request) -> web.Response:
    overrides = _query_overrides(request)
    content_type = request.content_type

    parser = next((fn for prefix, fn in _BODY_PARSERS if content_type.startswith(prefix)), None)
    if parser is None:
        return _json({"error": config._t("error.unsupported_content_type", lang=overrides["lang"])}, status=400)

    try:
        tasks, names, captions = await parser(request, overrides)
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
    if backend not in ("vllm", "ollama"):
        return _json(
            {"error": config._t("error.unknown_backend", backend=backend, lang=resolved_lang)}, status=400,
        )

    try:
        resolved_history = backends._parse_history_json(overrides["history"])
    except ValueError:
        return _json({"error": config._t("error.invalid_history", lang=resolved_lang)}, status=400)

    backend_was_explicit = bool(overrides["backend"])

    logging.info(
        "Получено изображений в запросе: %d (%s) | backend=%s model=%s lang=%s history=%d",
        len(tasks), ", ".join(names), backend, overrides["model"] or "auto",
        resolved_lang or "default", len(resolved_history),
    )

    try:
        results_raw = await asyncio.gather(*[
            backends._analyze_image(
                img_b64, img_mime,
                backend=backend, model=overrides["model"],
                allow_fallback=not backend_was_explicit,
                lang=resolved_lang, history=resolved_history,
                caption=cap,
            )
            for (img_b64, img_mime), cap in zip(tasks, captions)
        ])
    except aiohttp.ClientConnectorError:
        endpoint = config.VLLM_URL if backend == "vllm" else config.OLLAMA_HOST
        logging.error("Не удалось подключиться к бэкенду %s (%s)", backend, endpoint)
        return _json(
            {"error": config._t("error.backend_unavailable", backend=backend, endpoint=endpoint, lang=resolved_lang)},
            status=502,
        )
    except asyncio.TimeoutError:
        logging.warning(
            "Таймаут при обращении к backend=%s (не уложились в %.0fс) — "
            "проверь num_ctx (/sampling) и размер history, бэкенд может быть перегружен",
            backend, config.REQUEST_TIMEOUT.total,
        )
        return _json({"error": config._t("error.backend_timeout", lang=resolved_lang)}, status=504)
    except (ValueError, RuntimeError) as e:
        return _json({"error": str(e)}, status=400)
    except Exception as e:
        logging.exception("Ошибка при анализе изображений: %s", e)
        return _json({"error": config._t("error.analyze_failed", lang=resolved_lang)}, status=500)

    results = []
    for name, (report, actual_backend) in zip(names, results_raw):
        if "_raw" in report:
            logging.info(
                "Анализ %r завершён (backend=%s), ответ модели не по JSON-схеме", name, actual_backend,
            )
        else:
            logging.info("Анализ %r завершён (backend=%s)", name, actual_backend)
        results.append({"file": name, "backend": actual_backend, "report": report})

    return _json({"count": len(results), "requested_backend": backend, "results": results})
