"""
chat.py — всё, что относится к POST /chat: свободный, бесхис-схемный
диалог с моделью (текст + необязательные картинки, с историей), для
UI/ботов, которым не нужен риск-анализ по строгой JSON-схеме (см.
analyze.py: POST /analyze — тот всегда разовый, без истории).

Сервер ничего не хранит — история диалога целиком приходит от клиента
на каждый запрос (поле 'history'), ровно как раньше делал /analyze.

Вынесено отдельным модулем от analyze.py/backends.py по той же причине,
по которой сам /analyze вынесен из server.py: чтобы не мешать разную
модельную логику (риск-анализ по JSON-схеме vs свободный диалог) в одни
и те же файлы. Общее (подготовка изображений, data-url утилиты, JSON-
обёртка ответа) переиспользуется через импорт, а не копируется.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any

import aiohttp
from aiohttp import web

import backends
import chat_backends
import config
from analyze import _BodyError, _json, _prepare_image
from config import locales, prompt

# ---------------------------------------------------------------------------
# Разбор тела запроса — два формата (в отличие от /analyze, тут всегда
# нужен текст сообщения, поэтому "сырое изображение в теле" не подходит):
# (message, images) либо готовый web.Response с ошибкой (_BodyError).
# ---------------------------------------------------------------------------

def _query_overrides(request: web.Request) -> dict[str, Any]:
    return {
        "backend": request.query.get("backend"),
        "model": request.query.get("model"),
        "lang": request.query.get("lang"),
    }


async def _image_from_b64(raw: str, source_name: str, lang: str | None) -> str:
    """base64/data-url строка → подготовленный data URL (детект mime +
    апскейл при необходимости — тот же _prepare_image, что и у /analyze).
    Бросает _BodyError(400), если это не изображение."""
    try:
        data = base64.b64decode(backends._strip_data_url(raw))
    except Exception:
        raise _BodyError(_json({"error": config._t("error.not_image", lang=lang)}, status=400))

    prepared = await _prepare_image(data, source_name=source_name)
    if prepared is None:
        raise _BodyError(_json({"error": config._t("error.not_image", lang=lang)}, status=400))

    img_b64, mime = prepared
    return backends._ensure_data_url(img_b64, default_mime=mime)


async def _parse_json_body(request: web.Request, overrides: dict) -> tuple[str, list[str]]:
    """Content-Type: application/json, поля:
    message (обязательно), images/image (необязательно), history,
    system, backend/model/lang (см. также query-параметры выше)."""
    try:
        body = await request.json() or {}
    except json.JSONDecodeError:
        raise _BodyError(
            _json({"error": config._t("error.invalid_json_body", lang=overrides["lang"])}, status=400)
        )

    for key in overrides:
        overrides[key] = overrides[key] or body.get(key)

    message = (overrides["message"] or "").strip()
    if not message:
        raise _BodyError(_json({"error": config._t("error.empty_message", lang=overrides["lang"])}, status=400))

    raw_images = body.get("images")
    if not raw_images:
        single = body.get("image")
        raw_images = [single] if single else []

    images = [
        await _image_from_b64(img, source_name=f"chat_{idx + 1}", lang=overrides["lang"])
        for idx, img in enumerate(raw_images)
    ]
    return message, images


async def _parse_multipart_body(request: web.Request, overrides: dict) -> tuple[str, list[str]]:
    """Content-Type: multipart/form-data — текстовые поля message/
    system/history/backend/model/lang плюс одно или несколько полей
    'images'/'image' с файлами картинок."""
    reader = await request.multipart()
    images: list[str] = []
    text_fields = {"message": False, "system": False, "history": True, "backend": True, "model": True, "lang": True}

    async for part in reader:
        if part.name in text_fields:
            raw = (await part.read(decode=True)).decode("utf-8")
            overrides[part.name] = raw.strip() if text_fields[part.name] else raw
            continue
        if part.name not in ("images", "image"):
            continue

        data = await part.read(decode=True)
        prepared = await _prepare_image(data, source_name=part.filename or f"chat_image_{len(images) + 1}")
        if prepared is None:
            logging.warning("chat: пропускаю не-изображение: %s", part.filename)
            continue
        img_b64, mime = prepared
        images.append(backends._ensure_data_url(img_b64, default_mime=mime))

    message = (overrides.get("message") or "").strip()
    if not message:
        raise _BodyError(_json({"error": config._t("error.empty_message", lang=overrides["lang"])}, status=400))

    return message, images


_BODY_PARSERS = (
    ("application/json", _parse_json_body),
    ("multipart/", _parse_multipart_body),
)


# ---------------------------------------------------------------------------
# Хендлер
# ---------------------------------------------------------------------------

async def handle_chat(request: web.Request) -> web.Response:
    overrides = _query_overrides(request)
    overrides.setdefault("message", None)
    overrides.setdefault("system", None)
    overrides.setdefault("history", None)
    content_type = request.content_type

    parser = next((fn for prefix, fn in _BODY_PARSERS if content_type.startswith(prefix)), None)
    if parser is None:
        return _json({"error": config._t("error.chat_unsupported_content_type", lang=overrides["lang"])}, status=400)

    try:
        message, images = await parser(request, overrides)
    except _BodyError as e:
        return e.response

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
        resolved_history = chat_backends._parse_history_json(overrides["history"])
    except ValueError:
        return _json({"error": config._t("error.invalid_history", lang=resolved_lang)}, status=400)

    backend_was_explicit = bool(overrides["backend"])

    # Дефолтная "личность" ассистента по инструменту (см. prompt.py:
    # get_chat_system_prompt) — подставляется ВСЕГДА, даже если клиент
    # передал своё поле 'system': оно не заменяет базовый промпт, а
    # добавляется к нему как доп. инструкция на этот вызов. Если
    # prompt.py не найден на сервере (см. config.py) — ведём себя как
    # раньше и просто передаём system как есть (может быть None).
    system_prompt = (
        prompt.get_chat_system_prompt(resolved_lang or config._current_lang(), overrides["system"])
        if prompt is not None
        else overrides["system"]
    )

    logging.info(
        "chat: message=%d симв. картинок=%d | backend=%s model=%s lang=%s history=%d",
        len(message), len(images), backend, overrides["model"] or "auto",
        resolved_lang or "default", len(resolved_history),
    )

    try:
        reply, actual_backend, actual_model = await chat_backends.chat(
            message, images,
            backend=backend, model=overrides["model"], system=system_prompt,
            history=resolved_history, allow_fallback=not backend_was_explicit,
        )
    except aiohttp.ClientConnectorError as e:
        # backend — исходно запрошенный бэкенд; если сработал фолбэк
        # (см. chat_backends.chat) и упал ВТОРОЙ бэкенд, реально
        # неудачным был именно он, а не backend — chat_backends.chat
        # помечает это на самом исключении (chat_backend), чтобы здесь
        # не соврать про то, какой бэкенд/эндпоинт на самом деле недоступен.
        failed_backend = getattr(e, "chat_backend", backend)
        endpoint = config.VLLM_URL if failed_backend == "vllm" else config.OLLAMA_HOST
        if failed_backend != backend:
            logging.error(
                "chat: не удалось подключиться ни к одному бэкенду — %s (исходный) и %s (фолбэк) недоступны",
                backend, failed_backend,
            )
        else:
            logging.error("chat: не удалось подключиться к бэкенду %s (%s)", failed_backend, endpoint)
        return _json(
            {"error": config._t("error.backend_unavailable", backend=failed_backend, endpoint=endpoint, lang=resolved_lang)},
            status=502,
        )
    except asyncio.TimeoutError:
        logging.warning(
            "chat: таймаут при обращении к backend=%s (не уложились в %.0fс) — "
            "проверь num_ctx (/sampling) и размер history, бэкенд может быть перегружен",
            backend, config.REQUEST_TIMEOUT.total,
        )
        return _json({"error": config._t("error.backend_timeout", lang=resolved_lang)}, status=504)
    except (ValueError, RuntimeError) as e:
        return _json({"error": str(e)}, status=400)
    except Exception as e:
        logging.exception("chat: ошибка при обращении к модели: %s", e)
        return _json({"error": config._t("error.chat_failed", lang=resolved_lang)}, status=500)

    logging.info("chat: ответ получен (backend=%s model=%s)", actual_backend, actual_model)
    return _json({"reply": reply, "backend": actual_backend, "model": actual_model})