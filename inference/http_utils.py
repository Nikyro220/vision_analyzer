"""
http_utils.py — общие хелперы HTTP-слоя inference-сервера, которыми
пользуются и сервер/чат, и инструменты из tools/ (vision_analyzer,
embeddings, ...): JSON-ответ, внутренний сигнал _BodyError и подготовка
изображения (детект mime + необязательный апскейл + base64).

Вынесено из analyze.py, чтобы chat.py и tools/* не импортировали друг
друга ради пары общих функций.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging

from aiohttp import web
from PIL import Image

import config

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
    Общая точка для всех веток разбора тела /analyze и /chat, чтобы не повторять
    один и тот же порядок действий три раза.
    """
    real_mime = await _detect_image_mime(data)
    if real_mime is None:
        return None

    if config.image_upscaler is not None:
        data, new_mime = await config.image_upscaler.upscale_if_needed(data, source_name=source_name)
        if new_mime:
            real_mime = new_mime

    return base64.b64encode(data).decode("utf-8"), real_mime


# ---------------------------------------------------------------------------
# Ответы
# ---------------------------------------------------------------------------

class _BodyError(Exception):
    """Внутренний сигнал "верни этот JSON-ответ с ошибкой и не ходи дальше"."""

    def __init__(self, response: web.Response):
        self.response = response


def _json(data, status: int = 200) -> web.Response:
    """web.json_response с ensure_ascii=False и indent=2 по умолчанию.

    Лежит здесь (а не в server.py), потому что её используют и server.py,
    и chat.py, и инструменты из tools/ — если бы _json была в server.py,
    получился бы циклический импорт.
    """
    return web.json_response(
        data, status=status, dumps=lambda obj: json.dumps(obj, ensure_ascii=False, indent=2),
    )
