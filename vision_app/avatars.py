"""Аватары пользователей: проверка присланной картинки, перекодирование и хранение на диске.

Редактирование (сдвиг, масштаб, поворот, цвет...) выполняется в браузере (static/js/avatar-editor.js),
а сервер получает уже готовую картинку. Ей всё равно не доверяем: открываем через Pillow, проверяем
формат и размеры и ПЕРЕКОДИРУЕМ в квадратный WebP — метаданные и любой посторонний «хвост» в файле
пропадают, на диск попадает только то, что мы сами сохранили.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path

from flask import current_app
from PIL import Image, ImageOps, UnidentifiedImageError

from .config import Config

log = logging.getLogger("vision_app.avatars")

ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}


class AvatarError(ValueError):
    """Ошибка для пользователя: текст можно показывать как есть."""


def avatar_file(user_id: int) -> Path:
    return Path(current_app.config["UPLOAD_FOLDER"]) / Config.AVATARS_DIR / f"{int(user_id)}.webp"


def process_avatar(data: bytes) -> bytes:
    """Картинка (PNG/JPEG/WebP) -> квадратный WebP AVATAR_SIZE x AVATAR_SIZE."""
    limit = current_app.config.get("AVATAR_MAX_UPLOAD_BYTES", Config.AVATAR_MAX_UPLOAD_BYTES)
    if not data:
        raise AvatarError("Файл пустой.")
    if len(data) > limit:
        raise AvatarError(f"Файл слишком большой (максимум {limit // 1024} КБ).")

    max_side = current_app.config.get("AVATAR_MAX_SIDE", Config.AVATAR_MAX_SIDE)
    size = current_app.config.get("AVATAR_SIZE", Config.AVATAR_SIZE)
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.format not in ALLOWED_FORMATS:
                raise AvatarError("Поддерживаются только PNG, JPEG и WebP.")
            if max(img.size) > max_side:  # размер известен до распаковки пикселей
                raise AvatarError(f"Изображение слишком большое (максимум {max_side}x{max_side}).")
            img.load()
            img = ImageOps.exif_transpose(img).convert("RGBA")
    except AvatarError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError):
        raise AvatarError("Не удалось прочитать изображение.") from None

    # Редактор уже отдаёт квадрат; если прислали другое — вырезаем центр, без искажений.
    img = ImageOps.fit(img, (size, size), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    img.save(out, "WEBP", quality=90, method=4)
    return out.getvalue()


def save_avatar(user_id: int, data: bytes) -> None:
    """Атомарно записывает уже обработанный аватар."""
    target = avatar_file(user_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, target)


def remove_avatar(user_id: int) -> None:
    try:
        avatar_file(user_id).unlink(missing_ok=True)
    except OSError as exc:
        log.warning("Не удалось удалить аватар пользователя %s: %s", user_id, exc)
