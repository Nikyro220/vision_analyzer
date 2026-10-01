"""Миниатюры для списков истории: маленькие JPEG, создаются при первом запросе и кэшируются на диске.

Кэш лежит в UPLOAD_FOLDER/thumbs/... и повторяет структуру uploads/..., поэтому работает и для
уже загруженных раньше файлов. Удаляются миниатюры вместе с оригиналом (см. history.py).
"""

from __future__ import annotations

import logging
import os
import uuid
from pathlib import Path

from PIL import Image, ImageOps

from .config import Config, conf

log = logging.getLogger("vision_app.thumbs")

# Имя каталога — константа (менять его на ходу нельзя: в БД хранятся готовые пути). Размер и качество
# миниатюр читаем через conf() в момент создания — их можно менять в /panel/settings/.
THUMBS_DIR = Config.THUMBS_DIR


def thumb_rel(image_rel: str) -> str:
    """uploads/2026/09/28/abc.png -> thumbs/uploads/2026/09/28/abc.jpg"""
    stem = image_rel.rsplit(".", 1)[0] if "." in Path(image_rel).name else image_rel
    return f"{THUMBS_DIR}/{stem}.jpg"


def ensure_thumb(root: Path, image_rel: str) -> Path | None:
    """Путь к миниатюре (создаёт при необходимости). None — если исходник недоступен или не читается."""
    root = root.resolve()
    src = (root / image_rel).resolve()
    dst = (root / thumb_rel(image_rel)).resolve()
    thumbs_root = root / THUMBS_DIR
    if root not in src.parents or thumbs_root not in dst.parents:  # защита от «../»
        return None
    if dst.is_file():
        return dst
    if not src.is_file():
        return None

    size = int(conf("THUMB_SIZE"))
    tmp = dst.with_name(f"{dst.stem}.{uuid.uuid4().hex}.tmp")
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as im:
            im.draft("RGB", (size * 2, size * 2))  # для JPEG декодируем сразу уменьшенным
            im = ImageOps.exif_transpose(im)
            if im.mode in ("P", "LA", "PA") or "transparency" in im.info:
                im = im.convert("RGBA")
            if im.mode == "RGBA":  # прозрачность -> белая подложка, иначе в JPEG будет чёрный фон
                bg = Image.new("RGB", im.size, (255, 255, 255))
                bg.paste(im, mask=im.getchannel("A"))
                im = bg
            im = ImageOps.fit(im.convert("RGB"), (size, size), Image.Resampling.LANCZOS)
            im.save(tmp, "JPEG", quality=int(conf("THUMB_QUALITY")), optimize=True)
        os.replace(tmp, dst)  # атомарно: параллельный запрос не увидит недописанный файл
        return dst
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        log.warning("Не удалось сделать миниатюру для %s: %s", image_rel, exc)
        return None
    finally:
        tmp.unlink(missing_ok=True)
