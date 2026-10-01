"""Изображения, прикреплённые к сообщениям чата.

Отдельно от загрузок анализа (папка uploads/, таблица analysis_results): вложения чата
лежат в UPLOAD_FOLDER/chat_uploads/ГГГГ/ММ/ДД/<uuid>.<ext>, а в БД — только в
ChatMessage.images (список словарей {path, name, mime}). Они не попадают ни в очередь,
ни в историю анализов, ни в поиск по смыслу; отдаются владельцу чата через
blueprints/chat.py: attachment().

Как вложения попадают к модели:
  - по умолчанию картинки передаются ей напрямую вместе с сообщением (data-URL в поле
    `images` POST /chat сервера анализа) — to_data_url(); так модель отвечает на вопросы
    об изображении сама, без очереди;
  - в очередь анализа изображение ставится только по явной просьбе пользователя —
    инструментом analyze_image (chat_tools/images.py), который читает файл с диска по
    номеру вложения (chat_jobs.enqueue).
"""

from __future__ import annotations

import base64
import io
import logging
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import current_app
from PIL import Image, ImageOps
from sqlalchemy import select

from .config import Config, conf
from .extensions import db
from .forms import IMAGE_FORMATS
from .history import _prune_empty_dirs
from .models import ChatMessage, ChatSession
from .thumbs import THUMBS_DIR, thumb_rel

log = logging.getLogger("vision_app.chat_images")

# Числовые лимиты (число вложений, размер, сторона для модели, качество JPEG) — в config.py: CHAT_*.
CHAT_UPLOADS_DIR = Config.CHAT_UPLOADS_DIR
_PASSTHROUGH_FORMATS = {"image/jpeg", "image/png", "image/webp"}
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]+")


class AttachmentError(ValueError):
    """Вложение не принято; текст ошибки можно показывать пользователю."""


def _root() -> Path:
    return Path(current_app.config["UPLOAD_FOLDER"]).resolve()


def clean_name(name: str, limit: int | None = None) -> str:
    """Имя файла для отображения и для подстановки в промпт: без управляющих символов
    и кавычек-ёлочек (в промпте имя заключено в «»), укороченное."""
    limit = limit or conf("CHAT_ATTACHMENT_NAME_MAX")
    text = _CONTROL_RE.sub(" ", str(name or "")).replace("«", "").replace("»", "").strip()
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text or "image"


def validate_and_save(files) -> list[dict]:
    """Проверяет загруженные файлы (содержимое через Pillow, а не расширение) и пишет их на диск.

    Возвращает список {path, name, mime} — его нужно положить в ChatMessage.images.
    Любая ошибка — AttachmentError, при этом уже записанные файлы удаляются: вложения
    принимаются либо все, либо ни одного.
    """
    files = [f for f in files if getattr(f, "filename", "")]
    max_images = conf("CHAT_MAX_IMAGES_PER_MESSAGE")
    max_bytes = conf("CHAT_MAX_IMAGE_BYTES")
    if len(files) > max_images:
        raise AttachmentError(f"К одному сообщению можно прикрепить не больше {max_images} изображений.")

    root = _root()
    now = datetime.now(timezone.utc)
    saved: list[dict] = []
    try:
        for upload in files:
            name = clean_name(upload.filename)
            data = upload.read(max_bytes + 1)
            if not data:
                raise AttachmentError(f"Файл «{name}» пуст.")
            if len(data) > max_bytes:
                raise AttachmentError(f"Файл «{name}» слишком большой (максимум {max_bytes // (1024 * 1024)} МБ).")
            try:
                with Image.open(io.BytesIO(data)) as img:
                    fmt = img.format
                    img.verify()
            except Exception:  # noqa: BLE001 — Pillow бросает разные исключения на битых файлах
                raise AttachmentError(f"Файл «{name}» не является изображением или повреждён.") from None
            if fmt not in IMAGE_FORMATS:
                raise AttachmentError(f"Формат файла «{name}» не поддерживается.")

            ext, mime = IMAGE_FORMATS[fmt]
            rel = f"{CHAT_UPLOADS_DIR}/{now:%Y/%m/%d}/{uuid.uuid4().hex}{ext}"
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            saved.append({"path": rel, "name": name, "mime": mime})
    except BaseException:
        remove_files([item["path"] for item in saved])
        raise
    return saved


def resolve_path(rel: str) -> Path | None:
    """Абсолютный путь вложения или None, если путь выходит за chat_uploads/ или файла нет."""
    if not rel or not isinstance(rel, str):
        return None
    chat_root = _root() / CHAT_UPLOADS_DIR
    target = (_root() / rel).resolve()
    if chat_root not in target.parents or not target.is_file():
        return None
    return target


def read_bytes(rel: str) -> bytes | None:
    path = resolve_path(rel)
    if path is None:
        return None
    try:
        return path.read_bytes()
    except OSError as exc:
        log.warning("Не удалось прочитать вложение %s: %s", rel, exc)
        return None


def remove_files(paths: list[str]) -> None:
    """Удаляет вложения, их миниатюры и опустевшие папки с датой. Не выходит за chat_uploads/."""
    root = _root()
    chat_root = root / CHAT_UPLOADS_DIR
    thumbs_chat_root = root / THUMBS_DIR / CHAT_UPLOADS_DIR

    for rel in paths:
        if not rel or not isinstance(rel, str):
            continue
        target = (root / rel).resolve()
        if chat_root not in target.parents:
            log.warning("Пропускаю путь вне chat_uploads: %s", rel)
            continue
        try:
            target.unlink(missing_ok=True)
        except OSError as exc:
            log.warning("Не удалось удалить вложение %s: %s", target, exc)
            continue
        _prune_empty_dirs(target.parent, chat_root)

        thumb = (root / thumb_rel(rel)).resolve()
        if thumbs_chat_root in thumb.parents:
            try:
                thumb.unlink(missing_ok=True)
            except OSError as exc:
                log.warning("Не удалось удалить миниатюру %s: %s", thumb, exc)
            else:
                _prune_empty_dirs(thumb.parent, thumbs_chat_root)


def to_data_url(rel: str, mime: str = "") -> str | None:
    """Вложение -> data-URL для поля `images` POST /chat. None — файла нет или он не читается.

    Небольшие JPEG/PNG/WEBP уходят как есть; остальное (большие файлы, GIF/BMP/TIFF)
    перекодируется в JPEG с длинной стороной не более CHAT_MODEL_MAX_SIDE.
    """
    data = read_bytes(rel)
    if data is None:
        return None
    max_side = conf("CHAT_MODEL_MAX_SIDE")
    try:
        with Image.open(io.BytesIO(data)) as im:
            width, height = im.size
            needs_convert = (
                mime not in _PASSTHROUGH_FORMATS
                or len(data) > conf("CHAT_MODEL_PASSTHROUGH_BYTES")
                or max(width, height) > max_side
            )
            if needs_convert:
                im = ImageOps.exif_transpose(im)
                if im.mode in ("P", "LA", "PA") or "transparency" in im.info:
                    im = im.convert("RGBA")
                if im.mode == "RGBA":  # прозрачность -> белая подложка, иначе в JPEG будет чёрный фон
                    bg = Image.new("RGB", im.size, (255, 255, 255))
                    bg.paste(im, mask=im.getchannel("A"))
                    im = bg
                im = im.convert("RGB")
                im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                out = io.BytesIO()
                im.save(out, "JPEG", quality=conf("CHAT_MODEL_JPEG_QUALITY"))
                data, mime = out.getvalue(), "image/jpeg"
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        log.warning("Не удалось подготовить вложение %s для модели: %s", rel, exc)
        return None
    return f"data:{mime or 'image/jpeg'};base64,{base64.b64encode(data).decode('ascii')}"


def paths_of(messages) -> list[str]:
    """Пути всех вложений из набора ChatMessage."""
    out: list[str] = []
    for message in messages:
        for item in message.images or []:
            if isinstance(item, dict) and item.get("path"):
                out.append(item["path"])
    return out


def user_paths(user_id: int) -> list[str]:
    """Пути всех вложений всех чатов пользователя (для удаления аккаунта)."""
    rows = db.session.scalars(
        select(ChatMessage)
        .join(ChatSession, ChatSession.id == ChatMessage.session_id)
        .where(ChatSession.user_id == user_id)
    ).all()
    return paths_of(rows)
