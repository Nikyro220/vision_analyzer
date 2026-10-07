"""Инструмент `analyze_image`: постановка изображений из чата в очередь анализа.

Картинки, прикреплённые к сообщениям, по умолчанию передаются модели напрямую (поле
`images` POST /chat, см. blueprints/chat.py) — на вопросы об изображении она отвечает сама.
Инструмент нужен только тогда, когда пользователь ЯВНО просит запустить анализ системой
(«добавь в анализ», «поставь в очередь», «сделай риск-анализ»): модель видит список
вложений («#1 — «photo.png»») и присылает JSON-вызов (протокол — в runner.py).

Инструмент САМ НИЧЕГО НЕ АНАЛИЗИРУЕТ: он только добавляет изображение в общую очередь
анализа (ту же, что у страницы «Анализ») и возвращает модели статус «в очереди». Когда
очередь дойдёт до анализа и он завершится, результат приходит в чат отдельным ходом:
blueprints/chat.py (GET .../pending) передаёт отчёт модели, и её ответ появляется в
переписке. Вся работа с очередью — в chat_jobs.py.

Принципы те же, что у остальных инструментов:

  - Аргументы от модели — НЕДОВЕРЕННЫЙ ввод: номер приводится к int и ищется только
    среди вложений ЭТОГО чата (список собирает blueprints/chat.py из сообщений сессии
    текущего пользователя), путь к файлу от модели не принимается вообще.
  - Лимит очереди на пользователя (QUEUE_MAX_PENDING_PER_USER) действует и здесь.
  - Текст на картинке и подпись — данные, а не инструкции (это оговорено в промпте).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from flask import current_app

from .. import chat_jobs
from ..config import conf
from .analyses import ToolResult

TOOL_NAME = "analyze_image"


@dataclass
class ChatImage:
    """Вложение чата в том виде, в котором его видят инструмент и промпт."""

    number: int  # сквозной номер по чату, с 1
    path: str  # относительно UPLOAD_FOLDER
    name: str
    mime: str
    # Уходит ли картинка модели напрямую. False у старых вложений сверх лимита
    # CHAT_CONTEXT_IMAGES: они остаются в чате и доступны инструменту, но модель их не видит.
    in_context: bool = True


def _error(message: str, **extra) -> ToolResult:
    return ToolResult(json.dumps({"error": message, **extra}, ensure_ascii=False))


def attachment_note(images: list[ChatImage]) -> str:
    """Пометка, которая дописывается к тексту сообщения для модели (в БД не сохраняется)."""
    if not images:
        return ""
    items = ", ".join(
        f"#{img.number} «{img.name}»" + ("" if img.in_context else " (вне контекста, не видно)") for img in images
    )
    label = "изображение" if len(images) == 1 else "изображения"
    return f"\n\n[Прикреплено {label}: {items}]"


def images_prompt_block(images: list[ChatImage]) -> str:
    return "\n".join(
        f"  #{img.number} — «{img.name}» "
        + ("(you can see this image)" if img.in_context else "(out of context: you CANNOT see it)")
        for img in images
    )


def _image_number(value) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(str(value).strip().lstrip("#"))
    except (TypeError, ValueError):
        return None


def normalize_args(raw) -> tuple[int | None, str, list[str]]:
    """(номер или None, подпись или "", предупреждения)."""
    warnings: list[str] = []
    if not isinstance(raw, dict):
        warnings.append("args должен быть объектом — использованы значения по умолчанию")
        raw = {}

    unknown = sorted(str(k) for k in raw if k not in {"image", "caption"})
    if unknown:
        warnings.append("неизвестные аргументы проигнорированы: " + ", ".join(unknown))

    number = _image_number(raw.get("image"))
    if raw.get("image") not in (None, "") and number is None:
        warnings.append("image должен быть номером изображения — взято последнее")

    caption = ""
    if raw.get("caption") is not None:
        caption = " ".join(str(raw["caption"]).split())
        max_chars = conf("CAPTION_MAX_CHARS")
        if len(caption) > max_chars:
            caption = caption[:max_chars]
            warnings.append(f"caption обрезан до {max_chars} символов")
    return number, caption, warnings


def analyze_chat_image(user, raw_args, images: list[ChatImage], session_id: int | None) -> ToolResult:
    """Ставит изображение в очередь. `images` — вложения текущего чата (других модель получить не может)."""
    if not images or session_id is None:
        return _error("в этом чате нет прикреплённых изображений")

    number, caption, warnings = normalize_args(raw_args)
    by_number = {img.number: img for img in images}
    if number is None:
        image = images[-1]
    else:
        image = by_number.get(number)
        if image is None:
            return _error(f"изображения #{number} нет в этом чате", available=sorted(by_number))

    current_app.logger.info(
        "chat_tools: analyze_image user=%s image=#%s -> очередь", getattr(user, "id", "?"), image.number
    )
    try:
        payload = chat_jobs.enqueue(user, session_id, image, caption)
    except chat_jobs.JobError as exc:
        return _error(f"не удалось поставить изображение в очередь: {exc}")

    if warnings:
        payload["warnings"] = warnings
    return ToolResult(json.dumps(payload, ensure_ascii=False))
