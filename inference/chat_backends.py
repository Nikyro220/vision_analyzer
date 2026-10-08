"""
chat_backends.py — модельный слой POST /chat: свободный диалог с моделью
(текст + необязательные картинки, с историей), БЕЗ строгой JSON-схемы
риск-отчёта — в отличие от backends.py (двухпроходный /analyze, всегда
разовый, без истории).

Отсюда:
  - разбор/валидация 'history' (_parse_history_json) — формат бэкенд-агностичный
  - обёртка chat() с цепочкой фолбэка между бэкендами (см. fallback.py)

Всё, что зависит от конкретного бэкенда — конвертация истории в его формат,
защитная обрезка под контекст vLLM, сам HTTP-запрос без format=json/
response_format — живёт в providers/ (Provider.chat).

Как и backends.py, сам ничего не хранит: история приходит целиком от
клиента на каждый запрос (см. chat.py: handle_chat) — сервер её нигде
между вызовами не сохраняет.
"""

from __future__ import annotations

import json
import logging

import config
import fallback
import providers


# ---------------------------------------------------------------------------
# История: разбор и валидация входа
# ---------------------------------------------------------------------------

def _parse_history_json(raw) -> list:
    """Парсит и валидирует 'history' — список прошлых сообщений диалога.

    Сервер сам историю нигде не хранит — она целиком приходит от клиента
    (бота/UI) в каждом запросе. Формат бэкенд-агностичный:

        [
          {"role": "user", "content": "...", "images": ["data:image/png;base64,..."]},
          {"role": "assistant", "content": "..."}
        ]

    'content' и 'images' необязательны, но должны быть строкой/списком,
    если присутствуют. 'images' — data URL (или просто base64 — тоже
    примется, см. providers.strip_data_url/ensure_data_url).

    raw может быть уже списком (если пришло в JSON-теле запроса) либо
    JSON-строкой (если пришло через query-параметр или multipart-поле).
    Пустое/отсутствующее значение — просто "истории нет".
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        raw = json.loads(raw)  # может бросить json.JSONDecodeError (это ValueError)

    if not isinstance(raw, list):
        raise ValueError("history must be a list")

    for turn in raw:
        if not isinstance(turn, dict) or turn.get("role") not in ("user", "assistant"):
            raise ValueError("each history item needs role: 'user' or 'assistant'")
        if "content" in turn and not isinstance(turn["content"], str):
            raise ValueError("history 'content' must be a string")
        if "images" in turn and not isinstance(turn["images"], list):
            raise ValueError("history 'images' must be a list")

    return raw


async def chat(
    message: str,
    images: list[str] | None = None,
    backend: str = config.BACKEND,
    model: str | None = None,
    system: str | None = None,
    history: list | None = None,
    allow_fallback: bool = True,
) -> tuple[str, str, str]:
    """Точка входа для chat.py: одна реплика в свободном диалоге.

    Возвращает (reply_text, backend_used, model_used) — reply_text
    отдаётся клиенту как есть, без попытки распарсить его как JSON
    (в отличие от backends._analyze_image).
    """
    images = images or []
    history = history or []

    async def call(name: str, is_fallback: bool) -> tuple[str, str, str]:
        provider = providers.get(name)  # ValueError для неизвестного бэкенда
        # На запасном бэкенде модель исходного провайдера бессмысленна — автоопределение.
        resolved_model = (None if is_fallback else model) or await provider.discover_model()
        content = await provider.chat(resolved_model, system, history, message, images)
        return content, name, resolved_model

    return await fallback.run(backend, call, allow_fallback)
