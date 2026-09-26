"""
chat_backends.py — модельный слой POST /chat: свободный диалог с моделью
(текст + необязательные картинки, с историей), БЕЗ строгой JSON-схемы
риск-отчёта — в отличие от backends.py (двухпроходный /analyze, всегда
разовый, без истории).

Отсюда:
  - разбор/валидация 'history' (_parse_history_json)
  - конвертация истории в формат конкретного бэкенда
    (_history_to_ollama_messages/_history_to_vllm_messages)
  - защитная обрезка истории под реальный контекст vLLM
    (_truncate_history_for_vllm), как раньше делал backends.py для /analyze
  - низкоуровневая отправка одного chat-запроса без format=json/
    response_format (_chat_ollama/_chat_vllm) и обёртка с фолбэком между
    бэкендами (chat())

Автоопределение модели и реальный max_model_len vLLM не дублируются —
берутся из backends.py (backends._discover_model/_get_vllm_context_window),
как и общие утилиты data-url (backends._strip_data_url/_ensure_data_url).

Как и backends.py, сам ничего не хранит: история приходит целиком от
клиента на каждый запрос (см. chat.py: handle_chat) — сервер её нигде
между вызовами не сохраняет.
"""

from __future__ import annotations

import json
import logging
import uuid

import aiohttp

import backends
import config


# ---------------------------------------------------------------------------
# История: разбор входа + конвертация в формат бэкенда
# ---------------------------------------------------------------------------

def _history_to_ollama_messages(history: list) -> list[dict]:
    """История прошлых сообщений диалога → формат сообщений Ollama."""
    messages = []
    for turn in history:
        entry = {"role": turn["role"], "content": turn.get("content", "")}
        images = turn.get("images")
        if images:
            entry["images"] = [backends._strip_data_url(img) for img in images]
        messages.append(entry)
    return messages


def _history_to_vllm_messages(history: list) -> list[dict]:
    """История прошлых сообщений диалога → формат сообщений vLLM (OpenAI-style)."""
    messages = []
    for turn in history:
        blocks = []
        text = turn.get("content")
        if text:
            blocks.append({"type": "text", "text": text})
        for img in turn.get("images") or []:
            blocks.append({"type": "image_url", "image_url": {"url": backends._ensure_data_url(img)}})
        messages.append({"role": turn["role"], "content": blocks or ""})
    return messages


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
    примется, см. backends._strip_data_url/_ensure_data_url).

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


# Грубая оценка размера токенов для истории, отправляемой в vLLM — точного
# токенайзера конкретной модели у нас тут нет, поэтому это защитный запас,
# а не честный расчёт. Используется только для решения "обрезать ли
# историю", когда реальный max_model_len удалось узнать через
# backends._get_vllm_context_window; сама vLLM всё равно провалидирует
# запрос и кинет ошибку, если промпт всё же не влез.
_VLLM_EST_CHARS_PER_TOKEN = 4
_VLLM_EST_TOKENS_PER_IMAGE = 1500
_VLLM_CONTEXT_SAFETY_MARGIN = 0.9  # оставляем запас под system/ответ модели


def _estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // _VLLM_EST_CHARS_PER_TOKEN)


def _truncate_history_for_vllm(
    history: list, system_prompt: str, current_text: str, current_image_count: int, max_model_len: int,
) -> list:
    """Отбрасывает старые сообщения истории, если оценочно не влезаем
    в контекст vLLM. Идёт с конца истории (свежие сообщения важнее),
    оставляет максимум, что влезает в safety-margin от max_model_len."""
    budget = int(max_model_len * _VLLM_CONTEXT_SAFETY_MARGIN)
    fixed_tokens = (
        _estimate_tokens(system_prompt)
        + _estimate_tokens(current_text)
        + _VLLM_EST_TOKENS_PER_IMAGE * current_image_count  # картинки текущего сообщения
    )

    def turn_tokens(turn: dict) -> int:
        return _estimate_tokens(turn.get("content")) + _VLLM_EST_TOKENS_PER_IMAGE * len(turn.get("images") or [])

    kept = []
    total = fixed_tokens
    for turn in reversed(history):
        t = turn_tokens(turn)
        if total + t > budget:
            break
        kept.insert(0, turn)
        total += t

    dropped = len(history) - len(kept)
    if dropped:
        logging.warning(
            "chat/vLLM: history (%d сообщений) оценочно не влезает в контекст "
            "max_model_len=%d — отброшено %d старых сообщений, оставлено %d "
            "(оценочно ~%d/%d токенов, safety_margin=%.0f%%)",
            len(history), max_model_len, dropped, len(kept),
            total, max_model_len, _VLLM_CONTEXT_SAFETY_MARGIN * 100,
        )

    return kept


# ---------------------------------------------------------------------------
# Низкоуровневая отправка одного chat-запроса — БЕЗ format=json/
# response_format: /chat не парсит ответ модели, просто отдаёт как есть.
# ---------------------------------------------------------------------------

async def _chat_ollama(
    model: str, system: str | None, history: list, message: str, images: list[str],
) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(_history_to_ollama_messages(history))
    user_entry = {"role": "user", "content": message}
    if images:
        user_entry["images"] = [backends._strip_data_url(img) for img in images]
    messages.append(user_entry)

    options = {
        "temperature": config.SAMPLING_DEFAULTS["temperature"],
        "top_p": config.SAMPLING_DEFAULTS["top_p"],
        "top_k": config.SAMPLING_DEFAULTS["top_k"],
        "seed": config.SAMPLING_DEFAULTS["seed"],
    }
    if config.SAMPLING_DEFAULTS["num_ctx"] is not None:
        options["num_ctx"] = config.SAMPLING_DEFAULTS["num_ctx"]
    if config.SAMPLING_DEFAULTS["num_predict"] is not None:
        options["num_predict"] = config.SAMPLING_DEFAULTS["num_predict"]

    # stream=True — та же причина, что в backends._analyze_ollama: видно
    # 'thinking' модели в реальном времени в консоли сервера.
    payload = {
        "model": model,
        "stream": True,
        "messages": messages,
        "options": options,
        "think": config.SAMPLING_DEFAULTS["think"],
    }

    tag = uuid.uuid4().hex[:6]
    content_parts: list[str] = []
    thinking_open = False
    final: dict = {}

    async with aiohttp.ClientSession(timeout=config.REQUEST_TIMEOUT) as session:
        async with session.post(f"{config.OLLAMA_HOST}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for raw_line in resp.content:
                line = raw_line.strip()
                if not line:
                    continue
                chunk = json.loads(line)

                msg = chunk.get("message", {})
                thinking = msg.get("thinking")
                if thinking:
                    if not thinking_open:
                        print(f"\n[chat:{tag}] --- think ---", flush=True)
                        thinking_open = True
                    print(thinking, end="", flush=True)

                piece = msg.get("content")
                if piece:
                    content_parts.append(piece)

                if chunk.get("done"):
                    final = chunk

    if thinking_open:
        print(f"\n[chat:{tag}] --- /think ---", flush=True)

    content = "".join(content_parts).strip()
    logging.info(
        "chat/Ollama: prompt_tokens=%s gen_tokens=%s done_reason=%s content_chars=%d load=%.1fs total=%.1fs",
        final.get("prompt_eval_count"), final.get("eval_count"), final.get("done_reason"),
        len(content), final.get("load_duration", 0) / 1e9, final.get("total_duration", 0) / 1e9,
    )
    return content


async def _chat_vllm(
    model: str, system: str | None, history: list, message: str, images: list[str],
) -> str:
    # Как и в backends._analyze_vllm: если удалось узнать реальный
    # max_model_len — обрезаем историю под него; если нет — шлём как
    # есть, vLLM сама вернёт ошибку, если промпт не влезет.
    max_model_len = await backends._get_vllm_context_window(model)
    if max_model_len:
        history = _truncate_history_for_vllm(
            history, system or "", message, len(images), max_model_len,
        )

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.extend(_history_to_vllm_messages(history))

    blocks = [{"type": "text", "text": message}] if message else []
    for img in images:
        blocks.append({"type": "image_url", "image_url": {"url": backends._ensure_data_url(img)}})
    messages.append({"role": "user", "content": blocks})

    payload = {
        "model": model,
        "temperature": config.SAMPLING_DEFAULTS["temperature"],
        "top_p": config.SAMPLING_DEFAULTS["top_p"],
        "seed": config.SAMPLING_DEFAULTS["seed"],
        "messages": messages,
    }

    async with aiohttp.ClientSession(timeout=config.REQUEST_TIMEOUT) as session:
        async with session.post(f"{config.VLLM_URL}/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()

    return data["choices"][0]["message"]["content"].strip()


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

    try:
        resolved_model = model or await backends._discover_model(backend)

        if backend == "vllm":
            content = await _chat_vllm(resolved_model, system, history, message, images)
        elif backend == "ollama":
            content = await _chat_ollama(resolved_model, system, history, message, images)
        else:
            raise ValueError(config._t("error.unknown_backend", backend=backend))
    except aiohttp.ClientConnectorError as e:
        if not allow_fallback:
            # Помечаем исключение тем бэкендом, который реально сейчас
            # недоступен — если это фолбэк-попытка (см. ветку ниже),
            # backend тут уже не исходно запрошенный, а тот, на который
            # переключились. Без этого вызывающий код (chat.py:
            # handle_chat) не может отличить "упал только исходный
            # бэкенд" от "упали оба" и в логе/ответе клиенту называет
            # исходный бэкенд, даже если на самом деле последним упал
            # другой.
            e.chat_backend = backend
            raise
        fallback_backend = "ollama" if backend == "vllm" else "vllm"
        logging.warning(
            "chat: бэкенд %r недоступен по подключению, пробую фолбэк на %r",
            backend, fallback_backend,
        )
        return await chat(
            message, images, backend=fallback_backend, model=None, system=system,
            history=history, allow_fallback=False,
        )

    return content, backend, resolved_model