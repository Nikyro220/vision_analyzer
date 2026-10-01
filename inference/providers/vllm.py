"""providers/vllm.py — vLLM (OpenAI-совместимый /v1/chat/completions, /v1/models)."""

from __future__ import annotations

import logging

import aiohttp

import config
from .base import Provider, ensure_data_url


# Грубая оценка размера токенов для истории, отправляемой в vLLM — точного
# токенайзера конкретной модели у нас тут нет, поэтому это защитный запас,
# а не честный расчёт. Используется только для решения "обрезать ли
# историю", когда реальный max_model_len удалось узнать через
# context_window(); сама vLLM всё равно провалидирует запрос и кинет
# ошибку, если промпт всё же не влез.
_EST_CHARS_PER_TOKEN = 4
_EST_TOKENS_PER_IMAGE = 1500
_CONTEXT_SAFETY_MARGIN = 0.9  # оставляем запас под system/ответ модели


def _estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // _EST_CHARS_PER_TOKEN)


def _truncate_history(
    history: list, system_prompt: str, current_text: str, current_image_count: int, max_model_len: int,
) -> list:
    """Отбрасывает старые сообщения истории, если оценочно не влезаем
    в контекст vLLM. Идёт с конца истории (свежие сообщения важнее),
    оставляет максимум, что влезает в safety-margin от max_model_len."""
    budget = int(max_model_len * _CONTEXT_SAFETY_MARGIN)
    fixed_tokens = (
        _estimate_tokens(system_prompt)
        + _estimate_tokens(current_text)
        + _EST_TOKENS_PER_IMAGE * current_image_count  # картинки текущего сообщения
    )

    def turn_tokens(turn: dict) -> int:
        return _estimate_tokens(turn.get("content")) + _EST_TOKENS_PER_IMAGE * len(turn.get("images") or [])

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
            total, max_model_len, _CONTEXT_SAFETY_MARGIN * 100,
        )

    return kept


def _history_to_messages(history: list) -> list[dict]:
    """История прошлых сообщений диалога → формат сообщений vLLM (OpenAI-style)."""
    messages = []
    for turn in history:
        blocks = []
        text = turn.get("content")
        if text:
            blocks.append({"type": "text", "text": text})
        for img in turn.get("images") or []:
            blocks.append({"type": "image_url", "image_url": {"url": ensure_data_url(img)}})
        messages.append({"role": turn["role"], "content": blocks or ""})
    return messages


class VllmProvider(Provider):
    name = "vllm"
    fallback = "ollama"

    def __init__(self) -> None:
        super().__init__()
        self._context_cache: dict[str, int] = {}

    @property
    def endpoint(self) -> str:
        return config.VLLM_URL

    def reset_caches(self) -> None:
        super().reset_caches()
        self._context_cache.clear()

    async def list_models(self) -> list[str]:
        async with aiohttp.ClientSession(timeout=config.DISCOVERY_TIMEOUT) as session:
            async with session.get(f"{config.VLLM_URL}/models") as resp:
                resp.raise_for_status()
                data = await resp.json()
        return [m["id"] for m in data.get("data", [])]

    async def context_window(self, model: str | None = None) -> int | None:
        """Спрашивает у vLLM реальный размер контекстного окна модели.

        Некоторые сборки vLLM отдают 'max_model_len' прямо в GET /v1/models
        (в отличие от Ollama, где это per-request параметр, у vLLM это то,
        с чем сервер был запущен — --max-model-len). Кэшируется по имени
        модели; кэш сбрасывается в reset_caches() (из /config).

        Возвращает None, если бэкенд недоступен, модель не нашлась в ответе,
        или конкретная сборка vLLM просто не отдаёт это поле — в таком случае
        вызывающий код должен считать контекст неизвестным и не обрезать
        историю "вслепую".
        """
        try:
            resolved_model = model or await self.discover_model()
            if resolved_model in self._context_cache:
                return self._context_cache[resolved_model]

            async with aiohttp.ClientSession(timeout=config.DISCOVERY_TIMEOUT) as session:
                async with session.get(f"{config.VLLM_URL}/models") as resp:
                    resp.raise_for_status()
                    data = await resp.json()

            for m in data.get("data", []):
                if m.get("id") == resolved_model:
                    max_len = m.get("max_model_len")
                    if isinstance(max_len, int):
                        self._context_cache[resolved_model] = max_len
                        return max_len
            return None
        except Exception as e:
            logging.debug("vLLM: не удалось узнать max_model_len (%s)", e)
            return None

    # --- /analyze ----------------------------------------------------------

    async def analyze(
        self, image_b64: str, image_mime: str, model: str, system_prompt: str, user_prompt: str,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{image_mime};base64,{image_b64}"},
                    },
                ],
            },
        ]

        base_payload = {
            "model": model,
            "temperature": config.SAMPLING_DEFAULTS["temperature"],
            "top_p": config.SAMPLING_DEFAULTS["top_p"],
            "seed": config.SAMPLING_DEFAULTS["seed"],
            "messages": messages,
        }

        async with aiohttp.ClientSession(timeout=config.REQUEST_TIMEOUT) as session:
            # Пытаемся получить строгий JSON через response_format (guided decoding).
            payload = {**base_payload, "response_format": {"type": "json_object"}}
            async with session.post(f"{config.VLLM_URL}/chat/completions", json=payload) as resp:
                if resp.status == 400:
                    # Некоторые сборки vLLM без guided-decoding backend отвергают
                    # response_format — повторяем запрос без него.
                    logging.warning("vLLM отклонил response_format, повторяю запрос без него")
                    async with session.post(f"{config.VLLM_URL}/chat/completions", json=base_payload) as resp2:
                        resp2.raise_for_status()
                        data = await resp2.json()
                else:
                    resp.raise_for_status()
                    data = await resp.json()

        return data["choices"][0]["message"]["content"].strip()

    # --- /chat -------------------------------------------------------------

    async def chat(
        self, model: str, system: str | None, history: list, message: str, images: list[str],
    ) -> str:
        # Если удалось узнать реальный max_model_len — обрезаем историю под
        # него; если нет — шлём как есть, vLLM сама вернёт ошибку, если
        # промпт не влезет.
        max_model_len = await self.context_window(model)
        if max_model_len:
            history = _truncate_history(history, system or "", message, len(images), max_model_len)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(_history_to_messages(history))

        blocks = [{"type": "text", "text": message}] if message else []
        for img in images:
            blocks.append({"type": "image_url", "image_url": {"url": ensure_data_url(img)}})
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
