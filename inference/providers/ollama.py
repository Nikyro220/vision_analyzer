"""providers/ollama.py — Ollama (/api/chat, /api/tags)."""

from __future__ import annotations

import json
import logging
import uuid

import aiohttp

import config
from .base import ALL_SAMPLING_KEYS, Provider, sampling_options, strip_data_url


class OllamaProvider(Provider):
    name = "ollama"
    label = "Ollama"
    fallback = "vllm"
    sampling_keys = ALL_SAMPLING_KEYS  # единственный бэкенд, где num_ctx — per-request параметр

    @property
    def endpoint(self) -> str:
        return config.OLLAMA_HOST

    async def list_models(self) -> list[str]:
        async with aiohttp.ClientSession(timeout=config.DISCOVERY_TIMEOUT) as session:
            async with session.get(f"{config.OLLAMA_HOST}/api/tags") as resp:
                resp.raise_for_status()
                data = await resp.json()
        return [m["name"] for m in data.get("models", [])]

    # --- общий стриминговый запрос -----------------------------------------

    async def _stream_chat(self, payload: dict, label: str) -> str:
        """POST /api/chat со stream=True.

        stream=True — чтобы видеть 'thinking' модели в реальном времени в
        консоли сервера, а не ждать молча всю генерацию (может занимать много
        минут при включённом think). ВНИМАНИЕ: если запросы идут параллельно
        (asyncio.gather), вывод нескольких запросов перемежается в одной
        консоли — тег [xxxxxx] (label + 6 символов uuid) отличает потоки.
        """
        tag = f"{label}{uuid.uuid4().hex[:6]}"
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
                            print(f"\n[{tag}] --- think ---", flush=True)
                            thinking_open = True
                        print(thinking, end="", flush=True)

                    piece = msg.get("content")
                    if piece:
                        content_parts.append(piece)

                    if chunk.get("done"):
                        final = chunk

        if thinking_open:
            print(f"\n[{tag}] --- /think ---", flush=True)

        content = "".join(content_parts).strip()
        logging.info(
            "%sOllama: prompt_tokens=%s gen_tokens=%s done_reason=%s content_chars=%d load=%.1fs total=%.1fs",
            "chat/" if label else "",
            final.get("prompt_eval_count"), final.get("eval_count"), final.get("done_reason"),
            len(content), final.get("load_duration", 0) / 1e9, final.get("total_duration", 0) / 1e9,
        )
        return content

    # --- /analyze ----------------------------------------------------------

    async def analyze(
        self, image_b64: str, image_mime: str, model: str, system_prompt: str, user_prompt: str,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt, "images": [image_b64]},
        ]
        payload = {
            "model": model,
            "stream": True,
            "format": "json",
            "messages": messages,
            "options": sampling_options(),
            "think": config.SAMPLING_DEFAULTS["think"],
        }
        return await self._stream_chat(payload, label="")

    # --- /chat -------------------------------------------------------------

    @staticmethod
    def _history_to_messages(history: list) -> list[dict]:
        """История прошлых сообщений диалога → формат сообщений Ollama."""
        messages = []
        for turn in history:
            entry = {"role": turn["role"], "content": turn.get("content", "")}
            images = turn.get("images")
            if images:
                entry["images"] = [strip_data_url(img) for img in images]
            messages.append(entry)
        return messages

    async def chat(
        self, model: str, system: str | None, history: list, message: str, images: list[str],
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(self._history_to_messages(history))
        user_entry = {"role": "user", "content": message}
        if images:
            user_entry["images"] = [strip_data_url(img) for img in images]
        messages.append(user_entry)

        # Без format=json: /chat не парсит ответ модели, отдаёт как есть.
        payload = {
            "model": model,
            "stream": True,
            "messages": messages,
            "options": sampling_options(),
            "think": config.SAMPLING_DEFAULTS["think"],
        }
        return await self._stream_chat(payload, label="chat:")
