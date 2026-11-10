"""
providers/lmstudio.py — LM Studio (OpenAI-совместимый /v1/chat/completions + нативный /api/v0/models).

Отличия от vLLM, из-за которых это отдельный провайдер, а не просто другой URL:
  - структурированный вывод: LM Studio понимает только response_format={"type": "json_schema"},
    а {"type": "json_object"} (как у vLLM) отвергает с 400;
  - GET /v1/models отдаёт ВСЕ скачанные модели, в том числе эмбеддинговые, и не говорит, какая
    из них загружена. Нативный GET /api/v0/models (LM Studio 0.3.6+) даёт type (llm / vlm /
    embeddings), state (loaded / not-loaded) и loaded_context_length — по нему отбираются
    чат-модели и узнаётся реальный размер контекста. Если нативного API нет (старая версия) —
    откат на /v1/models;
  - reasoning-модели без включённой в LM Studio опции «отделять reasoning_content» кладут
    рассуждения прямо в content в тегах <think>…</think> — их нужно срезать до разбора JSON.

Контекст (n_ctx) задаётся при ЗАГРУЗКЕ модели в LM Studio и в запросе не меняется — поэтому
num_ctx в настройках нет, как и у vLLM.
"""

from __future__ import annotations

import logging
import re

import aiohttp

import config
from .base import Provider, ensure_data_url
from .settings import Setting
from .vllm import _history_to_messages, _truncate_history


# «Любой JSON-объект»: схема-заглушка. Конкретная схема отчёта проверяется уже на стороне
# backends.py, здесь нужно лишь заставить модель вернуть валидный JSON (грамматика llama.cpp).
_JSON_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "response", "schema": {"type": "object"}},
}

_THINK_RE = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL)


def _clean_content(data: dict) -> str:
    """choices[0].message.content без ведущего блока <think>…</think>."""
    content = data["choices"][0]["message"].get("content") or ""
    return _THINK_RE.sub("", content, count=1).strip()


class LmStudioProvider(Provider):
    name = "lmstudio"
    label = "LM Studio"
    fallback = "ollama"
    # Контекст задаётся при загрузке модели в LM Studio, think-режим — тоже её настройкой модели,
    # поэтому ни num_ctx, ни think здесь нет.
    settings_spec = (
        Setting("temperature", "float", 0.0, "temperature", min=0, max=2,
                hint="0 — жадный выбор: ответ детерминирован, но модель может зацикливаться."),
        Setting("top_p", "float", 1.0, "top_p", min=0, max=1, min_exclusive=True),
        Setting("top_k", "int", None, "top_k", min=1, max=100000, nullable=True,
                hint="auto — не передавать (берётся значение из настроек модели в LM Studio)."),
        Setting("seed", "int", 42, "seed", nullable=True, hint="auto — случайный seed."),
        Setting("num_predict", "int", None, "max_tokens", min=1, max=1048576, nullable=True,
                hint="Максимум токенов в ответе (max_tokens). auto — ограничивает только контекст модели."),
    )

    def _generation_params(self) -> dict:
        """Поля запроса из текущих настроек LM Studio (читаются в момент запроса)."""
        v = self.settings
        params: dict = {"temperature": v["temperature"], "top_p": v["top_p"]}
        if v["top_k"] is not None:
            params["top_k"] = v["top_k"]
        if v["seed"] is not None:
            params["seed"] = v["seed"]
        if v["num_predict"] is not None:
            params["max_tokens"] = v["num_predict"]
        return params

    @property
    def endpoint(self) -> str:
        return config.LMSTUDIO_URL

    # --- модели ------------------------------------------------------------

    @staticmethod
    def _native_models_url() -> str:
        """http://host:1234/v1 -> http://host:1234/api/v0/models (нативный API без /v1)."""
        root = config.LMSTUDIO_URL.rstrip("/")
        if root.endswith("/v1"):
            root = root[: -len("/v1")]
        return f"{root}/api/v0/models"

    @staticmethod
    async def _get_json(url: str) -> dict:
        async with aiohttp.ClientSession(timeout=config.DISCOVERY_TIMEOUT) as session:
            async with session.get(url) as resp:
                resp.raise_for_status()
                return await resp.json()

    async def _native_models(self) -> list[dict] | None:
        """Записи GET /api/v0/models; None — нативного API нет (старый LM Studio, 404 и т.п.).
        Ошибки подключения/таймаут не глотаются: недоступный сервер — это ошибка, а не «нет API»."""
        try:
            data = await self._get_json(self._native_models_url())
        except aiohttp.ClientResponseError:  # 404/5xx или ответ не JSON
            return None
        return [m for m in data.get("data", []) if isinstance(m, dict) and m.get("id")]

    async def list_models(self) -> list[str]:
        """Чат-модели LM Studio: сначала загруженные, среди них — vision-модели (vlm) вперёд.
        Эмбеддинговые модели отбрасываются, иначе автоопределение могло бы выбрать их."""
        native = await self._native_models()
        if native is not None:
            chat = [m for m in native if m.get("type") != "embeddings"]
            # sorted стабилен: внутри групп сохраняется порядок, который отдал LM Studio.
            chat.sort(key=lambda m: (m.get("state") != "loaded", m.get("type") != "vlm"))
            return [m["id"] for m in chat]

        data = await self._get_json(f"{config.LMSTUDIO_URL}/models")
        return [m["id"] for m in data.get("data", []) if "embed" not in m["id"].lower()]

    async def context_window(self, model: str | None = None) -> int | None:
        """Реальный контекст ЗАГРУЖЕННОЙ модели (loaded_context_length из /api/v0/models).

        Не кэшируется: пользователь может перезагрузить модель с другим контекстом, а запрос
        к локальному серверу дешёвый. None — нативного API нет, модель не загружена (max_context_length
        — это потолок модели, а не то, с чем она загружена, поэтому не годится) или сервер недоступен;
        вызывающий код тогда не обрезает историю вслепую.
        """
        try:
            resolved_model = model or await self.discover_model()
            native = await self._native_models()
            for m in native or []:
                if m["id"] == resolved_model and m.get("state") == "loaded":
                    ctx = m.get("loaded_context_length")
                    return ctx if isinstance(ctx, int) and ctx > 0 else None
            return None
        except Exception as e:
            logging.debug("LM Studio: не удалось узнать loaded_context_length (%s)", e)
            return None

    # --- общий запрос ------------------------------------------------------

    async def _complete(self, payload: dict, *, json_mode: bool) -> str:
        """POST /chat/completions. json_mode=True — просим json_schema; если LM Studio ответил 400
        (модель/версия не умеет структурированный вывод), один раз повторяем без него."""
        url = f"{config.LMSTUDIO_URL}/chat/completions"
        async with aiohttp.ClientSession(timeout=config.REQUEST_TIMEOUT) as session:
            if json_mode:
                body = {**payload, "response_format": _JSON_RESPONSE_FORMAT}
                async with session.post(url, json=body) as resp:
                    if resp.status != 400:
                        resp.raise_for_status()
                        return _clean_content(await resp.json())
                    logging.warning(
                        "LM Studio ответил 400 на response_format=json_schema, повторяю запрос без него: %.300s",
                        await resp.text(),
                    )
            async with session.post(url, json=payload) as resp:
                resp.raise_for_status()
                return _clean_content(await resp.json())

    # --- /analyze ----------------------------------------------------------

    async def analyze(
        self, image_b64: str, image_mime: str, model: str, system_prompt: str, user_prompt: str,
    ) -> str:
        payload = {
            "model": model,
            **self._generation_params(),
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{image_b64}"}},
                    ],
                },
            ],
        }
        return await self._complete(payload, json_mode=True)

    # --- /chat -------------------------------------------------------------

    async def chat(
        self, model: str, system: str | None, history: list, message: str, images: list[str],
    ) -> str:
        # Реальный контекст загруженной модели известен — обрезаем историю под него;
        # неизвестен — шлём как есть, LM Studio сама вернёт ошибку, если промпт не влезет.
        n_ctx = await self.context_window(model)
        if n_ctx:
            history = _truncate_history(history, system or "", message, len(images), n_ctx)

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.extend(_history_to_messages(history))

        blocks = [{"type": "text", "text": message}] if message else []
        for img in images:
            blocks.append({"type": "image_url", "image_url": {"url": ensure_data_url(img)}})
        messages.append({"role": "user", "content": blocks})

        payload = {"model": model, **self._generation_params(), "messages": messages}
        return await self._complete(payload, json_mode=False)
