"""
providers/gemini.py — Google Gemini через REST API (generateContent).

По умолчанию — gemini-2.5-flash (config.GEMINI_MODEL, можно поменять
переменной окружения VISION_ANALYZER_GEMINI_MODEL, через POST /config
(gemini_model) или на один запрос параметром model=...). Дополнительных
зависимостей нет — как и остальные провайдеры, ходит через aiohttp.

Ключ API (config.GEMINI_API_KEY) уходит только в заголовке x-goog-api-key —
не в URL, чтобы он не попадал в логи прокси и тексты исключений.

Особенности относительно локальных бэкендов:
  - Ollama/vLLM сами решают, что делать с картинкой; Gemini может ОТКЛОНИТЬ
    запрос фильтрами безопасности (promptFeedback.blockReason / finishReason
    = SAFETY и т.п.) — для риск-триажа это реальный сценарий, поэтому такие
    ответы превращаются в понятную RuntimeError (HTTP 400), а не в пустой
    отчёт. Порог фильтров можно задать VISION_ANALYZER_GEMINI_SAFETY.
  - Токены размышлений входят в лимит num_predict (maxOutputTokens): слишком
    маленький num_predict при включённом think даёт пустой/обрезанный ответ.
  - Картинки уходят inline (base64) — действует лимит Gemini на размер запроса.
"""

from __future__ import annotations

import logging
from urllib.parse import quote

import aiohttp

import config
from .base import Provider, split_data_url


# Размышления: SAMPLING_DEFAULTS["think"] — True / False / "low" / "medium" / "high".
# Gemini 2.5 управляется бюджетом токенов (thinkingBudget; 0 — выключить,
# отсутствие поля — динамический режим), Gemini 3.x — уровнем (thinkingLevel).
_THINK_BUDGETS = {"low": 1024, "medium": 8192, "high": 24576}

_SAFETY_CATEGORIES = (
    "HARM_CATEGORY_HARASSMENT",
    "HARM_CATEGORY_HATE_SPEECH",
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_DANGEROUS_CONTENT",
)


def _normalize_model(model: str) -> str:
    return model.removeprefix("models/")


def _thinking_config(think, model: str) -> dict | None:
    if think is True:
        return None  # динамическое размышление — дефолт модели, поле не нужно
    if model.startswith("gemini-3"):
        # У 3.x выключить размышления полностью нельзя — «выкл» = минимальный общий уровень.
        return {"thinkingLevel": think if think in _THINK_BUDGETS else "low"}
    budget = 0 if think is False else _THINK_BUDGETS.get(think)
    return None if budget is None else {"thinkingBudget": budget}


def _image_part(img: str, default_mime: str = "image/png") -> dict:
    mime, b64 = split_data_url(img, default_mime)
    return {"inlineData": {"mimeType": mime, "data": b64}}


class GeminiProvider(Provider):
    name = "gemini"
    # Облачный бэкенд: картинки покидают контур, поэтому на Gemini никогда не
    # уходим автоматически (он не является fallback у других провайдеров),
    # а при его недоступности автофолбэка на локальные бэкенды нет — выбор
    # бэкенда остаётся за вызывающей стороной.
    fallback = None

    @property
    def endpoint(self) -> str:
        return config.GEMINI_API_BASE

    def is_configured(self) -> bool:
        return bool(config.GEMINI_API_KEY)

    def _headers(self) -> dict:
        if not config.GEMINI_API_KEY:
            raise RuntimeError(config._t("error.gemini_no_key"))
        return {"x-goog-api-key": config.GEMINI_API_KEY}

    @staticmethod
    async def _check(resp: aiohttp.ClientResponse) -> None:
        """4xx → RuntimeError с текстом ошибки Google (неверный ключ, квота, модель
        недоступна...) — обработчики превращают его в понятный ответ клиенту;
        5xx → обычный ClientResponseError (500 на нашей стороне)."""
        if resp.status < 400:
            return
        if resp.status < 500:
            try:
                message = ((await resp.json()).get("error") or {}).get("message")
            except Exception:
                message = None
            raise RuntimeError(config._t(
                "error.gemini_api", status=resp.status, message=message or resp.reason or "",
            ))
        resp.raise_for_status()

    # --- модели ------------------------------------------------------------

    async def list_models(self) -> list[str]:
        """Модели, поддерживающие generateContent (эмбеддинги, TTS и т.п. отфильтрованы)."""
        names: list[str] = []
        params: dict = {"pageSize": 1000}
        async with aiohttp.ClientSession(timeout=config.DISCOVERY_TIMEOUT, headers=self._headers()) as session:
            while True:
                async with session.get(f"{config.GEMINI_API_BASE}/models", params=params) as resp:
                    await self._check(resp)
                    data = await resp.json()
                for m in data.get("models", []):
                    if "generateContent" in (m.get("supportedGenerationMethods") or []):
                        names.append(_normalize_model(m["name"]))
                token = data.get("nextPageToken")
                if not token:
                    break
                params = {"pageSize": 1000, "pageToken": token}
        return names

    async def discover_model(self) -> str:
        # «Первая в списке» для Gemini бессмысленна (там сотни моделей) —
        # без явного model=... берём модель из конфига.
        return config.GEMINI_MODEL

    async def _probe(self) -> str:
        # Для /health: проверяем и ключ, и связь одним списком моделей,
        # но показываем ту модель, которая реально будет использоваться.
        await self.list_models()
        return config.GEMINI_MODEL

    async def ping(self) -> dict:
        if not self.is_configured():
            return {"ok": False, "endpoint": self.endpoint, "error": config._t("error.gemini_no_key")}
        return await super().ping()

    # --- запрос / разбор ответа -------------------------------------------

    def _generation_config(self, model: str, json_mode: bool) -> dict:
        s = config.SAMPLING_DEFAULTS
        gen = {"temperature": s["temperature"], "topP": s["top_p"], "seed": s["seed"]}
        if s["top_k"] and s["top_k"] > 0:  # у Ollama -1/0 = «выключено» — Gemini такое не принимает
            gen["topK"] = s["top_k"]
        if s["num_predict"] is not None:
            gen["maxOutputTokens"] = s["num_predict"]
        if json_mode:
            gen["responseMimeType"] = "application/json"
        thinking = _thinking_config(s["think"], model)
        if thinking:
            gen["thinkingConfig"] = thinking
        return gen

    async def _generate(
        self, model: str, system: str | None, contents: list[dict], json_mode: bool,
    ) -> str:
        model = _normalize_model(model)
        body: dict = {
            "contents": contents,
            "generationConfig": self._generation_config(model, json_mode),
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if config.GEMINI_SAFETY:
            body["safetySettings"] = [
                {"category": c, "threshold": config.GEMINI_SAFETY} for c in _SAFETY_CATEGORIES
            ]

        url = f"{config.GEMINI_API_BASE}/models/{quote(model, safe='')}:generateContent"
        async with aiohttp.ClientSession(timeout=config.REQUEST_TIMEOUT, headers=self._headers()) as session:
            async with session.post(url, json=body) as resp:
                await self._check(resp)
                data = await resp.json()

        return self._extract_text(data, model)

    @staticmethod
    def _extract_text(data: dict, model: str) -> str:
        block_reason = (data.get("promptFeedback") or {}).get("blockReason")
        if block_reason:
            raise RuntimeError(config._t("error.gemini_blocked", reason=block_reason))

        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError(config._t("error.gemini_empty"))

        candidate = candidates[0]
        parts = (candidate.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if not p.get("thought")).strip()
        finish = candidate.get("finishReason")

        usage = data.get("usageMetadata") or {}
        logging.info(
            "Gemini(%s): prompt_tokens=%s gen_tokens=%s thought_tokens=%s finish_reason=%s content_chars=%d",
            model, usage.get("promptTokenCount"), usage.get("candidatesTokenCount"),
            usage.get("thoughtsTokenCount"), finish, len(text),
        )

        if not text and finish not in (None, "STOP", "MAX_TOKENS"):
            raise RuntimeError(config._t("error.gemini_blocked", reason=finish))
        if finish == "MAX_TOKENS":
            logging.warning(
                "Gemini(%s): ответ обрезан по maxOutputTokens (num_predict) — токены размышлений "
                "тоже входят в этот лимит; увеличьте num_predict или уменьшите think (/sampling)", model,
            )
        return text

    # --- /analyze ----------------------------------------------------------

    async def analyze(
        self, image_b64: str, image_mime: str, model: str, system_prompt: str, user_prompt: str,
    ) -> str:
        contents = [{
            "role": "user",
            "parts": [
                {"text": user_prompt},
                {"inlineData": {"mimeType": image_mime, "data": image_b64}},
            ],
        }]
        return await self._generate(model, system_prompt, contents, json_mode=True)

    # --- /chat -------------------------------------------------------------

    @staticmethod
    def _history_to_contents(history: list) -> list[dict]:
        """История диалога → contents Gemini (роль assistant называется 'model')."""
        contents = []
        for turn in history:
            parts = []
            if turn.get("content"):
                parts.append({"text": turn["content"]})
            parts.extend(_image_part(img) for img in turn.get("images") or [])
            if parts:  # Gemini отклоняет сообщения с пустым parts
                contents.append({"role": "model" if turn["role"] == "assistant" else "user", "parts": parts})
        return contents

    async def chat(
        self, model: str, system: str | None, history: list, message: str, images: list[str],
    ) -> str:
        contents = self._history_to_contents(history)
        parts = [{"text": message}] if message else []
        parts.extend(_image_part(img) for img in images)
        contents.append({"role": "user", "parts": parts})
        return await self._generate(model, system, contents, json_mode=False)
