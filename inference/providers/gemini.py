"""
providers/gemini.py — Google Gemini через Interactions API (POST /v1beta/interactions).

Раньше провайдер ходил в generateContent. Google с июня 2026 считает его устаревшим (legacy):
новые модели и возможности выходят в Interactions API, а модели 2.5 закрыты для новых
пользователей (404 «no longer available to new users»). Список моделей по-прежнему берётся из
Models API (GET /v1beta/models) — он общий для обоих API.

По умолчанию — config.GEMINI_MODEL (VISION_ANALYZER_GEMINI_MODEL, POST /config → gemini_model
или model=... на один запрос). Дополнительных зависимостей нет — как и остальные провайдеры,
ходит через aiohttp.

Ключ API сервер НЕ хранит: клиент присылает его в каждом запросе заголовком
X-Api-Key-Gemini (см. providers/base.py), отсюда он уходит в Google только в заголовке
x-goog-api-key — не в URL, чтобы не попадать в логи прокси и тексты исключений.

Особенности относительно generateContent:
  - Запрос не хранится у Google: по умолчанию Interactions API сохраняет запросы (store=true:
    55 дней на платном тарифе, 1 день на бесплатном), а здесь уходят картинки на риск-триаж,
    поэтому всегда store=false. Побочный эффект: previous_interaction_id недоступен, историю
    /chat мы передаём целиком (stateless, массив steps).
  - Ответ — массив steps; текст модели лежит в шагах type=model_output (шаги thought и
    служебные пропускаются).
  - Размышления: вместо thinkingBudget (2.5) — generation_config.thinking_level
    (minimal/low/medium/high). У моделей gemini-2.* этого поля нет — для них не передаётся.
  - В generation_config не передаются top_p и top_k: в справочнике Interactions API их нет
    (SAMPLING_DEFAULTS их по-прежнему хранит — для Ollama/vLLM).
  - Gemini может ОТКЛОНИТЬ запрос фильтрами безопасности — для риск-триажа это реальный сценарий,
    поэтому такие ответы превращаются в понятную RuntimeError (HTTP 400), а не в пустой отчёт.
    Порог задаётся VISION_ANALYZER_GEMINI_SAFETY → safety_settings.
  - Токены размышлений входят в лимит num_predict (max_output_tokens): слишком маленький
    num_predict при включённом think даёт пустой/обрезанный ответ (status=incomplete).
  - Картинки уходят inline (base64) — действует лимит Gemini на размер запроса.
"""

from __future__ import annotations

import asyncio
import logging
from urllib.parse import quote

import aiohttp

import config
from .base import Credential, Provider, split_data_url


# Размышления: SAMPLING_DEFAULTS["think"] — True / False / "low" / "medium" / "high".
# True — динамический режим по умолчанию (поле не передаём); False — минимум, который
# модель позволяет; уровни low/medium/high передаются как есть.
_THINK_LEVELS = ("low", "medium", "high")

# Категории фильтров безопасности в Interactions API (safety_settings[].type) — snake_case.
_SAFETY_CATEGORIES = ("harassment", "hate_speech", "sexually_explicit", "dangerous_content")

# Для /analyze и /chat годятся только текстовые/мультимодальные модели семейств Gemini и Gemma
# (картинка → текст). Models API отдаёт ВСЁ подряд: TTS, генерацию картинок/видео/музыки, Live
# (он идёт по WebSocket, а не по REST), эмбеддинги, транскрипцию, computer-use, робототехнику,
# агентов (antigravity, deep-research). Поэтому сначала белый список по префиксу, потом исключения
# по подстрокам — новые медиа-модели с незнакомым именем (nano-banana, lyria, ...) отсекаются сами.
_TEXT_MODEL_PREFIXES = ("gemini-", "gemma-")
_NON_TEXT_MARKERS = (
    "-tts", "-image", "-live", "native-audio", "embedding", "transcribe", "computer-use",
    "robotics", "omni", "aqa",
)

# Допустимые mime картинок в Interactions API; частые синонимы приводим к ним.
_MIME_ALIASES = {"image/jpg": "image/jpeg", "image/pjpeg": "image/jpeg"}


def _normalize_model(model: str) -> str:
    return model.removeprefix("models/")


def _is_text_model(name: str) -> bool:
    n = name.lower()
    return n.startswith(_TEXT_MODEL_PREFIXES) and not any(m in n for m in _NON_TEXT_MARKERS)


def _thinking_level(think, model: str) -> str | None:
    if think is True or model.startswith("gemini-2."):
        return None  # динамический режим по умолчанию / у 2.x поля thinking_level нет
    if think is False:
        # Pro-модели не умеют «minimal» — для них минимум «low».
        return "low" if "pro" in model else "minimal"
    return think if think in _THINK_LEVELS else None


def _image_part(img: str, default_mime: str = "image/png") -> dict:
    mime, b64 = split_data_url(img, default_mime)
    return {"type": "image", "mime_type": _MIME_ALIASES.get(mime, mime), "data": b64}


def _text_part(text: str) -> dict:
    return {"type": "text", "text": text}


class GeminiProvider(Provider):
    name = "gemini"
    label = "Google Gemini"
    # Облачный бэкенд: картинки покидают контур, поэтому на Gemini никогда не
    # уходим автоматически (он не является fallback у других провайдеров),
    # а при его недоступности автофолбэка на локальные бэкенды нет — выбор
    # бэкенда остаётся за вызывающей стороной.
    fallback = None
    # num_ctx не используется; think -> thinking_level (см. _thinking_level)
    # (sampling_keys — по умолчанию из базового класса: без num_ctx)

    # Ключ хранит клиент (vision_app, в БД, зашифрованным) и присылает его в каждом запросе
    # заголовком X-Api-Key-Gemini. Здесь он нигде не сохраняется, в том числе в памяти.
    credential = Credential(
        label="API-ключ Google AI",
        hint="Хранится в БД веб-приложения в зашифрованном виде и передаётся серверу анализа "
             "в каждом запросе; сам сервер анализа ключ не сохраняет.",
    )

    def __init__(self) -> None:
        super().__init__()
        self._session: aiohttp.ClientSession | None = None
        self._session_loop: asyncio.AbstractEventLoop | None = None

    def _get_session(self) -> aiohttp.ClientSession:
        """Одна сессия на все запросы к Google: keep-alive избавляет от нового TLS-рукопожатия
        на каждый вызов. Заголовки с ключом задаются на каждый запрос отдельно — в самой
        сессии ключа нет."""
        loop = asyncio.get_running_loop()
        if self._session is None or self._session.closed or self._session_loop is not loop:
            self._session = aiohttp.ClientSession()
            self._session_loop = loop
        return self._session

    async def aclose(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
        self._session = None

    @property
    def endpoint(self) -> str:
        return config.GEMINI_API_BASE

    def _headers(self) -> dict:
        key = self.api_key
        if not key:
            raise RuntimeError(config._t("error.gemini_no_key"))
        return {"x-goog-api-key": key}

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
            text = config._t(
                "error.gemini_api", status=resp.status, message=message or resp.reason or "",
            )
            if resp.status == 404:
                # «Модель закрыта для новых пользователей / выведена из эксплуатации» —
                # подсказываем, где взять актуальную, а не гадаем, какая сейчас рекомендована.
                text += " " + config._t("error.gemini_model_unavailable_hint")
            raise RuntimeError(text)
        resp.raise_for_status()

    # --- модели ------------------------------------------------------------

    async def list_models(self) -> list[str]:
        """Текстовые/мультимодальные модели из Models API (генерация медиа, TTS, Live,
        эмбеддинги отфильтрованы).

        Список — это всё, что Google отдаёт по ключу; он НЕ гарантирует, что модель можно
        вызвать именно этому аккаунту (gemini-2.5-* значатся в списке, но новым пользователям
        отвечают 404)."""
        names: list[str] = []
        params: dict = {"pageSize": 1000}
        session = self._get_session()
        headers = self._headers()
        while True:
            # Таймаут — на каждую страницу отдельно (раньше один общий на весь цикл пагинации).
            async with session.get(
                f"{config.GEMINI_API_BASE}/models", params=params, headers=headers,
                timeout=config.DISCOVERY_TIMEOUT,
            ) as resp:
                await self._check(resp)
                data = await resp.json()
            for m in data.get("models", []):
                name = _normalize_model(m["name"])
                if "generateContent" in (m.get("supportedGenerationMethods") or []) \
                        and _is_text_model(name):
                    names.append(name)
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
        # Для /health: ОДИН дешёвый запрос — метаданные выбранной модели (GET /models/<model>).
        # Он проверяет и ключ, и связь, и что модель вообще существует. Генерацию не запускаем
        # (токены на каждый опрос статуса), поэтому «закрыта для новых пользователей» он
        # не ловит — такая ошибка проявится на первом /analyze или /chat.
        model = _normalize_model(config.GEMINI_MODEL)
        async with self._get_session().get(
            f"{config.GEMINI_API_BASE}/models/{quote(model, safe='')}", headers=self._headers(),
            timeout=config.DISCOVERY_TIMEOUT,
        ) as resp:
            await self._check(resp)
        return config.GEMINI_MODEL

    async def ping(self) -> dict:
        if not self.is_configured():
            return {"ok": False, "endpoint": self.endpoint, "error": config._t("error.gemini_no_key")}
        return await super().ping()

    # --- запрос / разбор ответа -------------------------------------------

    def _generation_config(self, model: str) -> dict:
        s = config.SAMPLING_DEFAULTS
        gen: dict = {"temperature": s["temperature"]}
        if s["seed"] is not None:
            gen["seed"] = s["seed"]
        if s["num_predict"] is not None:
            gen["max_output_tokens"] = s["num_predict"]
        level = _thinking_level(s["think"], model)
        if level:
            gen["thinking_level"] = level
        return gen

    async def _generate(
        self, model: str, system: str | None, input_: list[dict], json_mode: bool,
    ) -> str:
        model = _normalize_model(model)
        body: dict = {
            "model": model,
            "input": input_,
            "store": False,  # картинки не должны оседать у Google (см. докстринг модуля)
            "generation_config": self._generation_config(model),
        }
        if system:
            body["system_instruction"] = system
        if json_mode:
            body["response_format"] = {"type": "text", "mime_type": "application/json"}
        if config.GEMINI_SAFETY:
            body["safety_settings"] = [
                {"type": c, "threshold": config.GEMINI_SAFETY.lower()} for c in _SAFETY_CATEGORIES
            ]

        async with self._get_session().post(
            f"{config.GEMINI_API_BASE}/interactions", json=body, headers=self._headers(),
            timeout=config.REQUEST_TIMEOUT,
        ) as resp:
            await self._check(resp)
            data = await resp.json()

        return self._extract_text(data, model)

    @staticmethod
    def _extract_text(data: dict, model: str) -> str:
        status = data.get("status")
        errors = data.get("errors") or []
        if status in ("failed", "cancelled"):
            reason = "; ".join(
                str(e.get("message") or e.get("code") or "") for e in errors if isinstance(e, dict)
            ) or status
            raise RuntimeError(config._t("error.gemini_blocked", reason=reason))

        parts: list[str] = []
        for step in data.get("steps") or []:
            # Нужны только ответы модели: user_input (если вернулся), thought и пр. пропускаем.
            if step.get("type") != "model_output":
                continue
            content = step.get("content")
            if isinstance(content, str):
                parts.append(content)
                continue
            for block in content or []:
                if block.get("type") == "text" and block.get("text"):
                    parts.append(block["text"])
        text = "".join(parts).strip()

        usage = data.get("usage") or {}
        logging.info(
            "Gemini(%s): prompt_tokens=%s gen_tokens=%s thought_tokens=%s status=%s content_chars=%d",
            model, usage.get("total_input_tokens"), usage.get("total_output_tokens"),
            usage.get("total_thought_tokens"), status, len(text),
        )

        if not text:
            if status not in (None, "completed"):
                raise RuntimeError(config._t("error.gemini_blocked", reason=status))
            raise RuntimeError(config._t("error.gemini_empty"))
        if status == "incomplete":
            logging.warning(
                "Gemini(%s): ответ обрезан по max_output_tokens (num_predict) — токены размышлений "
                "тоже входят в этот лимит; увеличьте num_predict или уменьшите think (/sampling)", model,
            )
        return text

    # --- /analyze ----------------------------------------------------------

    async def analyze(
        self, image_b64: str, image_mime: str, model: str, system_prompt: str, user_prompt: str,
    ) -> str:
        # Картинка перед текстом — порядок из руководств Google для одного изображения.
        content = [
            {"type": "image", "mime_type": _MIME_ALIASES.get(image_mime, image_mime), "data": image_b64},
            _text_part(user_prompt),
        ]
        return await self._generate(model, system_prompt, content, json_mode=True)

    # --- /chat -------------------------------------------------------------

    @staticmethod
    def _history_to_steps(history: list) -> list[dict]:
        """История диалога → steps (stateless-режим): user_input / model_output.

        Документация просит пересылать модельные шаги «как получены» (thought с подписями) —
        мы храним только текст, поэтому подписей нет. Для обычного текстового диалога без
        инструментов это работает; если на конкретной модели /chat начнёт падать на втором
        сообщении — смотреть сюда."""
        steps = []
        for turn in history:
            is_model = turn["role"] == "assistant"
            parts = []
            if turn.get("content"):
                parts.append(_text_part(turn["content"]))
            if not is_model:  # картинки — только во входе пользователя
                parts.extend(_image_part(img) for img in turn.get("images") or [])
            if parts:  # пустые шаги API не принимает
                steps.append({"type": "model_output" if is_model else "user_input", "content": parts})
        return steps

    async def chat(
        self, model: str, system: str | None, history: list, message: str, images: list[str],
    ) -> str:
        steps = self._history_to_steps(history)
        parts = [_text_part(message)] if message else []
        parts.extend(_image_part(img) for img in images)
        steps.append({"type": "user_input", "content": parts})
        return await self._generate(model, system, steps, json_mode=False)