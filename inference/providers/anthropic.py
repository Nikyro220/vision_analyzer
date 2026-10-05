"""
providers/anthropic.py — Anthropic Claude через Messages API (POST /v1/messages).

По умолчанию — config.ANTHROPIC_MODEL (VISION_ANALYZER_ANTHROPIC_MODEL, POST /config →
anthropic_model или model=... на один запрос). Дополнительных зависимостей нет (SDK `anthropic`
не нужен) — как и остальные провайдеры, ходит через aiohttp.

Ключ API сервер НЕ хранит: клиент присылает его в каждом запросе заголовком
X-Api-Key-Anthropic (см. providers/base.py), отсюда он уходит в Anthropic только в заголовке
x-api-key — не в URL, чтобы не попадать в логи прокси и тексты исключений.

Особенности относительно Gemini-провайдера:
  - Нет «JSON-режима» без схемы. Принудительный JSON (prefill ответа ассистента) на актуальных
    моделях Claude закрыт — запрос с ним возвращает 400. Поэтому в /analyze требование «только
    JSON» добавляется в system-промпт, а из ответа вырезаются ```-обёртка и текст вокруг объекта
    (_extract_json): вызывающий код (backends.py) делает json.loads(content) напрямую.
  - Параметры сэмплинга НЕ передаются. На актуальных моделях temperature/top_p/top_k с
    нестандартными значениями дают 400, а seed в Messages API нет вовсе. Поэтому
    sampling_keys = (num_predict, think), а temperature/top_p/top_k/seed из /sampling
    игнорируются (они по-прежнему действуют на Ollama/vLLM).
  - Размышления: вместо бюджета токенов — adaptive thinking, глубина задаётся
    output_config.effort (low/medium/high). think=true (по умолчанию) — ничего не передаём,
    модель работает как задумано; think="low"/"medium"/"high" → effort; think=false → effort=low
    (полностью выключить thinking нельзя: thinking={"type": "disabled"} на новых моделях даёт 400).
    effort поддерживают не все модели (например, Haiku 4.5 — нет): для них оставляйте think=true.
  - Токены размышлений входят в max_tokens. Он у Messages API обязателен: берётся из num_predict
    (/sampling), иначе config.ANTHROPIC_MAX_TOKENS. Слишком малое значение при включённом
    thinking даёт обрезанный ответ (stop_reason=max_tokens).
  - Claude может ОТКАЗАТЬСЯ обрабатывать запрос (stop_reason=refusal) — для риск-триажа это
    реальный сценарий, поэтому такой ответ превращается в понятную RuntimeError (HTTP 400),
    а не в пустой отчёт.
  - Картинки уходят inline (base64). Claude принимает только jpeg/png/gif/webp — остальное
    (bmp, tiff, ...) перекодируется в PNG (_ensure_supported_image). Лимиты на размер картинки
    и запроса действуют на стороне Anthropic, их текст пробрасывается в ошибке API.
  - У Messages API нет параметра, аналогичного store=false у Gemini: срок хранения запросов
    определяется настройками аккаунта Anthropic (например, zero data retention).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import re
from urllib.parse import quote

import aiohttp
from PIL import Image

import config
from .base import Credential, Provider, split_data_url


# Уровни effort, которые можно получить из SAMPLING_DEFAULTS["think"] (True → ничего не шлём).
_EFFORT_LEVELS = ("low", "medium", "high")

# Допустимые mime картинок в Messages API; частые синонимы приводим к ним.
_SUPPORTED_MIMES = ("image/jpeg", "image/png", "image/gif", "image/webp")
_MIME_ALIASES = {"image/jpg": "image/jpeg", "image/pjpeg": "image/jpeg"}

# Добавляется к system-промпту в /analyze: JSON-режима у Messages API нет, а prefill закрыт.
_JSON_ONLY_SUFFIX = (
    "\n\nReturn ONLY a single valid JSON object. Do not wrap it in Markdown code fences and "
    "do not add any text before or after it."
)

_FENCE_RE = re.compile(r"^```[A-Za-z0-9_-]*\s*(.*?)\s*```$", re.DOTALL)


def _effort(think) -> str | None:
    if think is True:
        return None  # поведение по умолчанию — поле не передаём
    if think is False:
        return "low"  # минимум; полностью выключить thinking на новых моделях нельзя
    return think if think in _EFFORT_LEVELS else None


def _extract_json(text: str) -> str:
    """Достаёт JSON-объект из ответа модели: убирает ```-обёртку и пояснения вокруг объекта.
    Если разобрать не удалось — возвращает текст как есть (вызывающий код сам сообщит о
    невалидном JSON и покажет сырой ответ)."""
    candidate = text.strip()
    m = _FENCE_RE.match(candidate)
    if m:
        candidate = m.group(1)
    for attempt in (candidate, _outer_braces(candidate)):
        if not attempt:
            continue
        try:
            json.loads(attempt)
            return attempt
        except (json.JSONDecodeError, TypeError):
            continue
    return text


def _outer_braces(text: str) -> str | None:
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end > start else None


def _convert_to_png(b64: str) -> str:
    """Синхронная перекодировка картинки в PNG (вызывается через executor)."""
    img = Image.open(io.BytesIO(base64.b64decode(b64)))
    img.load()
    has_alpha = img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info)
    img = img.convert("RGBA" if has_alpha else "RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


async def _ensure_supported_image(b64: str, mime: str) -> tuple[str, str]:
    """(b64, mime) → то же, но в формате, который принимает Claude (jpeg/png/gif/webp)."""
    mime = _MIME_ALIASES.get(mime, mime)
    if mime in _SUPPORTED_MIMES:
        return b64, mime
    logging.info("Anthropic: формат %s не поддерживается API, перекодирую в PNG", mime)
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _convert_to_png, b64), "image/png"


def _text_block(text: str) -> dict:
    return {"type": "text", "text": text}


async def _image_block(b64: str, mime: str) -> dict:
    b64, mime = await _ensure_supported_image(b64, mime)
    return {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}}


async def _image_block_from_data_url(img: str, default_mime: str = "image/png") -> dict:
    mime, b64 = split_data_url(img, default_mime)
    return await _image_block(b64, mime)


class AnthropicProvider(Provider):
    name = "anthropic"
    label = "Anthropic Claude"
    # Облачный бэкенд: картинки покидают контур, поэтому на него никогда не уходим
    # автоматически (он не является fallback у других провайдеров), а при его недоступности
    # автофолбэка на локальные бэкенды нет — выбор бэкенда остаётся за вызывающей стороной.
    fallback = None
    # Сэмплинг (temperature/top_p/top_k/seed) не передаётся — см. докстринг модуля.
    sampling_keys = ("num_predict", "think")

    # Ключ хранит клиент (vision_app, в БД, зашифрованным) и присылает его в каждом запросе
    # заголовком X-Api-Key-Anthropic. Здесь он нигде не сохраняется, в том числе в памяти.
    credential = Credential(
        label="API-ключ Anthropic",
        hint="Хранится в БД веб-приложения в зашифрованном виде и передаётся серверу анализа "
             "в каждом запросе; сам сервер анализа ключ не сохраняет.",
    )

    def __init__(self) -> None:
        super().__init__()
        self._session: aiohttp.ClientSession | None = None
        self._session_loop: asyncio.AbstractEventLoop | None = None

    def _get_session(self) -> aiohttp.ClientSession:
        """Одна сессия на все запросы к Anthropic: keep-alive избавляет от нового TLS-рукопожатия
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
        return config.ANTHROPIC_API_BASE

    def _headers(self) -> dict:
        key = self.api_key
        if not key:
            raise RuntimeError(config._t("error.anthropic_no_key"))
        return {"x-api-key": key, "anthropic-version": config.ANTHROPIC_VERSION}

    @staticmethod
    async def _check(resp: aiohttp.ClientResponse) -> None:
        """4xx → RuntimeError с текстом ошибки Anthropic (неверный ключ, квота, модель
        недоступна, картинка слишком большая...) — обработчики превращают его в понятный ответ
        клиенту; 5xx (в т.ч. 529 overloaded) → обычный ClientResponseError (ошибка апстрима)."""
        if resp.status < 400:
            return
        if resp.status < 500:
            try:
                message = ((await resp.json()).get("error") or {}).get("message")
            except Exception:
                message = None
            text = config._t(
                "error.anthropic_api", status=resp.status, message=message or resp.reason or "",
            )
            if resp.status == 404:
                text += " " + config._t("error.anthropic_model_unavailable_hint")
            raise RuntimeError(text)
        resp.raise_for_status()

    # --- модели ------------------------------------------------------------

    async def list_models(self) -> list[str]:
        """Модели Claude, принимающие картинки, из Models API (GET /v1/models, самые новые —
        первыми). Список — это всё, что API отдаёт по ключу."""
        names: list[str] = []
        params: dict = {"limit": 1000}
        headers = self._headers()  # сначала ключ: без него сессию не создаём
        session = self._get_session()
        while True:
            # Таймаут — на каждую страницу отдельно.
            async with session.get(
                f"{config.ANTHROPIC_API_BASE}/models", params=params, headers=headers,
                timeout=config.DISCOVERY_TIMEOUT,
            ) as resp:
                await self._check(resp)
                data = await resp.json()
            for m in data.get("data", []):
                model_id = m.get("id") or ""
                image_input = ((m.get("capabilities") or {}).get("image_input") or {})
                # capabilities может отсутствовать (null) — тогда не отсекаем модель.
                if model_id.startswith("claude-") and image_input.get("supported", True):
                    names.append(model_id)
            last_id = data.get("last_id")
            if not data.get("has_more") or not last_id:
                break
            params = {"limit": 1000, "after_id": last_id}
        return names

    async def discover_model(self) -> str:
        # «Первая в списке» для Anthropic бессмысленна (там и старые модели, и разные
        # тарифные уровни) — без явного model=... берём модель из конфига.
        return config.ANTHROPIC_MODEL

    async def _probe(self) -> str:
        # Для /health: ОДИН дешёвый запрос — метаданные выбранной модели (GET /models/<model>).
        # Он проверяет и ключ, и связь, и что модель вообще существует. Генерацию не запускаем
        # (токены на каждый опрос статуса).
        headers = self._headers()
        async with self._get_session().get(
            f"{config.ANTHROPIC_API_BASE}/models/{quote(config.ANTHROPIC_MODEL, safe='')}",
            headers=headers, timeout=config.DISCOVERY_TIMEOUT,
        ) as resp:
            await self._check(resp)
        return config.ANTHROPIC_MODEL

    async def ping(self) -> dict:
        if not self.is_configured():
            return {"ok": False, "endpoint": self.endpoint, "error": config._t("error.anthropic_no_key")}
        return await super().ping()

    # --- запрос / разбор ответа -------------------------------------------

    async def _generate(
        self, model: str, system: str | None, messages: list[dict], json_mode: bool,
    ) -> str:
        s = config.SAMPLING_DEFAULTS
        body: dict = {
            "model": model,
            "max_tokens": s["num_predict"] or config.ANTHROPIC_MAX_TOKENS,
            "messages": messages,
        }
        system_text = system or ""
        if json_mode:
            system_text += _JSON_ONLY_SUFFIX
        if system_text.strip():
            body["system"] = system_text.strip()
        effort = _effort(s["think"])
        if effort:
            body["output_config"] = {"effort": effort}

        headers = self._headers()
        async with self._get_session().post(
            f"{config.ANTHROPIC_API_BASE}/messages", json=body, headers=headers,
            timeout=config.REQUEST_TIMEOUT,
        ) as resp:
            await self._check(resp)
            data = await resp.json()

        text = self._extract_text(data, model)
        return _extract_json(text) if json_mode else text

    @staticmethod
    def _extract_text(data: dict, model: str) -> str:
        stop_reason = data.get("stop_reason")
        # Нужны только текстовые блоки ответа; thinking и пр. пропускаем.
        text = "".join(
            block.get("text") or ""
            for block in data.get("content") or []
            if block.get("type") == "text"
        ).strip()

        usage = data.get("usage") or {}
        logging.info(
            "Anthropic(%s): input_tokens=%s output_tokens=%s stop_reason=%s content_chars=%d",
            model, usage.get("input_tokens"), usage.get("output_tokens"), stop_reason, len(text),
        )

        if stop_reason == "refusal":
            raise RuntimeError(config._t("error.anthropic_refused", reason=stop_reason))
        if not text:
            if stop_reason == "max_tokens":
                raise RuntimeError(config._t("error.anthropic_truncated"))
            raise RuntimeError(config._t("error.anthropic_empty"))
        if stop_reason == "max_tokens":
            logging.warning(
                "Anthropic(%s): ответ обрезан по max_tokens (num_predict) — токены размышлений "
                "тоже входят в этот лимит; увеличьте num_predict или уменьшите think (/sampling)", model,
            )
        return text

    # --- /analyze ----------------------------------------------------------

    async def analyze(
        self, image_b64: str, image_mime: str, model: str, system_prompt: str, user_prompt: str,
    ) -> str:
        # Картинка перед текстом — порядок из рекомендаций Anthropic для одного изображения.
        content = [
            await _image_block(image_b64, image_mime),
            _text_block(user_prompt),
        ]
        return await self._generate(
            model, system_prompt, [{"role": "user", "content": content}], json_mode=True,
        )

    # --- /chat -------------------------------------------------------------

    @staticmethod
    async def _history_to_messages(history: list) -> list[dict]:
        """История диалога → messages (stateless): user / assistant. Картинки — только во
        входе пользователя; пустые реплики пропускаем (пустой text-блок API не принимает)."""
        messages: list[dict] = []
        for turn in history:
            is_model = turn["role"] == "assistant"
            blocks: list[dict] = []
            if not is_model:
                blocks.extend([await _image_block_from_data_url(img) for img in turn.get("images") or []])
            if (turn.get("content") or "").strip():
                blocks.append(_text_block(turn["content"]))
            if blocks:
                messages.append({"role": "assistant" if is_model else "user", "content": blocks})
        return messages

    async def chat(
        self, model: str, system: str | None, history: list, message: str, images: list[str],
    ) -> str:
        messages = await self._history_to_messages(history)
        blocks = [await _image_block_from_data_url(img) for img in images]
        if message and message.strip():
            blocks.append(_text_block(message))
        messages.append({"role": "user", "content": blocks})
        return await self._generate(model, system, messages, json_mode=False)
