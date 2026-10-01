"""
providers/base.py — общий интерфейс провайдера модели и утилиты data-url.

Провайдер — это всё, что умеет говорить с конкретным бэкендом модели
(vLLM, Ollama, Gemini, ...): список моделей, автоопределение модели,
health-пинг, разовый запрос с картинкой для /analyze (analyze) и
диалоговый запрос с историей для /chat (chat). Конвейер /analyze
(двухпроходный анализ, фолбэк между бэкендами) живёт в backends.py и
chat_backends.py и работает с провайдерами только через этот интерфейс —
поэтому новый бэкенд добавляется одним файлом в providers/ плюс строкой
в providers/__init__.py.

ВАЖНО про конфигурацию: провайдеры всегда читают значения как
`config.X` в момент вызова (а не копируют при импорте), потому что
POST /config меняет их "на лету" (см. config.py).
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod

import aiohttp

import config


# ---------------------------------------------------------------------------
# Утилиты data-url (общие для /analyze и /chat)
# ---------------------------------------------------------------------------

def strip_data_url(img: str) -> str:
    """Убирает 'data:...;base64,' префикс, если есть — Ollama ждёт чистый base64."""
    if img.startswith("data:") and ";base64," in img:
        return img.split(";base64,", 1)[1]
    return img


def ensure_data_url(img: str, default_mime: str = "image/png") -> str:
    """Добавляет 'data:...;base64,' префикс, если его нет — нужен для vLLM image_url."""
    if img.startswith("data:"):
        return img
    return f"data:{default_mime};base64,{img}"


def split_data_url(img: str, default_mime: str = "image/png") -> tuple[str, str]:
    """('data:image/png;base64,AAA' | 'AAA') -> (mime, чистый_base64)."""
    if img.startswith("data:") and ";base64," in img:
        header, b64 = img.split(";base64,", 1)
        return (header[len("data:"):] or default_mime), b64
    return default_mime, img


def sampling_options(*, with_top_k: bool = True, with_ctx: bool = True) -> dict:
    """Ollama-style options из config.SAMPLING_DEFAULTS (читается в момент вызова)."""
    s = config.SAMPLING_DEFAULTS
    options = {"temperature": s["temperature"], "top_p": s["top_p"], "seed": s["seed"]}
    if with_top_k:
        options["top_k"] = s["top_k"]
    if with_ctx:
        if s["num_ctx"] is not None:
            options["num_ctx"] = s["num_ctx"]
        if s["num_predict"] is not None:
            options["num_predict"] = s["num_predict"]
    return options


# ---------------------------------------------------------------------------
# Интерфейс провайдера
# ---------------------------------------------------------------------------

class Provider(ABC):
    #: короткое имя бэкенда — то, что клиент передаёт в ?backend=... и /config
    name: str = ""
    #: на какого провайдера уходить, если этот недоступен по подключению и
    #: клиент не просил его явно (None — без автофолбэка)
    fallback: str | None = None

    def __init__(self) -> None:
        self._model_cache: str | None = None

    # --- конфигурация -----------------------------------------------------

    @property
    @abstractmethod
    def endpoint(self) -> str:
        """Адрес бэкенда — для сообщений об ошибках и /health, /models."""

    def is_configured(self) -> bool:
        """False, если провайдеру не хватает обязательных настроек (например, ключа API)."""
        return True

    def reset_caches(self) -> None:
        """Сбрасывает кэши (автоопределённая модель и т.п.) — вызывается из /config."""
        self._model_cache = None

    # --- модели -----------------------------------------------------------

    @abstractmethod
    async def list_models(self) -> list[str]:
        """Все модели, которые сейчас отдаёт бэкенд (сырое сканирование, без кэша)."""

    async def discover_model(self) -> str:
        """Первая доступная модель бэкенда; результат кэшируется. Используется,
        только когда клиент не указал модель явно."""
        if self._model_cache:
            return self._model_cache

        models = await self.list_models()
        if not models:
            raise RuntimeError(config._t("error.no_models_returned", backend=self.name))

        self._model_cache = models[0]
        logging.info("Автоопределена модель для backend=%s: %s", self.name, models[0])
        return models[0]

    async def context_window(self, model: str | None = None) -> int | None:
        """Реальный размер контекста модели, если бэкенд его отдаёт; иначе None."""
        return None

    # --- health -----------------------------------------------------------

    async def ping(self) -> dict:
        """Статус для /health. Никогда не бросает исключение наружу — любая
        ошибка превращается в {"ok": False, "error": ...}, чтобы падение одного
        бэкенда не мешало проверить остальные."""
        try:
            model = await self._probe()
            return {"ok": True, "endpoint": self.endpoint, "model": model}
        except aiohttp.ClientConnectorError:
            return {"ok": False, "endpoint": self.endpoint, "error": config._t("error.backend_conn_refused")}
        except asyncio.TimeoutError:
            return {"ok": False, "endpoint": self.endpoint, "error": config._t("error.backend_timeout")}
        except Exception as e:
            return {"ok": False, "endpoint": self.endpoint, "error": str(e)}

    async def _probe(self) -> str:
        """Проверка доступности для ping(); возвращает имя модели."""
        return await self.discover_model()

    # --- запросы к модели -------------------------------------------------

    @abstractmethod
    async def analyze(
        self, image_b64: str, image_mime: str, model: str, system_prompt: str, user_prompt: str,
    ) -> str:
        """Один запрос system+user+картинка с требованием JSON-ответа (для /analyze).
        Возвращает сырой текст модели — схему ответа разбирает вызывающая сторона."""

    @abstractmethod
    async def chat(
        self, model: str, system: str | None, history: list, message: str, images: list[str],
    ) -> str:
        """Одна реплика свободного диалога с историей, БЕЗ принудительного JSON (для /chat)."""
