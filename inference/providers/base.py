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
from dataclasses import dataclass

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
# Описание настраиваемых «на лету» полей провайдера
# ---------------------------------------------------------------------------

#: все параметры генерации, которыми управляет POST /sampling
ALL_SAMPLING_KEYS = ("temperature", "top_p", "top_k", "seed", "num_ctx", "num_predict", "think")


@dataclass(frozen=True)
class SettingField:
    """Поле, которое можно менять у провайдера через POST /providers/<name>/settings
    (и которое клиент — например, веб-панель — рисует сам по этому описанию).

    name        — ключ в API (например, "api_key")
    label       — подпись для человека
    config_attr — атрибут модуля config, в который пишется значение (читается
                  провайдером в момент вызова — см. примечание про конфигурацию выше)
    secret      — секрет: значение НИКОГДА не отдаётся наружу, клиенту видно лишь
                  факт, что оно задано (describe() -> "set")
    hint        — короткая подсказка под полем
    """

    name: str
    label: str
    config_attr: str
    secret: bool = False
    hint: str = ""


# ---------------------------------------------------------------------------
# Интерфейс провайдера
# ---------------------------------------------------------------------------

class Provider(ABC):
    #: короткое имя бэкенда — то, что клиент передаёт в ?backend=... и /config
    name: str = ""
    #: человекочитаемое название для UI (пусто — клиент покажет name)
    label: str = ""
    #: какие параметры /sampling этот бэкенд реально использует; клиент строит
    #: форму параметров генерации только из них (num_ctx, например, есть лишь у Ollama)
    sampling_keys: tuple[str, ...] = tuple(k for k in ALL_SAMPLING_KEYS if k != "num_ctx")
    #: поля, которые клиент может менять через /providers/<name>/settings
    settings_fields: tuple[SettingField, ...] = ()
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

    def describe(self) -> dict:
        """Описание провайдера для GET /providers: по нему клиент строит свой UI,
        не зная ничего о конкретных бэкендах. Значения секретных полей не отдаются."""
        fields = []
        for f in self.settings_fields:
            current = getattr(config, f.config_attr, None)
            item = {"name": f.name, "label": f.label, "secret": f.secret, "hint": f.hint, "set": bool(current)}
            if not f.secret:
                item["value"] = "" if current is None else str(current)
            fields.append(item)
        return {
            "name": self.name,
            "label": self.label or self.name,
            "fallback": self.fallback,
            "configured": self.is_configured(),
            "endpoint": self.endpoint,
            "sampling_keys": list(self.sampling_keys),
            "settings": fields,
        }

    def update_settings(self, values: dict, clear: list | tuple = ()) -> list[str]:
        """Применяет значения полей из settings_fields (пустые строки пропускаются —
        «не менять»; имена из clear сбрасываются в пустое значение). Возвращает имена
        изменённых полей. Неизвестное поле — ValueError (обработчик отдаёт 400)."""
        by_name = {f.name: f for f in self.settings_fields}
        unknown = [n for n in (*values, *clear) if n not in by_name]
        if unknown:
            raise ValueError(config._t("error.unknown_setting", backend=self.name, field=", ".join(map(str, unknown))))

        changed: list[str] = []
        for name, raw in values.items():
            if not isinstance(raw, str) or not raw.strip():
                continue
            setattr(config, by_name[name].config_attr, raw.strip())
            changed.append(name)
        for name in clear:
            setattr(config, by_name[name].config_attr, "")
            changed.append(name)
        if changed:
            self.reset_caches()
        return changed

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
