"""
providers — бэкенды модели (vLLM, Ollama, Gemini, Anthropic), по одному модулю на каждый.

Общий интерфейс — providers.base.Provider. Чтобы добавить новый бэкенд:
  1. создать providers/<name>.py с классом-наследником Provider;
  2. добавить его в _PROVIDERS ниже;
  3. добавить настройки (URL/ключ/модель) в config.py.
Больше нигде имена бэкендов перечислять не нужно: валидация ?backend=,
/health, /models, фолбэк и GET /providers берут список отсюда.

Веб-панель (vision_app) тоже ничего не зашивает: карточку провайдера на странице
статуса, форму параметров генерации и поля настроек (ключ API и т.п.) она строит по
GET /providers. Для этого у класса провайдера задаются:
  label           — название для UI;
  sampling_keys   — какие параметры /sampling он использует;
  credential      — секрет (API-ключ), который клиент присылает в КАЖДОМ запросе заголовком
                    X-Api-Key-<Имя>; сервер ключ нигде не хранит (см. providers/base.py).
"""

from __future__ import annotations

import config
from .base import (
    ALL_SAMPLING_KEYS, CREDENTIAL_HEADER_PREFIX, Credential, Provider,
    ensure_data_url, request_credentials, strip_data_url,
)
from .anthropic import AnthropicProvider
from .gemini import GeminiProvider
from .ollama import OllamaProvider
from .vllm import VllmProvider

_PROVIDERS: dict[str, Provider] = {
    p.name: p for p in (VllmProvider(), OllamaProvider(), GeminiProvider(), AnthropicProvider())
}


def names() -> tuple[str, ...]:
    """Имена всех зарегистрированных провайдеров (в порядке регистрации)."""
    return tuple(_PROVIDERS)


def is_known(name: str) -> bool:
    return name in _PROVIDERS


def get(name: str) -> Provider:
    """Провайдер по имени; для неизвестного — ValueError с локализованным текстом
    (обработчики превращают его в HTTP 400)."""
    try:
        return _PROVIDERS[name]
    except KeyError:
        raise ValueError(config._t("error.unknown_backend", backend=name)) from None


def all_providers() -> list[Provider]:
    return list(_PROVIDERS.values())


def reset_all_caches() -> None:
    """Сбрасывает кэши всех провайдеров — после смены адресов/модели через /config."""
    for p in _PROVIDERS.values():
        p.reset_caches()


__all__ = [
    "Provider", "Credential", "ALL_SAMPLING_KEYS", "CREDENTIAL_HEADER_PREFIX", "request_credentials",
    "AnthropicProvider", "GeminiProvider", "OllamaProvider", "VllmProvider",
    "names", "is_known", "get", "all_providers", "reset_all_caches",
    "strip_data_url", "ensure_data_url",
]
