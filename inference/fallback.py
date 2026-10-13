"""
fallback.py — цепочка фолбэка между бэкендами для /chat и /analyze.

Правило простое: любая ошибка бэкенда (подключение, таймаут, 4xx/5xx, safety-блок,
пустой/сломанный ответ провайдера) → пробуем следующий бэкенд. Какая именно это ошибка —
пользователю не важно.

  - порядок: сам бэкенд → его Provider.fallback → остальные локальные → облачные.
    Облачные участвуют, только если в ЭТОМ запросе пришёл их ключ (Provider.is_configured)
    и не выключено VISION_ANALYZER_CLOUD_FALLBACK=0 — иначе картинки уходили бы во вне;
  - на запасном бэкенде модель всегда автоопределяется (model=None): имя модели с одного
    провайдера на другом бессмысленно;
  - если упали все, наружу уходит ПЕРВАЯ ошибка (исходно запрошенного бэкенда).
"""

from __future__ import annotations

import logging
from typing import Awaitable, Callable, TypeVar

import config
import providers

T = TypeVar("T")


def parse_flag(raw, default: bool) -> bool:
    """Булев параметр запроса (?fallback=1 / JSON true); пусто — default."""
    if raw is None or raw == "":
        return default
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in ("1", "true", "yes", "on")


def chain(backend: str) -> list[str]:
    """Кого пробовать ПОСЛЕ backend, в порядке приоритета."""
    primary = providers.get(backend)
    others = [p for p in providers.all_providers() if p.name != backend]
    local = [p for p in others if p.credential is None]
    cloud = [p for p in others if p.credential is not None and config.CLOUD_FALLBACK and p.is_configured()]

    ordered = sorted(local, key=lambda p: p.name != primary.fallback) + cloud
    return [p.name for p in ordered]


async def run(
    backend: str,
    call: Callable[[str, bool], Awaitable[T]],
    allow_fallback: bool,
) -> T:
    """Вызывает call(имя_бэкенда, это_фолбэк) по цепочке, пока не получится.

    Ошибку исходного бэкенда помечает атрибутом .chat_backend (имя бэкенда, на котором
    она случилась) — обработчики используют его для текста ошибки.
    """
    names = [backend]
    if allow_fallback:
        names += chain(backend)

    first: Exception | None = None
    for i, name in enumerate(names):
        try:
            return await call(name, i > 0)
        except Exception as e:  # noqa: BLE001 — любая ошибка = пробуем следующий
            if i == 0:
                e.chat_backend = name
                first = e
                logging.warning("Бэкенд %r не ответил (%s: %s)", name, type(e).__name__, e)
            else:
                logging.warning("Фолбэк %r тоже не сработал (%s: %s)", name, type(e).__name__, e)
            if i + 1 < len(names):
                logging.warning("Пробую следующий бэкенд: %r", names[i + 1])

    assert first is not None
    raise first
