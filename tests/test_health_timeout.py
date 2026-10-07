"""/health не должен ждать недоступные локальные бэкенды дольше HEALTH_PROBE_TIMEOUT
и не должен опрашивать их заново на каждый запрос (кэш).

Запуск из корня репозитория:  python -m pytest tests/test_health_timeout.py -q
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inference"))

import config  # noqa: E402
import server  # noqa: E402


class HangingProvider:
    """Имитация локального бэкенда, у которого TCP-connect «зависает» (пакеты дропаются)."""

    name = "hang"
    credential = None
    endpoint = "http://10.255.255.1:11434"

    def __init__(self) -> None:
        self.calls = 0

    async def ping(self) -> dict:
        self.calls += 1
        await asyncio.sleep(30)
        return {"ok": True}


@pytest.fixture(autouse=True)
def _fast_config(monkeypatch):
    monkeypatch.setattr(config, "HEALTH_PROBE_TIMEOUT", 0.2)
    monkeypatch.setattr(config, "HEALTH_CACHE_FAIL_TTL", 5)
    monkeypatch.setattr(config, "HEALTH_CACHE_OK_TTL", 5)
    server._health_cache.clear()
    yield
    server._health_cache.clear()


def test_hanging_backend_is_bounded_and_cached():
    p = HangingProvider()

    async def run():
        t0 = time.monotonic()
        first = await server._ping_bounded(p)
        t_first = time.monotonic() - t0
        t0 = time.monotonic()
        second = await server._ping_bounded(p)
        t_second = time.monotonic() - t0
        return first, t_first, second, t_second

    first, t_first, second, t_second = asyncio.run(run())

    assert first["ok"] is False and first["endpoint"] == p.endpoint
    assert t_first < 1.0           # не 30 с и не 10 с DISCOVERY_TIMEOUT
    assert second == first and t_second < 0.05
    assert p.calls == 1            # второй вызов обслужен из кэша


def test_cache_is_keyed_by_endpoint():
    p = HangingProvider()

    async def run():
        await server._ping_bounded(p)
        p.endpoint = "http://other:11434"   # как после POST /config
        await server._ping_bounded(p)

    asyncio.run(run())
    assert p.calls == 2


def test_credentialed_provider_is_not_cached():
    p = HangingProvider()
    p.credential = object()
    asyncio.run(server._ping_bounded(p))
    asyncio.run(server._ping_bounded(p))
    assert p.calls == 2
