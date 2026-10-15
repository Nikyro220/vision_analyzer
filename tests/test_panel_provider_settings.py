"""Панель: форма параметров строится по описанию ЭТОГО провайдера (GET /providers → settings);
сохранение и сброс затрагивают только его.

Сквозной тест: настоящий сервер анализа (aiohttp, в отдельном потоке) + мок-бэкенды vLLM и Ollama,
чтобы оба были «доступны», + настоящая панель (Flask test client).

Запуск из корня репозитория:  python -m pytest tests/test_panel_provider_settings.py -q
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
from pathlib import Path

import pytest
from aiohttp import web
from cryptography.fernet import Fernet

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inference"))

import config  # noqa: E402
import providers  # noqa: E402
import server  # noqa: E402

from vision_app import create_app, services  # noqa: E402
from vision_app.extensions import db  # noqa: E402
from vision_app.models import Role, User  # noqa: E402

PASSWORD = "Sup3r-secret-pass"


class _Stack:
    """Сервер анализа + мок vLLM + мок Ollama в одном фоновом event loop."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.ports: dict[str, int] = {}
        self._runners: list[web.AppRunner] = []
        started = threading.Event()
        self.thread = threading.Thread(target=self._run, args=(started,), daemon=True)
        self.thread.start()
        assert started.wait(10)

    def _run(self, started: threading.Event) -> None:
        asyncio.set_event_loop(self.loop)

        async def vllm_models(_):
            return web.json_response({"data": [{"id": "mock-vllm", "max_model_len": 8192}]})

        async def ollama_tags(_):
            return web.json_response({"models": [{"name": "mock-ollama"}]})

        async def start():
            vllm = web.Application()
            vllm.router.add_get("/v1/models", vllm_models)
            ollama = web.Application()
            ollama.router.add_get("/api/tags", ollama_tags)
            for name, app in (("vllm", vllm), ("ollama", ollama), ("server", server.build_app())):
                runner = web.AppRunner(app)
                await runner.setup()
                site = web.TCPSite(runner, "127.0.0.1", 0)
                await site.start()
                self.ports[name] = runner.addresses[0][1]
                self._runners.append(runner)

        self.loop.run_until_complete(start())
        started.set()
        self.loop.run_forever()

    def stop(self) -> None:
        async def close():
            for r in self._runners:
                await r.cleanup()
        asyncio.run_coroutine_threadsafe(close(), self.loop).result(10)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(10)


@pytest.fixture()
def client(tmp_path, monkeypatch):
    stack = _Stack()
    monkeypatch.setattr(config, "VLLM_URL", f"http://127.0.0.1:{stack.ports['vllm']}/v1")
    monkeypatch.setattr(config, "OLLAMA_HOST", f"http://127.0.0.1:{stack.ports['ollama']}")
    server._health_cache.clear()
    for p in providers.all_providers():
        p.reset_settings()

    os.environ["VISION_CREDENTIALS_KEY"] = Fernet.generate_key().decode()
    app = create_app({
        "TESTING": True, "WTF_CSRF_ENABLED": False, "SECRET_KEY": "test",
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path}/t.db", "UPLOAD_FOLDER": str(tmp_path / "up"),
    })
    monkeypatch.setattr(services, "_base_url", lambda: f"http://127.0.0.1:{stack.ports['server']}")
    services.invalidate_health()
    services.invalidate_providers()
    with app.app_context():
        db.create_all()
        u = User(username="boss", role=Role.ADMIN)
        u.set_password(PASSWORD)
        db.session.add(u)
        db.session.commit()
    c = app.test_client()
    c.post("/accounts/login/", data={"username": "boss", "password": PASSWORD})
    yield c
    stack.stop()
    for p in providers.all_providers():
        p.reset_settings()


def _page(c) -> str:
    return c.get("/health/?refresh=1").get_data(as_text=True)


def test_each_card_has_its_own_fields(client):
    html = _page(client)
    # у vLLM свой бюджет рассуждений и нет num_ctx/think; у Ollama наоборот
    assert 'name="vllm-think_budget"' in html and 'name="vllm-num_ctx"' not in html
    assert 'name="vllm-think"' not in html
    assert 'name="ollama-num_ctx"' in html and 'name="ollama-think"' in html
    assert 'name="ollama-think_budget"' not in html


def test_save_and_reset_affect_only_that_provider(client):
    _page(client)  # прогрев кэша провайдеров
    r = client.post("/health/backend/vllm/sampling", data={
        "vllm-temperature": "0.4", "vllm-think_budget": "1000", "vllm-top_k": "auto",
    })
    assert r.status_code == 302
    assert providers.get("vllm").settings["temperature"] == 0.4
    assert providers.get("vllm").settings["think_budget"] == 1000
    assert providers.get("ollama").settings["temperature"] == 0.0

    # форма показывает уже сохранённые значения
    html = _page(client)
    assert 'name="vllm-temperature"' in html and 'value="0.4"' in html and 'value="1000"' in html

    r = client.post("/health/backend/ollama/sampling", data={"ollama-think": "high", "ollama-num_ctx": "4096"})
    assert r.status_code == 302
    assert providers.get("ollama").settings["think"] == "high"
    assert providers.get("ollama").settings["num_ctx"] == 4096
    assert providers.get("vllm").settings["temperature"] == 0.4  # не тронут

    r = client.post("/health/backend/vllm/sampling/reset")
    assert r.status_code == 302
    assert providers.get("vllm").settings["temperature"] == 0.0
    assert providers.get("vllm").settings["think_budget"] == 4096
    assert providers.get("ollama").settings["think"] == "high"  # сброс vLLM не трогает Ollama


def test_invalid_value_is_rejected_by_panel_and_changes_nothing(client):
    _page(client)
    r = client.post("/health/backend/vllm/sampling", data={"vllm-top_p": "5", "vllm-temperature": "0.9"})
    assert r.status_code == 400
    assert "Допустимое значение" in r.get_data(as_text=True)
    assert providers.get("vllm").settings["temperature"] == 0.0

    # num_ctx у vLLM нет — такого поля в форме нет, значение просто не принимается
    r = client.post("/health/backend/vllm/sampling", data={"vllm-num_ctx": "4096"})
    assert r.status_code == 302 and "num_ctx" not in providers.get("vllm").settings
