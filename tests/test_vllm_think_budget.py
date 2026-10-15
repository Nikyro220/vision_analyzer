"""vLLM-провайдер: бюджет рассуждений (настройка think_budget) и max_tokens попадают в запрос.

Запуск из корня репозитория:  python -m pytest tests/test_vllm_think_budget.py -q
Реальный vLLM не нужен — используется мок-сервер /v1/chat/completions.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inference"))

import config  # noqa: E402
import providers  # noqa: E402


class MockVllm:
    def __init__(self) -> None:
        self.bodies: list[dict] = []

    async def completions(self, request: web.Request) -> web.Response:
        self.bodies.append(await request.json())
        return web.json_response({"choices": [{"message": {"content": " {\"ok\": true} "}}]})


@pytest.fixture
def vllm(monkeypatch):
    mock = MockVllm()

    async def start():
        app = web.Application()
        app.router.add_post("/v1/chat/completions", mock.completions)
        server = TestServer(app)
        await server.start_server()
        return server

    loop = asyncio.new_event_loop()
    server = loop.run_until_complete(start())
    monkeypatch.setattr(config, "VLLM_URL", f"http://127.0.0.1:{server.port}/v1")
    providers.get("vllm").reset_settings()
    yield mock, loop
    providers.get("vllm").reset_settings()
    loop.run_until_complete(server.close())
    loop.close()


def _run(loop, coro):
    return loop.run_until_complete(coro)


def test_budget_sent_in_analyze_and_chat(vllm, monkeypatch):
    mock, loop = vllm
    p = providers.get("vllm")
    p.update_settings({"think_budget": 1234})

    _run(loop, p.analyze("AAAA", "image/png", "m", "sys", "user"))
    _run(loop, p.chat("m", None, [], "hi", []))

    assert [b.get("thinking_token_budget") for b in mock.bodies] == [1234, 1234]
    assert all("max_tokens" not in b for b in mock.bodies)


def test_budget_zero_means_no_limit(vllm, monkeypatch):
    mock, loop = vllm
    providers.get("vllm").update_settings({"think_budget": 0})
    _run(loop, providers.get("vllm").chat("m", None, [], "hi", []))
    assert "thinking_token_budget" not in mock.bodies[0]


def test_num_predict_becomes_max_tokens(vllm, monkeypatch):
    mock, loop = vllm
    providers.get("vllm").update_settings({"think_budget": 0, "num_predict": 777})
    _run(loop, providers.get("vllm").chat("m", None, [], "hi", []))
    assert mock.bodies[0]["max_tokens"] == 777
