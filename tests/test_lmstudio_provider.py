"""LM Studio-провайдер: отбор чат-моделей, контекст загруженной модели, json_schema и его откат,
срезание <think>, настройки и регистрация.

Запуск из корня репозитория:  python -m pytest tests/test_lmstudio_provider.py -q
Реальный LM Studio не нужен — используется мок-сервер (/v1/chat/completions, /v1/models, /api/v0/models).
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
import fallback  # noqa: E402

NATIVE_MODELS = {"data": [
    {"id": "text-embedding-nomic-embed-text-v1.5", "type": "embeddings", "state": "not-loaded"},
    {"id": "qwen-text", "type": "llm", "state": "loaded", "loaded_context_length": 8192},
    {"id": "gemma-vision-unloaded", "type": "vlm", "state": "not-loaded", "max_context_length": 131072},
    {"id": "gemma-vision", "type": "vlm", "state": "loaded", "loaded_context_length": 32768},
]}
OPENAI_MODELS = {"data": [{"id": "text-embedding-nomic-embed-text-v1.5"}, {"id": "some-model"}]}


class MockLmStudio:
    def __init__(self) -> None:
        self.bodies: list[dict] = []
        self.native = True            # False -> /api/v0/models отвечает 404 (старый LM Studio)
        self.reject_json_schema = False
        self.content = '{"ok": true}'

    async def completions(self, request: web.Request) -> web.Response:
        body = await request.json()
        self.bodies.append(body)
        if self.reject_json_schema and "response_format" in body:
            return web.json_response({"error": "response_format unsupported"}, status=400)
        return web.json_response({"choices": [{"message": {"content": self.content}}]})

    async def v1_models(self, request: web.Request) -> web.Response:
        return web.json_response(OPENAI_MODELS)

    async def native_models(self, request: web.Request) -> web.Response:
        if not self.native:
            return web.json_response({"error": "not found"}, status=404)
        return web.json_response(NATIVE_MODELS)


@pytest.fixture
def lm(monkeypatch):
    mock = MockLmStudio()

    async def start():
        app = web.Application()
        app.router.add_post("/v1/chat/completions", mock.completions)
        app.router.add_get("/v1/models", mock.v1_models)
        app.router.add_get("/api/v0/models", mock.native_models)
        server = TestServer(app)
        await server.start_server()
        return server

    loop = asyncio.new_event_loop()
    server = loop.run_until_complete(start())
    monkeypatch.setattr(config, "LMSTUDIO_URL", f"http://127.0.0.1:{server.port}/v1")
    provider = providers.get("lmstudio")
    provider.reset_settings()
    provider.reset_caches()
    yield mock, loop, provider
    provider.reset_settings()
    provider.reset_caches()
    loop.run_until_complete(server.close())
    loop.close()


# --- регистрация ----------------------------------------------------------------

def test_registered_with_own_settings():
    assert providers.is_known("lmstudio")
    p = providers.get("lmstudio")
    assert p.credential is None and p.is_configured()
    assert tuple(p.settings) == ("temperature", "top_p", "top_k", "seed", "num_predict")
    assert p.describe()["label"] == "LM Studio"


def test_fallback_chain_includes_lmstudio_as_local():
    # Первым идёт «родной» запасной бэкенд, затем остальные локальные (в т.ч. lmstudio).
    assert fallback.chain("vllm")[:2] == ["ollama", "lmstudio"]
    assert fallback.chain("ollama")[:2] == ["vllm", "lmstudio"]
    assert fallback.chain("lmstudio")[:2] == ["ollama", "vllm"]


# --- модели ---------------------------------------------------------------------

def test_list_models_skips_embeddings_and_puts_loaded_vision_first(lm):
    _, loop, p = lm
    models = loop.run_until_complete(p.list_models())
    assert models == ["gemma-vision", "qwen-text", "gemma-vision-unloaded"]
    assert loop.run_until_complete(p.discover_model()) == "gemma-vision"


def test_list_models_falls_back_to_openai_endpoint(lm):
    mock, loop, p = lm
    mock.native = False
    assert loop.run_until_complete(p.list_models()) == ["some-model"]


def test_context_window_is_loaded_context_only(lm):
    mock, loop, p = lm
    assert loop.run_until_complete(p.context_window("gemma-vision")) == 32768
    assert loop.run_until_complete(p.context_window("gemma-vision-unloaded")) is None  # не загружена
    assert loop.run_until_complete(p.context_window("no-such-model")) is None
    mock.native = False
    assert loop.run_until_complete(p.context_window("gemma-vision")) is None


def test_ping_reports_model(lm):
    _, loop, p = lm
    st = loop.run_until_complete(p.ping())
    assert st["ok"] and st["model"] == "gemma-vision"


def test_ping_connection_refused(monkeypatch):
    monkeypatch.setattr(config, "LMSTUDIO_URL", "http://127.0.0.1:1/v1")
    st = asyncio.run(providers.get("lmstudio").ping())
    assert st["ok"] is False and st["endpoint"] == "http://127.0.0.1:1/v1"


# --- запросы --------------------------------------------------------------------

def test_analyze_requests_json_schema_not_json_object(lm):
    mock, loop, p = lm
    out = loop.run_until_complete(p.analyze("AAAA", "image/png", "m", "sys", "user"))
    assert out == '{"ok": true}'
    rf = mock.bodies[0]["response_format"]
    assert rf["type"] == "json_schema" and rf["json_schema"]["schema"] == {"type": "object"}
    img = mock.bodies[0]["messages"][1]["content"][1]["image_url"]["url"]
    assert img == "data:image/png;base64,AAAA"


def test_analyze_retries_without_response_format_on_400(lm):
    mock, loop, p = lm
    mock.reject_json_schema = True
    out = loop.run_until_complete(p.analyze("AAAA", "image/png", "m", "sys", "user"))
    assert out == '{"ok": true}'
    assert len(mock.bodies) == 2
    assert "response_format" in mock.bodies[0] and "response_format" not in mock.bodies[1]


def test_chat_has_no_response_format(lm):
    mock, loop, p = lm
    loop.run_until_complete(p.chat("m", "sys", [], "hi", []))
    assert "response_format" not in mock.bodies[0]
    assert mock.bodies[0]["messages"][0] == {"role": "system", "content": "sys"}


def test_think_block_is_stripped(lm):
    mock, loop, p = lm
    mock.content = '<think>\nхм, думаю {не json}\n</think>\n{"ok": true}'
    assert loop.run_until_complete(p.analyze("AAAA", "image/png", "m", "s", "u")) == '{"ok": true}'


def test_empty_content_does_not_crash(lm):
    mock, loop, p = lm
    mock.content = None
    assert loop.run_until_complete(p.chat("m", None, [], "hi", [])) == ""


def test_chat_truncates_history_to_loaded_context(lm):
    mock, loop, p = lm
    # gemma-vision загружена с контекстом 32768 (~29500 токенов бюджета): старые сообщения
    # по ~20000 токенов каждое не влезают все вместе — должно остаться не больше одного.
    big = "x" * 80_000  # ~20000 оценочных токенов
    history = [{"role": "user", "content": big}, {"role": "assistant", "content": big}]
    loop.run_until_complete(p.chat("gemma-vision", None, history, "hi", []))
    sent = mock.bodies[0]["messages"]
    assert len(sent) == 2  # одно сообщение истории + текущее


# --- настройки ------------------------------------------------------------------

def test_generation_params_in_request(lm):
    mock, loop, p = lm
    p.update_settings({"temperature": 0.3, "top_k": 40, "seed": "auto", "num_predict": 512})
    loop.run_until_complete(p.chat("m", None, [], "hi", []))
    body = mock.bodies[0]
    assert (body["temperature"], body["top_p"], body["top_k"], body["max_tokens"]) == (0.3, 1.0, 40, 512)
    assert "seed" not in body


def test_default_request_omits_top_k_and_max_tokens(lm):
    mock, loop, p = lm
    loop.run_until_complete(p.chat("m", None, [], "hi", []))
    body = mock.bodies[0]
    assert "top_k" not in body and "max_tokens" not in body and body["seed"] == 42


def test_url_trailing_slash_and_native_url(monkeypatch):
    monkeypatch.setattr(config, "LMSTUDIO_URL", "http://host:1234/v1")
    assert providers.get("lmstudio")._native_models_url() == "http://host:1234/api/v0/models"
    monkeypatch.setattr(config, "LMSTUDIO_URL", "http://host:1234")
    assert providers.get("lmstudio")._native_models_url() == "http://host:1234/api/v0/models"
