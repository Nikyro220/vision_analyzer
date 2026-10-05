"""Провайдер Anthropic (inference/providers/anthropic.py) против мок-сервера Messages API:
заголовки и тело запроса, разбор ответа, JSON из ```-обёртки, отказы, ошибки API, пагинация
моделей, перекодирование неподдерживаемых форматов картинок.

Запуск из корня репозитория:  python -m pytest tests/test_anthropic_provider.py -q
Реальный API не вызывается и ключ не нужен.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import sys
from pathlib import Path

import aiohttp
import pytest
from aiohttp import web
from aiohttp.test_utils import TestServer
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inference"))

import config  # noqa: E402
import providers  # noqa: E402

KEY = "sk-ant-test"


def _image_b64(fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), (200, 10, 10)).save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


class MockAnthropic:
    """Минимальная имитация Anthropic API; запоминает последний запрос к /messages."""

    def __init__(self) -> None:
        self.last_headers: dict = {}
        self.last_body: dict = {}
        self.models_calls: list[dict] = []
        # что отвечать на /messages: (status, json)
        self.reply: tuple[int, dict] = (200, self.ok("{\"ok\": true}"))

    @staticmethod
    def ok(text: str, stop_reason: str = "end_turn") -> dict:
        return {
            "type": "message", "role": "assistant", "stop_reason": stop_reason,
            "content": [{"type": "thinking", "thinking": "..."}, {"type": "text", "text": text}],
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }

    async def models(self, request: web.Request) -> web.Response:
        self.models_calls.append(dict(request.query))
        self.last_headers = dict(request.headers)
        if request.query.get("after_id") == "claude-b":
            return web.json_response({
                "data": [
                    {"id": "claude-c", "capabilities": None},
                    {"id": "claude-novision", "capabilities": {"image_input": {"supported": False}}},
                ],
                "has_more": False, "last_id": "claude-novision",
            })
        return web.json_response({
            "data": [
                {"id": "claude-a", "capabilities": {"image_input": {"supported": True}}},
                {"id": "claude-b", "capabilities": {"image_input": {"supported": True}}},
                {"id": "not-claude", "capabilities": {"image_input": {"supported": True}}},
            ],
            "has_more": True, "last_id": "claude-b",
        })

    async def model(self, request: web.Request) -> web.Response:
        self.last_headers = dict(request.headers)
        if request.match_info["model_id"] == "missing":
            return web.json_response(
                {"type": "error", "error": {"type": "not_found_error", "message": "model: missing"}},
                status=404,
            )
        return web.json_response({"id": request.match_info["model_id"], "type": "model"})

    async def messages(self, request: web.Request) -> web.Response:
        self.last_headers = dict(request.headers)
        self.last_body = await request.json()
        status, payload = self.reply
        return web.json_response(payload, status=status)


@pytest.fixture()
def mock(monkeypatch):
    m = MockAnthropic()
    app = web.Application()
    app.router.add_get("/v1/models", m.models)
    app.router.add_get("/v1/models/{model_id}", m.model)
    app.router.add_post("/v1/messages", m.messages)
    server = TestServer(app)
    loop = asyncio.new_event_loop()
    loop.run_until_complete(server.start_server())
    monkeypatch.setattr(config, "ANTHROPIC_API_BASE", f"http://{server.host}:{server.port}/v1")
    monkeypatch.setattr(config, "ANTHROPIC_MODEL", "claude-sonnet-5-5")
    m.run = lambda coro_fn, key=KEY: loop.run_until_complete(_with_key(coro_fn, key))
    yield m
    loop.run_until_complete(providers.get("anthropic").aclose())
    loop.run_until_complete(server.close())
    loop.close()


async def _with_key(coro_fn, key: str | None = KEY):
    token = providers.request_credentials.set({"anthropic": key} if key else {})
    try:
        return await coro_fn(providers.get("anthropic"))
    finally:
        providers.request_credentials.reset(token)


def test_registered_and_described():
    p = providers.get("anthropic")
    d = p.describe()
    assert d["credential"]["header"] == "X-Api-Key-Anthropic"
    assert d["sampling_keys"] == ["num_predict", "think"]
    assert d["fallback"] is None  # картинки не уходят в облако автоматически


def test_list_models_paginates_and_filters(mock):
    names = mock.run(lambda p: p.list_models())
    assert names == ["claude-a", "claude-b", "claude-c"]
    assert mock.models_calls == [{"limit": "1000"}, {"limit": "1000", "after_id": "claude-b"}]
    assert mock.last_headers["x-api-key"] == KEY
    assert mock.last_headers["anthropic-version"] == config.ANTHROPIC_VERSION


def test_analyze_request_shape_and_json_cleanup(mock, monkeypatch):
    monkeypatch.setitem(config.SAMPLING_DEFAULTS, "num_predict", None)
    monkeypatch.setitem(config.SAMPLING_DEFAULTS, "think", True)
    mock.reply = (200, mock.ok('Вот отчёт:\n```json\n{"risk_level": "low"}\n```\nГотово.'))

    out = mock.run(lambda p: p.analyze(_image_b64(), "image/png", "claude-sonnet-5-5", "SYS", "USER"))

    assert json.loads(out) == {"risk_level": "low"}
    body = mock.last_body
    assert body["model"] == "claude-sonnet-5-5"
    assert body["max_tokens"] == config.ANTHROPIC_MAX_TOKENS
    assert body["system"].startswith("SYS") and "ONLY a single valid JSON" in body["system"]
    # сэмплинг и prefill на актуальных моделях дают 400 — не передаём
    for forbidden in ("temperature", "top_p", "top_k", "seed", "thinking", "output_config"):
        assert forbidden not in body
    assert body["messages"][-1]["role"] == "user"  # без prefill ассистента
    blocks = body["messages"][0]["content"]
    assert [b["type"] for b in blocks] == ["image", "text"]  # картинка перед текстом
    assert blocks[0]["source"]["media_type"] == "image/png"
    assert blocks[1]["text"] == "USER"


def test_json_extracted_from_surrounding_text(mock):
    mock.reply = (200, mock.ok('Ответ: {"a": 1} — конец'))
    out = mock.run(lambda p: p.analyze(_image_b64(), "image/png", "m", "S", "U"))
    assert json.loads(out) == {"a": 1}


def test_invalid_json_returned_as_is(mock):
    mock.reply = (200, mock.ok("не json"))
    out = mock.run(lambda p: p.analyze(_image_b64(), "image/png", "m", "S", "U"))
    assert out == "не json"  # backends.py сам покажет сырой ответ


@pytest.mark.parametrize("think,expected", [("medium", "medium"), ("high", "high"), (False, "low")])
def test_think_maps_to_effort(mock, monkeypatch, think, expected):
    monkeypatch.setitem(config.SAMPLING_DEFAULTS, "think", think)
    mock.run(lambda p: p.chat("m", None, [], "hi", []))
    assert mock.last_body["output_config"] == {"effort": expected}
    assert "thinking" not in mock.last_body  # thinking={"type": "disabled"} на новых моделях = 400


def test_num_predict_becomes_max_tokens(mock, monkeypatch):
    monkeypatch.setitem(config.SAMPLING_DEFAULTS, "num_predict", 777)
    mock.run(lambda p: p.chat("m", None, [], "hi", []))
    assert mock.last_body["max_tokens"] == 777


def test_chat_history_mapping(mock):
    img = "data:image/png;base64," + _image_b64()
    history = [
        {"role": "user", "content": "привет", "images": [img]},
        {"role": "assistant", "content": "здравствуйте", "images": [img]},  # картинки ассистента не шлём
        {"role": "user", "content": "   "},  # пустая реплика пропускается
    ]
    mock.reply = (200, mock.ok("ответ"))
    out = mock.run(lambda p: p.chat("m", "системный", history, "а это?", [img]))

    assert out == "ответ"
    body = mock.last_body
    assert body["system"] == "системный"  # без JSON-суффикса в /chat
    msgs = body["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert [b["type"] for b in msgs[0]["content"]] == ["image", "text"]
    assert [b["type"] for b in msgs[1]["content"]] == ["text"]
    assert [b["type"] for b in msgs[2]["content"]] == ["image", "text"]


def test_chat_images_only_has_no_empty_text_block(mock):
    img = "data:image/png;base64," + _image_b64()
    mock.run(lambda p: p.chat("m", None, [], "", [img]))
    assert [b["type"] for b in mock.last_body["messages"][0]["content"]] == ["image"]
    assert "system" not in mock.last_body


def test_unsupported_image_format_is_converted_to_png(mock):
    mock.run(lambda p: p.analyze(_image_b64("BMP"), "image/bmp", "m", "S", "U"))
    src = mock.last_body["messages"][0]["content"][0]["source"]
    assert src["media_type"] == "image/png"
    assert Image.open(io.BytesIO(base64.b64decode(src["data"]))).format == "PNG"


def test_jpg_alias_normalized(mock):
    mock.run(lambda p: p.analyze(_image_b64("JPEG"), "image/jpg", "m", "S", "U"))
    assert mock.last_body["messages"][0]["content"][0]["source"]["media_type"] == "image/jpeg"


def test_refusal_becomes_runtime_error(mock):
    mock.reply = (200, {"stop_reason": "refusal", "content": [], "usage": {}})
    with pytest.raises(RuntimeError, match="отказался|refused"):
        mock.run(lambda p: p.analyze(_image_b64(), "image/png", "m", "S", "U"))


def test_empty_and_truncated_answers(mock):
    mock.reply = (200, {"stop_reason": "end_turn", "content": [], "usage": {}})
    with pytest.raises(RuntimeError):
        mock.run(lambda p: p.chat("m", None, [], "hi", []))
    mock.reply = (200, {"stop_reason": "max_tokens", "content": [{"type": "thinking", "thinking": "x"}], "usage": {}})
    with pytest.raises(RuntimeError, match="max_tokens"):
        mock.run(lambda p: p.chat("m", None, [], "hi", []))


def test_4xx_surfaces_api_message_5xx_is_upstream_error(mock):
    mock.reply = (400, {"type": "error", "error": {"type": "invalid_request_error", "message": "image too large"}})
    with pytest.raises(RuntimeError, match="400.*image too large"):
        mock.run(lambda p: p.chat("m", None, [], "hi", []))
    mock.reply = (529, {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}})
    with pytest.raises(aiohttp.ClientResponseError) as exc:
        mock.run(lambda p: p.chat("m", None, [], "hi", []))
    assert exc.value.status == 529


def test_missing_key(mock):
    with pytest.raises(RuntimeError, match="X-Api-Key-Anthropic"):
        mock.run(lambda p: p.chat("m", None, [], "hi", []), key=None)
    ping = mock.run(lambda p: p.ping(), key=None)
    assert ping["ok"] is False and "X-Api-Key-Anthropic" in ping["error"]
    assert mock.last_body == {}  # до сервера дело не дошло


def test_ping_ok_and_model_not_found(mock, monkeypatch):
    ping = mock.run(lambda p: p.ping())
    assert ping == {"ok": True, "endpoint": config.ANTHROPIC_API_BASE, "model": "claude-sonnet-5-5"}

    monkeypatch.setattr(config, "ANTHROPIC_MODEL", "missing")
    ping = mock.run(lambda p: p.ping())
    assert ping["ok"] is False and "404" in ping["error"]
