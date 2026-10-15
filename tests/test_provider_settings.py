"""Настройки провайдеров: у каждого свой набор, значения по умолчанию и допустимые значения;
изменение одного провайдера не затрагивает другие.

Запуск из корня репозитория:  python -m pytest tests/test_provider_settings.py -q
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
from aiohttp.test_utils import TestClient, TestServer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "inference"))

import config  # noqa: E402
import providers  # noqa: E402
import server  # noqa: E402
from providers.settings import Setting, SettingError  # noqa: E402


@pytest.fixture(autouse=True)
def _reset():
    for p in providers.all_providers():
        p.reset_settings()
    yield
    for p in providers.all_providers():
        p.reset_settings()


# --- у каждого провайдера свой набор -----------------------------------------

def test_each_provider_has_its_own_keys():
    keys = {p.name: tuple(p.settings) for p in providers.all_providers()}
    assert "num_ctx" in keys["ollama"] and "think" in keys["ollama"]
    assert "think_budget" in keys["vllm"] and "num_ctx" not in keys["vllm"] and "think" not in keys["vllm"]
    assert keys["anthropic"] == ("num_predict", "think")
    assert "top_p" not in keys["gemini"] and "top_k" not in keys["gemini"]


def test_describe_keeps_sampling_keys_and_adds_schema():
    d = providers.get("anthropic").describe()
    assert d["sampling_keys"] == ["num_predict", "think"]
    think = next(s for s in d["settings"] if s["key"] == "think")
    assert think["type"] == "choice" and think["value"] is True and think["default"] is True
    assert [c["value"] for c in think["choices"]] == ["true", "false", "low", "medium", "high"]


# --- изоляция ------------------------------------------------------------------

def test_changing_one_provider_does_not_touch_others():
    before = {p.name: p.settings for p in providers.all_providers()}
    providers.get("ollama").update_settings({"temperature": 0.7, "num_predict": 512, "think": "high"})
    providers.get("vllm").update_settings({"temperature": 0.3, "think_budget": 100})

    assert providers.get("ollama").settings["temperature"] == 0.7
    assert providers.get("vllm").settings["temperature"] == 0.3
    for name in ("gemini", "anthropic"):
        assert providers.get(name).settings == before[name]
    # одинаковые ключи у разных провайдеров независимы
    assert providers.get("ollama").settings["num_predict"] == 512
    assert providers.get("vllm").settings["num_predict"] is None


# --- свои допустимые значения ---------------------------------------------------

def test_values_are_validated_per_provider():
    with pytest.raises(SettingError) as e:
        providers.get("vllm").update_settings({"num_ctx": 8192})
    assert e.value.kind == "unknown" and "think_budget" in e.value.available

    with pytest.raises(SettingError) as e:
        providers.get("gemini").update_settings({"temperature": 5})
    assert e.value.kind == "invalid"

    with pytest.raises(SettingError):
        providers.get("anthropic").update_settings({"think": "extreme"})
    with pytest.raises(SettingError):
        providers.get("ollama").update_settings({"top_p": 0})  # (0, 1]: ноль не входит


def test_update_is_all_or_nothing():
    p = providers.get("ollama")
    before = p.settings
    with pytest.raises(SettingError):
        p.update_settings({"temperature": 0.9, "top_p": 7})
    assert p.settings == before


def test_parsing_types_and_auto():
    p = providers.get("ollama")
    p.update_settings({"num_predict": "1024", "think": "false", "temperature": "0,5", "seed": "auto"})
    s = p.settings
    assert s["num_predict"] == 1024 and s["think"] is False and s["temperature"] == 0.5 and s["seed"] is None
    p.update_settings({"num_predict": "auto"})
    assert p.settings["num_predict"] is None
    with pytest.raises(SettingError):
        p.update_settings({"top_k": "auto"})  # top_k у Ollama не nullable


def test_reset_only_affects_given_keys_of_that_provider():
    p, other = providers.get("ollama"), providers.get("gemini")
    p.update_settings({"temperature": 1.0, "top_k": 5})
    other.update_settings({"temperature": 1.5})
    p.reset_settings(["temperature"])
    assert p.settings["temperature"] == 0.0 and p.settings["top_k"] == 5
    assert other.settings["temperature"] == 1.5


# --- значения по умолчанию из окружения ----------------------------------------------

def test_env_defaults_provider_specific_beats_legacy(monkeypatch):
    st = Setting("temperature", "float", 0.0, min=0, max=2)
    assert st.initial("vllm") == 0.0
    monkeypatch.setenv("VISION_ANALYZER_TEMPERATURE", "0.4")          # старое общее имя
    assert st.initial("vllm") == 0.4 and st.initial("ollama") == 0.4
    monkeypatch.setenv("VISION_ANALYZER_VLLM_TEMPERATURE", "0.9")     # только vLLM
    assert st.initial("vllm") == 0.9 and st.initial("ollama") == 0.4
    monkeypatch.setenv("VISION_ANALYZER_OLLAMA_TEMPERATURE", "9")
    with pytest.raises(ValueError, match="VISION_ANALYZER_OLLAMA_TEMPERATURE"):
        st.initial("ollama")


# --- HTTP ---------------------------------------------------------------------------

def _with_client(coro):
    async def run():
        async with TestClient(TestServer(server.build_app())) as client:
            return await coro(client)
    return asyncio.run(run())


def test_http_settings_endpoints(monkeypatch):
    monkeypatch.setattr(config, "EMBEDDING_ENABLED", False, raising=False)

    async def scenario(c):
        r = await c.get("/providers/ollama/settings")
        assert r.status == 200
        body = await r.json()
        assert body["provider"] == "ollama" and body["values"]["temperature"] == 0.0

        r = await c.post("/providers/ollama/settings", json={"temperature": 0.8, "think": "low"})
        assert r.status == 200 and (await r.json())["values"]["think"] == "low"

        # у vLLM — своё, ollama не затронут
        r = await c.get("/providers/vllm/settings")
        assert (await r.json())["values"]["temperature"] == 0.0

        r = await c.post("/providers/vllm/settings?num_ctx=4096")
        assert r.status == 400 and (await r.json())["key"] == "num_ctx"

        r = await c.post("/providers/gemini/settings", json={"temperature": 10})
        assert r.status == 400

        r = await c.post("/providers/nope/settings", json={"temperature": 1})
        assert r.status == 404

        r = await c.delete("/providers/ollama/settings?keys=think")
        body = await r.json()
        assert body["values"]["think"] is True and body["values"]["temperature"] == 0.8

        r = await c.get("/providers")
        data = await r.json()
        by_name = {p["name"]: p for p in data["providers"]}
        assert by_name["ollama"]["settings"][0]["key"] == "temperature"
        assert "think_budget" in [s["key"] for s in by_name["vllm"]["settings"]]

    _with_client(scenario)


def test_http_legacy_sampling_alias_is_scoped_to_one_backend(monkeypatch):
    monkeypatch.setattr(config, "BACKEND", "vllm")

    async def scenario(c):
        r = await c.post("/sampling", json={"temperature": 0.6, "num_ctx": 8192})
        body = await r.json()
        assert r.status == 200 and body["backend"] == "vllm" and body["ignored"] == ["num_ctx"]
        assert r.headers.get("Deprecation") == "true"
        assert providers.get("vllm").settings["temperature"] == 0.6
        assert providers.get("ollama").settings["temperature"] == 0.0

        r = await c.post("/sampling?backend=ollama", json={"temperature": 0.1})
        assert r.status == 200 and providers.get("ollama").settings["temperature"] == 0.1
        assert providers.get("vllm").settings["temperature"] == 0.6

        r = await c.post("/sampling", json={"num_ctx": 8192})  # у vLLM нет ни одного из ключей
        assert r.status == 400

    _with_client(scenario)
