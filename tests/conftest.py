import io

import pytest
from PIL import Image

from vision_app import create_app
from vision_app.extensions import db


@pytest.fixture()
def app(tmp_path):
    app = create_app(
        {
            "TESTING": True,
            "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'test.sqlite3'}",
            "UPLOAD_FOLDER": str(tmp_path / "media"),
            "WTF_CSRF_ENABLED": False,
            "SECRET_KEY": "test",
            "VISION_API_BASE_URL": "http://vision.test:6769",
        }
    )
    yield app
    with app.app_context():
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app):
    return app.test_client()


def make_png(size=(8, 8), color=(200, 30, 30)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture()
def png_bytes():
    return make_png()


def register(client, username, password="Str0ng-pass-42", email=""):
    return client.post(
        "/accounts/register/",
        data={"username": username, "email": email, "password1": password, "password2": password},
        follow_redirects=True,
    )


def login(client, username, password="Str0ng-pass-42"):
    return client.post(
        "/accounts/login/", data={"username": username, "password": password}, follow_redirects=True
    )


def logout(client):
    return client.post("/accounts/logout/", follow_redirects=True)


class FakeResponse:
    def __init__(self, payload=None, status_code=200, text=""):
        self._payload = payload
        self.status_code = status_code
        self.text = text or (str(payload) if payload is not None else "")

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def api_ok(report=None, backend="vllm"):
    report = report if report is not None else {
        "risk_level": "high",
        "needs_human_review": True,
        "description": "На изображении обнаружен подозрительный объект.",
        "signals": [{"id": "S1", "category": "weapon", "detail": "нечто похожее на нож"}],
        "rationale": "Потому что так.",
        "recommendation": "Передать модератору.",
        "text_on_image": "STOP",
        "context": "Улица",
    }
    return FakeResponse({"count": 1, "requested_backend": backend,
                         "results": [{"file": "x.png", "backend": backend, "report": report}]})


class FakeVision:
    """Имитация vision_analyzer_server.py: /health, /models, /sampling, /analyze.
    Помнит состояние sampling и записывает все обращения."""

    def __init__(self):
        self.vllm_ok = True
        self.ollama_ok = True
        self.models = {"vllm": ["google/gemma-4-31B-it", "qwen2.5-vl"], "ollama": ["llama3.2-vision", "gemma3:27b"]}
        self.models_shape = "plain"  # plain | openai | ollama | error
        self.sampling = {"temperature": 0.2, "top_p": 0.9, "top_k": 40, "seed": None, "num_ctx": 8192}
        self.sampling_fail = False
        self.calls = []          # (метод, путь, params, json)
        self.analyze_params = []  # params каждого /analyze

    def _health(self):
        return {
            "ok": True,
            "default_backend": "vllm",
            "backends": {
                "vllm": {"ok": self.vllm_ok, "endpoint": "http://gpu:8000", "model": "google/gemma-4-31B-it"},
                "ollama": {"ok": self.ollama_ok, "endpoint": "http://gpu:11434", "model": "llama3.2-vision"},
            },
        }

    def _models_payload(self, backend):
        names = self.models[backend]
        if self.models_shape == "openai":
            return {"object": "list", "data": [{"id": n} for n in names]}
        if self.models_shape == "ollama":
            return {"models": [{"name": n, "model": n} for n in names]}
        if self.models_shape == "error":
            return {backend: {"ok": False, "error": "connection refused"}}
        return {"backend": backend, "models": names}

    def get(self, url, params=None, timeout=None, **kw):
        path = url.split(":6769", 1)[1]
        self.calls.append(("GET", path, params, None))
        if path == "/health":
            return FakeResponse(self._health())
        if path == "/models":
            return FakeResponse(self._models_payload((params or {}).get("backend", "vllm")))
        if path == "/sampling":
            if self.sampling_fail:
                return FakeResponse({"error": "sampling недоступен"}, status_code=500)
            return FakeResponse({"sampling": dict(self.sampling)})
        return FakeResponse({"error": "nf"}, status_code=404)

    def post(self, url, data=None, json=None, headers=None, params=None, timeout=None, **kw):
        path = url.split(":6769", 1)[1]
        self.calls.append(("POST", path, params, json))
        if path == "/sampling":
            if self.sampling_fail:
                return FakeResponse({"error": "запись запрещена"}, status_code=400)
            self.sampling.update(json)
            return FakeResponse({"ok": True, "sampling": dict(self.sampling)})
        if path == "/analyze":
            self.analyze_params.append(dict(params or {}))
            return api_ok()
        return FakeResponse({"error": "nf"}, status_code=404)

    def posts(self, path):
        return [c for c in self.calls if c[0] == "POST" and c[1] == path]


@pytest.fixture()
def vision(monkeypatch):
    from vision_app import services

    fake = FakeVision()
    monkeypatch.setattr(services.requests, "get", fake.get)
    monkeypatch.setattr(services.requests, "post", fake.post)
    return fake
