"""Язык интерфейса (cookie lang) уходит серверу анализа как lang — модель отвечает на нём же.

Анализ идёт в фоне (там нет cookie), поэтому язык запоминается в AnalysisResult.lang при
постановке в очередь. Чат выполняется прямо в запросе и берёт язык из cookie напрямую.
"""

import base64
import io
import os
from unittest import mock

import pytest
from cryptography.fernet import Fernet
from PIL import Image

from vision_app import create_app, queue_worker, services
from vision_app.config import conf
from vision_app.extensions import db
from vision_app.models import AnalysisResult, Role, Status, User


def _png(color: str = "red") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture()
def app(tmp_path):
    os.environ["VISION_CREDENTIALS_KEY"] = Fernet.generate_key().decode()
    app = create_app({
        "TESTING": True, "WTF_CSRF_ENABLED": False, "SECRET_KEY": "test",
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path}/t.db", "UPLOAD_FOLDER": str(tmp_path / "up"),
    })
    with app.app_context():
        db.create_all()
        user = User(username="head", role=Role.HEAD_ADMIN)
        user.set_password("Sup3r-secret-pass")
        db.session.add(user)
        db.session.commit()
    return app


def _login(app, lang: str | None):
    c = app.test_client()
    if lang:
        c.set_cookie("lang", lang)
    c.post("/accounts/login/", data={"username": "head", "password": "Sup3r-secret-pass"})
    return c


def _sent_lang(kwargs: dict) -> str | None:
    """lang из запроса к серверу: в JSON-теле (с категориями / ссылки) или в query-параметрах."""
    return (kwargs.get("json") or {}).get("lang") or (kwargs.get("params") or {}).get("lang")


class _Recorder:
    """Подмена requests.post: запоминает kwargs и отвечает валидным результатом /analyze."""

    def __init__(self):
        self.calls: list[dict] = []

    def __call__(self, url, **kw):
        self.calls.append(kw)
        resp = mock.Mock(text="", status_code=200)
        link = (kw.get("json") or {}).get("url")
        item = {"file": "x", "backend": "ollama", "report": {"risk_level": "low", "description": "d"}}
        if link:
            item["source"] = {
                "caption": "", "image_b64": base64.b64encode(_png("blue")).decode(), "image_mime": "image/png",
            }
        resp.json.return_value = {"results": [item]}
        return resp


def _drain(app, rec):
    with mock.patch.object(services.requests, "post", rec), \
         mock.patch.object(queue_worker.vector_search, "index_analysis", lambda _id: None):
        return queue_worker.drain(app)


# --- анализ изображений через очередь ---------------------------------------------------------
@pytest.mark.parametrize("cookie, expected", [("en", "en"), ("ru", "ru"), (None, "ru")])
def test_uploaded_image_uses_ui_language(app, cookie, expected):
    c = _login(app, cookie)
    c.post("/", data={"image": (io.BytesIO(_png()), "a.png")}, content_type="multipart/form-data")

    with app.app_context():
        row = db.session.scalars(db.select(AnalysisResult)).one()
        assert row.status == Status.QUEUED and row.lang == expected

    rec = _Recorder()
    assert _drain(app, rec) == 1
    assert [_sent_lang(k) for k in rec.calls] == [expected]


def test_language_is_fixed_at_enqueue_time(app):
    """Переключили язык ПОСЛЕ загрузки — задача всё равно идёт на языке, выбранном при загрузке."""
    c = _login(app, "en")
    c.post("/", data={"image": (io.BytesIO(_png()), "a.png")}, content_type="multipart/form-data")
    c.set_cookie("lang", "ru")
    c.get("/history/")

    rec = _Recorder()
    _drain(app, rec)
    assert [_sent_lang(k) for k in rec.calls] == ["en"]


def test_legacy_job_without_lang_falls_back_to_default(app):
    """Записи, поставленные до появления колонки (lang=""), получают DEFAULT_LANG из настроек."""
    _login(app, "en")
    with app.app_context():
        user = db.session.scalars(db.select(User)).one()
        path = os.path.join(app.config["UPLOAD_FOLDER"], "old.png")
        os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
        with open(path, "wb") as fh:
            fh.write(_png())
        db.session.add(AnalysisResult(
            user_id=user.id, image_path="old.png", image_mime="image/png", status=Status.QUEUED, lang="",
        ))
        db.session.commit()
        default = conf("DEFAULT_LANG")

    rec = _Recorder()
    _drain(app, rec)
    assert [_sent_lang(k) for k in rec.calls] == [default]


# --- анализ по ссылке --------------------------------------------------------------------------
def test_link_job_uses_ui_language(app):
    c = _login(app, "en")
    c.post("/", data={"links": "https://example.com/p/1"})

    rec = _Recorder()
    assert _drain(app, rec) == 1
    assert [_sent_lang(k) for k in rec.calls] == ["en"]
    with app.app_context():
        assert db.session.scalars(db.select(AnalysisResult)).one().lang == "en"


# --- чат: выполняется в запросе, язык берётся из cookie напрямую ------------------------------------
def _chat_lang(app, cookie_header: str | None) -> str:
    rec = mock.Mock()
    rec.return_value = mock.Mock(status_code=200, text="")
    rec.return_value.json.return_value = {"reply": "ok", "backend": "b", "model": "m"}
    headers = {"Cookie": cookie_header} if cookie_header else {}
    with app.test_request_context("/", headers=headers), \
         mock.patch.object(services.requests, "post", rec):
        services.chat_with_model("hi")
    return rec.call_args.kwargs["json"]["lang"]


def test_chat_message_uses_ui_language(app):
    assert _chat_lang(app, "lang=en") == "en"
    assert _chat_lang(app, "lang=ru") == "ru"


def test_chat_without_cookie_is_russian(app):
    assert _chat_lang(app, None) == "ru"


def test_explicit_lang_wins_over_ui_language(app):
    with app.test_request_context("/", headers={"Cookie": "lang=en"}):
        assert services.resolve_lang("ru") == "ru"
        assert services.resolve_lang(None) == "en"


def test_ui_lang_outside_request_is_none(app):
    with app.app_context():
        assert services.ui_lang() is None
        assert services.resolve_lang() == conf("DEFAULT_LANG")
