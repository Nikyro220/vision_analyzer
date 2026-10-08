"""Анализ по ссылке на пост: форма -> очередь -> обработчик -> сохранённые картинки."""

import base64
import io
import os
from unittest import mock

import pytest
from cryptography.fernet import Fernet
from PIL import Image

from vision_app import create_app, credentials, queue_worker, services
from vision_app.extensions import db
from vision_app.forms import parse_links
from vision_app.models import AnalysisResult, Role, Status, User


def _png(color: str) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (32, 32), color).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture()
def client(tmp_path):
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
    c = app.test_client()
    c.post("/accounts/login/", data={"username": "head", "password": "Sup3r-secret-pass"})
    c.app_ = app
    return c


def test_parse_links():
    good, bad = parse_links("https://a.com/1  t.me/chan/5\nftp://x.com javascript:alert(1) https://a.com/1")
    assert good == ["https://a.com/1", "https://t.me/chan/5"]
    assert [reason for _, reason in bad] and len(bad) == 2


def test_empty_submit_rejected(client):
    r = client.post("/", data={}, content_type="multipart/form-data")
    assert r.status_code == 200 and "Выберите файл" in r.get_data(as_text=True)


def test_link_is_queued_and_processed(client):
    app = client.app_
    client.post("/", data={"links": "https://example.com/p/1 https://example.com/p/2"})

    def fake_post(url, **kw):
        link = kw["json"]["url"]
        resp = mock.Mock(text="")
        if link.endswith("/2"):
            resp.status_code = 422
            resp.json.return_value = {"error": "Картинка не найдена"}
            return resp
        resp.status_code = 200
        resp.json.return_value = {"results": [
            {"file": f"{link}#{i}", "backend": "ollama",
             "source": {"caption": "Автор: bob", "image_b64": base64.b64encode(_png(c)).decode(), "image_mime": "image/png"},
             "report": {"risk_level": "low", "description": f"d{i}"}}
            for i, c in ((1, "red"), (2, "blue"))
        ]}
        return resp

    with mock.patch.object(services.requests, "post", fake_post), \
         mock.patch.object(queue_worker.vector_search, "index_analysis", lambda _id: None):
        assert queue_worker.drain(app) == 2

    with app.app_context():
        rows = db.session.scalars(db.select(AnalysisResult).order_by(AnalysisResult.id)).all()
        assert len(rows) == 3 and all(r.status == Status.DONE for r in rows)
        failed = rows[1]  # ссылка /2
        assert failed.error and not failed.image_path
        ok = [r for r in rows if r.image_path]
        assert len(ok) == 2 and all(r.caption == "Автор: bob" for r in ok)
        assert all(os.path.exists(os.path.join(app.config["UPLOAD_FOLDER"], r.image_path)) for r in ok)

    for path in ("/history/", f"/result/{rows[0].id}/", f"/result/{failed.id}/", "/panel/analyses/"):
        assert client.get(path).status_code == 200


def test_clear_credential(client):
    app = client.app_
    with app.app_context():
        credentials.set_key("gemini", "AIzaSyFAKEKEY1234567890abcd")
    provs = [services.ProviderInfo(name="gemini", label="Gemini", credential=services.CredentialField(label="Ключ"))]
    health = {"ok": True, "default_backend": "gemini", "backends": {"gemini": {"ok": True}}}
    with mock.patch("vision_app.blueprints.analyzer.get_providers", lambda: provs), \
         mock.patch("vision_app.blueprints.analyzer.check_health", lambda force=False: health), \
         mock.patch("vision_app.services.get_providers", lambda: provs), \
         mock.patch("vision_app.services.check_health", lambda force=False: health):
        html = client.get("/health/").get_data(as_text=True)
        assert "input-locked" in html and "Стереть" in html
        client.post("/health/backend/gemini/settings", data={"clear_credential": "1"})
        html = client.get("/health/").get_data(as_text=True)
        assert "input-locked" not in html and 'type="password"' in html
