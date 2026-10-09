"""Экспорт чата: JSON-файл и текст для буфера обмена; чужие чаты недоступны."""

import json
import os

import pytest
from cryptography.fernet import Fernet

from vision_app import create_app
from vision_app.extensions import db
from vision_app.models import ChatMessage, ChatRole, ChatSession, Role, User

PASSWORD = "Sup3r-secret-pass"


@pytest.fixture()
def app(tmp_path):
    os.environ["VISION_CREDENTIALS_KEY"] = Fernet.generate_key().decode()
    app = create_app({
        "TESTING": True, "WTF_CSRF_ENABLED": False, "SECRET_KEY": "test",
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path}/t.db", "UPLOAD_FOLDER": str(tmp_path / "up"),
    })
    with app.app_context():
        db.create_all()
        for name in ("alice", "bob"):
            u = User(username=name, role=Role.USER)
            u.set_password(PASSWORD)
            db.session.add(u)
        db.session.commit()
        alice = User.query.filter_by(username="alice").one()
        s = ChatSession(user_id=alice.id, title="Поиск символики")
        db.session.add(s)
        db.session.commit()
        db.session.add_all([
            ChatMessage(session_id=s.id, role=ChatRole.USER, content="поищи никаб",
                        images=[{"path": "chat_uploads/x.png", "name": "x.png", "mime": "image/png"}]),
            ChatMessage(session_id=s.id, role=ChatRole.ASSISTANT, content="Нашла №16.\n\nВторой абзац",
                        backend="anthropic", model="claude-haiku-5-5",
                        refs=[{"id": 16, "label": "test3.jpeg", "risk_label": "Средний",
                               "risk_level": "medium", "url": "/result/16", "thumb_url": "/thumb/x"}]),
        ])
        db.session.commit()
    return app


def _login(app, username):
    c = app.test_client()
    c.post("/accounts/login/", data={"username": username, "password": PASSWORD})
    return c


def test_json_export_has_full_dialog_and_downloads_as_file(app):
    r = _login(app, "alice").get("/chat/1/export.json")
    assert r.status_code == 200
    assert r.mimetype == "application/json"
    assert 'attachment; filename="chat-1-' in r.headers["Content-Disposition"]
    data = json.loads(r.get_data(as_text=True))
    assert data["session"]["title"] == "Поиск символики"
    user_msg, bot_msg = data["messages"]
    assert user_msg["role"] == "user" and user_msg["content"] == "поищи никаб"
    assert user_msg["attachments"] == ["x.png"]  # только имена файлов
    assert bot_msg["model"] == "claude-haiku-5-5" and bot_msg["content"].endswith("Второй абзац")
    assert bot_msg["refs"] == [{"id": 16, "label": "test3.jpeg", "risk_level": "medium",
                                "risk_label": "Средний", "url": "/result/16"}]  # без thumb_url


def test_json_is_not_escaped_for_cyrillic(app):
    body = _login(app, "alice").get("/chat/1/export.json").get_data(as_text=True)
    assert "Поиск символики" in body and "\\u" not in body


def test_text_export_for_clipboard(app):
    r = _login(app, "alice").get("/chat/1/export.txt")
    assert r.status_code == 200 and r.mimetype == "text/plain"
    text = r.get_data(as_text=True)
    assert text.startswith("Чат: Поиск символики")
    assert "Вы:\nпоищи никаб\nВложения: x.png" in text
    assert "Модель (anthropic · claude-haiku-5-5):\nНашла №16." in text
    assert "Анализы: №16 test3.jpeg (Средний)" in text


def test_foreign_chat_is_404_and_anonymous_is_redirected(app):
    bob = _login(app, "bob")
    assert bob.get("/chat/1/export.json").status_code == 404
    assert bob.get("/chat/1/export.txt").status_code == 404
    anon = app.test_client()
    assert anon.get("/chat/1/export.json").status_code in (301, 302, 401)


def test_chat_page_has_export_buttons(app):
    html = _login(app, "alice").get("/chat/1").get_data(as_text=True)
    assert 'id="chat-copy-btn"' in html and "/chat/1/export.txt" in html
    assert 'id="chat-export-btn"' in html and "/chat/1/export.json" in html
