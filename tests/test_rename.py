"""Переименование записи: автор и админы — можно, чужой обычный пользователь — нельзя."""

import os

import pytest
from cryptography.fernet import Fernet

from vision_app import create_app
from vision_app.extensions import db
from vision_app.models import AnalysisResult, Role, Status, User

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
        for name, role in (("author", Role.USER), ("other", Role.USER), ("adm", Role.ADMIN)):
            u = User(username=name, role=role)
            u.set_password(PASSWORD)
            db.session.add(u)
        db.session.commit()
        author = User.query.filter_by(username="author").one()
        db.session.add(AnalysisResult(
            user_id=author.id, image_path="uploads/x.png", original_name="old.png", status=Status.DONE,
        ))
        db.session.commit()
    return app


def _login(app, username):
    c = app.test_client()
    c.post("/accounts/login/", data={"username": username, "password": PASSWORD})
    return c


def _name(app):
    with app.app_context():
        return db.session.get(AnalysisResult, 1).original_name


def _rename(client, name):
    return client.post("/result/1/rename/", data={"name": name})


def test_author_can_rename(app):
    r = _rename(_login(app, "author"), "  new\n name.png ")
    assert r.status_code == 302
    assert _name(app) == "new name.png"  # пробелы схлопнуты


def test_admin_can_rename_foreign(app):
    assert _rename(_login(app, "adm"), "by-admin").status_code == 302
    assert _name(app) == "by-admin"


def test_other_user_cannot_rename(app):
    assert _rename(_login(app, "other"), "hacked").status_code == 404
    assert _name(app) == "old.png"


def test_anonymous_redirected(app):
    r = _rename(app.test_client(), "hacked")
    assert r.status_code in (301, 302, 401) and _name(app) == "old.png"


@pytest.mark.parametrize("bad", ["", "   ", "x" * 256])
def test_invalid_names_rejected(app, bad):
    _rename(_login(app, "author"), bad)
    assert _name(app) == "old.png"


def test_form_visibility(app):
    assert 'id="rename-form"' in _login(app, "author").get("/result/1/").get_data(as_text=True)
    assert 'id="rename-form"' in _login(app, "adm").get("/result/1/").get_data(as_text=True)
    assert "Изменить название" not in _login(app, "author").get("/result/1/").get_data(as_text=True)
    assert _login(app, "other").get("/result/1/").status_code == 404
