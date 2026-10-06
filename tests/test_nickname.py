"""Логин (уникальный) + никнейм и стиль отображения имени; перенос старого ФИО."""

import os
import sqlite3

import pytest
from cryptography.fernet import Fernet
from sqlalchemy import inspect

from vision_app import create_app
from vision_app.extensions import db
from vision_app.models import Role, User

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
        for name, role in (("alice", Role.USER), ("bob", Role.USER), ("adm", Role.ADMIN)):
            u = User(username=name, role=role)
            u.set_password(PASSWORD)
            db.session.add(u)
        db.session.commit()
    return app


def _login(app, username):
    c = app.test_client()
    c.post("/accounts/login/", data={"username": username, "password": PASSWORD})
    return c


def _user(app, name):
    with app.app_context():
        u = User.query.filter_by(username=name).one()
        return u.id, u.username, u.nickname, u.display_style


def _save(client, **over):
    data = {"username": "alice", "email": "", "nickname": "", "display_style": "nickname", **over}
    return client.post("/accounts/profile/", data=data)


def test_profile_saves_nickname_and_style(app):
    c = _login(app, "alice")
    r = _save(c, nickname="  Алиса   Смит ", display_style="both")
    assert r.status_code == 302
    _, _, nick, style = _user(app, "alice")
    assert (nick, style) == ("Алиса Смит", "both")  # пробелы схлопнуты
    assert "Алиса Смит (@alice)" in c.get("/history/").get_data(as_text=True)


def test_invalid_style_rejected(app):
    c = _login(app, "alice")
    r = _save(c, nickname="X", display_style="hacker")
    assert r.status_code == 200  # форма вернулась с ошибкой
    assert _user(app, "alice")[2:] == ("", "nickname")


def test_nickname_too_long_rejected(app):
    assert _save(_login(app, "alice"), nickname="я" * 151).status_code == 200
    assert _user(app, "alice")[2] == ""


def test_nickname_not_unique_but_username_is(app):
    _save(_login(app, "alice"), nickname="Same")
    assert _save(_login(app, "bob"), username="bob", nickname="Same").status_code == 302  # никнеймы могут совпадать
    r = _save(_login(app, "bob"), username="ALICE", nickname="Same")  # логин — нет, и без учёта регистра
    assert r.status_code == 200 and "уже существует" in r.get_data(as_text=True)
    assert _user(app, "bob")[1] == "bob"


def test_profile_form_has_select_and_avatar_on_top(app):
    html = _login(app, "alice").get("/accounts/profile/").get_data(as_text=True)
    assert '<select' in html and 'name="display_style"' in html and 'name="nickname"' in html
    assert "first_name" not in html and "Фамилия" not in html
    edit = html.index("Изменить данные")
    assert edit < html.index('id="avatar-card"') < html.index('name="nickname"')  # аватар — вверху карточки
    assert 'class="card avatar-card"' not in html  # отдельной карточки больше нет


def test_admin_can_edit_nickname_and_style(app):
    c = _login(app, "adm")
    uid = _user(app, "bob")[0]
    r = c.post(f"/panel/users/{uid}/edit/", data={"username": "bob", "email": "", "nickname": "Боб", "display_style": "username"})
    assert r.status_code == 302
    assert _user(app, "bob")[2:] == ("Боб", "username")
    page = c.get(f"/panel/users/{uid}/").get_data(as_text=True)
    assert "Показывается как" in page and "Боб" in page


# ---- перенос ФИО из старой базы ----

def _old_db(path):
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE users (id INTEGER PRIMARY KEY, username VARCHAR(150) NOT NULL, email VARCHAR(254) NOT NULL DEFAULT '',
        first_name VARCHAR(150) NOT NULL, last_name VARCHAR(150) NOT NULL, password_hash VARCHAR(256) NOT NULL,
        active BOOLEAN NOT NULL DEFAULT 1, role VARCHAR(20) NOT NULL DEFAULT 'user', created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP)""")
    con.executemany("INSERT INTO users (username, first_name, last_name, password_hash) VALUES (?,?,?,'x')", [
        ("full", "Иван", "Петров"), ("onlyfirst", "Иван", ""), ("none", "", ""), ("padded", "  Анна ", " Сидорова  "),
    ])
    con.commit()
    con.close()


def test_legacy_names_migrated_and_columns_dropped(tmp_path):
    os.environ["VISION_CREDENTIALS_KEY"] = Fernet.generate_key().decode()
    _old_db(tmp_path / "old.db")
    cfg = {"TESTING": True, "SECRET_KEY": "t", "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path}/old.db",
           "UPLOAD_FOLDER": str(tmp_path / "up"), "WTF_CSRF_ENABLED": False}
    for _ in range(2):  # второй запуск — проверка, что миграция идемпотентна
        app = create_app(cfg)
        with app.app_context():
            names = {u.username: u.nickname for u in User.query.all()}
            assert names == {"full": "Петров Иван", "onlyfirst": "Иван", "none": "", "padded": "Сидорова Анна"}
            cols = {c["name"] for c in inspect(db.engine).get_columns("users")}
            assert {"nickname", "display_style"} <= cols and not ({"first_name", "last_name"} & cols)
            assert User.query.filter_by(username="full").one().display_style == "nickname"
    with app.app_context():  # новый пользователь вставляется без ошибок NOT NULL
        u = User(username="fresh")
        u.set_password(PASSWORD)
        db.session.add(u)
        db.session.commit()
        assert u.display_name == "fresh"
