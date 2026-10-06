"""Левая панель: имя аккаунта (Фамилия Имя, иначе логин), порядок блоков в подвале."""

import os

import pytest
from cryptography.fernet import Fernet

from vision_app import create_app
from vision_app.extensions import db
from vision_app.models import Role, User

PASSWORD = "Sup3r-secret-pass"


def test_display_name_styles_and_initials():
    u = User(username="ivan_p", nickname="Иван Петров", display_style="nickname")
    assert (u.display_name, u.initials) == ("Иван Петров", "ИП")
    u.display_style = "username"
    assert u.display_name == "ivan_p"
    u.display_style = "both"
    assert u.display_name == "Иван Петров (@ivan_p)"
    # без никнейма всегда логин, какой бы стиль ни был выбран
    for style in ("nickname", "username", "both"):
        u = User(username="nick", nickname="  ", display_style=style)
        assert (u.display_name, u.initials) == ("nick", "N")
    # никнейм совпадает с логином — не дублируем
    assert User(username="Bob", nickname="bob", display_style="both").display_name == "bob"
    assert User(username="x", nickname="Один", display_style="both").initials == "О"


@pytest.fixture()
def app(tmp_path):
    os.environ["VISION_CREDENTIALS_KEY"] = Fernet.generate_key().decode()
    app = create_app({
        "TESTING": True, "WTF_CSRF_ENABLED": False, "SECRET_KEY": "test",
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path}/t.db", "UPLOAD_FOLDER": str(tmp_path / "up"),
    })
    with app.app_context():
        db.create_all()
        for name, nick in (("named", "Иван Петров"), ("plain", "")):
            u = User(username=name, nickname=nick, role=Role.USER)
            u.set_password(PASSWORD)
            db.session.add(u)
        db.session.commit()
    return app


def _page(app, username):
    c = app.test_client()
    c.post("/accounts/login/", data={"username": username, "password": PASSWORD})
    return c.get("/history/").get_data(as_text=True)


def test_sidebar_shows_nickname_or_username(app):
    html = _page(app, "named")
    assert ">Иван Петров<" in html and ">named<" not in html
    assert ">plain<" in _page(app, "plain")


def test_appearance_is_above_account(app):
    html = _page(app, "named")
    assert html.index("Оформление") < html.index('class="account-card"')
