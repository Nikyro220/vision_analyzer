"""Левая панель: имя аккаунта (Фамилия Имя, иначе логин), порядок блоков в подвале."""

import os

import pytest
from cryptography.fernet import Fernet

from vision_app import create_app
from vision_app.extensions import db
from vision_app.models import Role, User

PASSWORD = "Sup3r-secret-pass"


def test_display_name_styles_and_initials():
    u = User(username="ivan_p", family_name="Петров", given_name="Иван", middle_name="Сергеевич",
             nickname="Ванёк", display_style="fio")
    assert (u.display_name, u.initials) == ("Петров Иван Сергеевич", "ПИ")
    u.display_style = "fi"
    assert (u.display_name, u.initials) == ("Петров Иван", "ПИ")
    u.display_style = "nickname"
    assert (u.display_name, u.initials) == ("Ванёк", "В")
    # нет нужных данных -> «Название аккаунта», а без него логин
    assert User(username="x", nickname="Ник", display_style="fio").display_name == "Ник"
    assert User(username="x", given_name="Иван", display_style="fio").display_name == "Иван"  # заполнена часть ФИО
    for style in ("fio", "fi", "nickname"):
        u = User(username="nick", nickname="  ", display_style=style)
        assert (u.display_name, u.initials) == ("nick", "N")
    # старые значения стиля (username / both) ведут себя как «Никнейм»
    for style in ("username", "both"):
        assert User(username="x", nickname="Один", display_style=style).display_name == "Один"


def test_user_color_helpers():
    u = User(username="c", color="#FF0000")
    assert (u.color_hex, u.color_fg) == ("#ff0000", "#ffffff")
    assert User(username="c", color="#ffff00").color_fg == "#111111"  # на светлом фоне — тёмный текст
    assert User(username="c", color="oops", id=3).color_hex.startswith("#")  # битое значение -> цвет по id


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
