"""Логин (уникальный), ФИО, название аккаунта, стиль отображения имени, цвет; перенос старого ФИО."""

import html as htmllib
import os
import re
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
    r = _save(c, nickname="  Алиса   Смит ", display_style="nickname")
    assert r.status_code == 302
    _, _, nick, style = _user(app, "alice")
    assert (nick, style) == ("Алиса Смит", "nickname")  # пробелы схлопнуты
    assert ">Алиса Смит<" in c.get("/history/").get_data(as_text=True)


def test_full_name_styles(app):
    c = _login(app, "alice")
    names = {"family_name": "Смит", "given_name": "Алиса", "middle_name": "Ивановна", "nickname": "Али"}
    assert _save(c, display_style="fio", **names).status_code == 302
    assert ">Смит Алиса Ивановна<" in c.get("/history/").get_data(as_text=True)
    assert _save(c, display_style="fi", **names).status_code == 302
    assert ">Смит Алиса<" in c.get("/history/").get_data(as_text=True)
    assert _save(c, display_style="nickname", **names).status_code == 302
    assert ">Али<" in c.get("/history/").get_data(as_text=True)


def test_color_saved_and_validated(app):
    c = _login(app, "alice")
    assert _save(c, color="#AABBCC").status_code == 302
    with app.app_context():
        assert User.query.filter_by(username="alice").one().color == "#aabbcc"
    assert _save(c, color="red").status_code == 200  # не #rrggbb
    assert _save(c).status_code == 302  # поле пустое — цвет остаётся прежним
    with app.app_context():
        assert User.query.filter_by(username="alice").one().color == "#aabbcc"


def test_color_is_random_on_register(app):
    colors = set()
    for i in range(6):
        c = app.test_client()
        r = c.post("/accounts/register/", data={"username": f"newbie{i}", "email": "", "password1": PASSWORD, "password2": PASSWORD})
        assert r.status_code == 302, r.get_data(as_text=True)[:300]
        with app.app_context():
            colors.add(User.query.filter_by(username=f"newbie{i}").one().color)
    assert all(len(c) == 7 and c.startswith("#") for c in colors) and len(colors) > 1


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
    assert 'name="family_name"' in html and 'name="given_name"' in html and 'name="middle_name"' in html
    assert 'type="color"' in html and "Название аккаунта" in html
    opts = html[html.index('name="display_style"'):]
    opts = opts[: opts.index("</select>")]
    assert re.findall(r"<option[^>]*>([^<]+)</option>", opts) == ["ФИО", "Фамилия Имя", "Никнейм"]
    assert "Логин (@" not in html and "Показывается как" not in html
    edit = html.index("Изменить данные")
    assert edit < html.index('id="avatar-card"') < html.index('name="nickname"')  # аватар — вверху карточки
    assert 'class="card avatar-card"' not in html  # отдельной карточки больше нет


def test_data_card_labels(app):
    c = _login(app, "alice")
    _save(c, nickname="Ник", family_name="Смит", given_name="Алиса", display_style="fi")
    html = c.get("/accounts/profile/").get_data(as_text=True)
    card = html[html.index("Данные аккаунта"): html.index("Изменить данные")]
    assert "<dt>Название аккаунта</dt><dd>Ник</dd>" in card
    assert "<dt>Никнейм</dt><dd>Смит Алиса</dd>" in card  # то, что показывается, теперь подписано «Никнейм»
    assert "Показывается как" not in card


def test_admin_can_edit_names_and_style(app):
    c = _login(app, "adm")
    uid = _user(app, "bob")[0]
    r = c.post(f"/panel/users/{uid}/edit/", data={
        "username": "bob", "email": "", "nickname": "Боб", "display_style": "fi",
        "family_name": "Марли", "given_name": "Боб", "middle_name": "", "color": "#112233",
    })
    assert r.status_code == 302
    assert _user(app, "bob")[2:] == ("Боб", "fi")
    page = c.get(f"/panel/users/{uid}/").get_data(as_text=True)
    assert "Название аккаунта" in page and "Марли Боб" in page and "#112233" in page


def _page_text(html):
    return html[html.index("<h3 class=\"subtitle stats-title\">") :]


def test_profile_and_panel_show_analysis_stats(app):
    from vision_app.models import AnalysisResult
    uid = _user(app, "bob")[0]
    with app.app_context():
        for i, (status, risk, review) in enumerate([("done", "high", True), ("done", "low", False), ("done", "medium", False), ("queued", "unknown", False)]):
            db.session.add(AnalysisResult(user_id=uid, image_path=f"x/{i}.png", status=status, risk_level=risk, needs_human_review=review))
        db.session.commit()
    own = _page_text(_login(app, "bob").get("/accounts/profile/").get_data(as_text=True))
    panel = _page_text(_login(app, "adm").get(f"/panel/users/{uid}/").get_data(as_text=True))
    for page in (own, panel):
        assert '<span class="stat-num">4</span><span class="stat-label">всего анализов' in page
        assert '<span class="stat-num">3</span><span class="stat-label">завершено' in page
        assert '<span class="stat-num">1</span><span class="stat-label">высокий риск' in page
        assert "<dt>В очереди / в обработке</dt><dd>1</dd>" in page
        assert "<dt>Риск средний / низкий</dt><dd>1 / 1</dd>" in page


def test_stats_for_user_without_anything(app):
    html = _page_text(_login(app, "alice").get("/accounts/profile/").get_data(as_text=True))
    assert '<span class="stat-num">0</span><span class="stat-label">всего анализов' in html
    assert "<dt>Первый анализ</dt><dd>—</dd>" in html


# ---- удаление аккаунта: подтверждение логином ----

def _delete(client, url, text):
    return client.post(url, data={"confirm_sql": text})


def test_delete_own_account_requires_username(app):
    c = _login(app, "alice")
    uid = _user(app, "alice")[0]
    assert "DELETE FROM users WHERE username = 'alice';" in htmllib.unescape(c.get("/accounts/profile/").get_data(as_text=True))
    _delete(c, "/accounts/profile/delete/", f"DELETE FROM users WHERE id = {uid};")  # старая команда с id больше не годится
    _delete(c, "/accounts/profile/delete/", "DELETE FROM users WHERE username = 'bob';")  # чужой логин
    _delete(c, "/accounts/profile/delete/", "alice")  # одного логина мало
    with app.app_context():
        assert User.query.filter_by(username="alice").count() == 1
    assert _delete(c, "/accounts/profile/delete/", "DELETE FROM users WHERE username = 'alice';").status_code == 302
    with app.app_context():
        assert User.query.filter_by(username="alice").count() == 0


def test_admin_delete_requires_target_username(app):
    c = _login(app, "adm")
    uid = _user(app, "bob")[0]
    url = f"/panel/users/{uid}/delete/"
    page = c.get(f"/panel/users/{uid}/").get_data(as_text=True)
    assert "DELETE FROM users WHERE username = 'bob';" in htmllib.unescape(page)
    _delete(c, url, f"DELETE FROM users WHERE id = {uid};")
    _delete(c, url, "DELETE FROM users WHERE username = 'adm';")
    with app.app_context():
        assert User.query.filter_by(username="bob").count() == 1
    _delete(c, url, "DELETE FROM users WHERE username = 'bob';")
    with app.app_context():
        assert User.query.filter_by(username="bob").count() == 0


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
            assert {"nickname", "display_style", "family_name", "given_name", "middle_name", "color", "avatar_url"} <= cols
            assert not ({"first_name", "last_name"} & cols)
            assert User.query.filter_by(username="full").one().display_style == "nickname"
            assert all(u.color.startswith("#") and len(u.color) == 7 for u in User.query.all())  # цвет выдан старым аккаунтам
    with app.app_context():  # новый пользователь вставляется без ошибок NOT NULL
        u = User(username="fresh")
        u.set_password(PASSWORD)
        db.session.add(u)
        db.session.commit()
        assert u.display_name == "fresh"


def test_old_display_styles_and_missing_colors_are_backfilled(app):
    from vision_app.schema import backfill_user_fields
    with app.app_context():
        a, b = User.query.filter_by(username="alice").one(), User.query.filter_by(username="bob").one()
        a.display_style, a.color = "both", ""  # значения из старой версии
        b.display_style, b.color = "username", "#123456"
        db.session.commit()
        assert backfill_user_fields() == 2
        a, b = User.query.filter_by(username="alice").one(), User.query.filter_by(username="bob").one()
        assert (a.display_style, b.display_style) == ("nickname", "nickname")
        assert a.color.startswith("#") and len(a.color) == 7 and b.color == "#123456"  # чужой цвет не трогаем
