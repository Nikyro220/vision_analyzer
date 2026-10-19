"""i18n: профиль переключается между RU (без cookie) и EN (cookie lang=en); проверка en .po.

Образец: tests/test_nickname.py — тот же фикстур `app` и `_login`.
Cookie задаётся как того требует установленный Werkzeug (3.1.x):
`client.set_cookie("lang", "en")` (первый аргумент — ключ, домен по умолчанию "localhost").
Ожидаемый английский текст фразы берётся из .po через babel.messages.pofile.read_po, а не хардкодится.
"""

import os
import re

import pytest
from cryptography.fernet import Fernet
from babel.messages.pofile import read_po

from vision_app import create_app
from vision_app.extensions import db
from vision_app.models import Role, User

PASSWORD = "Sup3r-secret-pass"

PO_PATH = os.path.join(
    os.path.dirname(__file__),
    "..", "vision_app", "translations", "en", "LC_MESSAGES", "messages.po",
)

# Ключ, для которого проверяем наличие английского перевода из .po (русский текст — «Данные аккаунта»).
MSGID = "account.data.title"
KEY_RE = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+)+$")
PH_RE = re.compile(r"%\(\w+?\)s")
KEYWORDS = {"_": None, "gettext": None, "lazy_gettext": None, "ngettext": (1, 2)}


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


def _profile_html(client):
    return client.get("/accounts/profile/").get_data(as_text=True)


def _po(locale="en"):
    path = PO_PATH.replace(os.sep + "en" + os.sep, os.sep + locale + os.sep)
    with open(path, encoding="utf-8") as f:
        return read_po(f)


def _en_po():
    return _po("en")


def _by_key(catalog):
    """key -> запись; у записей с множественным числом id — кортеж, ключ — его первый элемент."""
    return {(e.id[0] if isinstance(e.id, tuple) else e.id): e for e in catalog if e.id}


def _text(value):
    return "\n".join(value) if isinstance(value, (tuple, list)) else (value or "")


def _expected_translation(msgid):
    entry = _by_key(_en_po()).get(msgid)
    return entry.string if entry else ""


def test_profile_is_russian_without_cookie(app):
    html = _profile_html(_login(app, "alice"))
    assert "Данные аккаунта" in html
    assert '<html lang="ru"' in html


def test_profile_is_english_with_lang_cookie(app):
    expected = _expected_translation(MSGID)
    assert expected, f"перевод для {MSGID!r} не найден в en .po"
    c = _login(app, "alice")
    c.set_cookie("lang", "en")
    html = _profile_html(c)
    assert expected in html
    assert '<html lang="en"' in html


def test_profile_falls_back_to_russian_for_unknown_language(app):
    c = app.test_client()
    c.post("/accounts/login/", data={"username": "alice", "password": PASSWORD})
    c.set_cookie("lang", "xx")
    html = _profile_html(c)
    assert "Данные аккаунта" in html
    assert '<html lang="ru"' in html


def test_catalogs_have_nonempty_nonfuzzy_translations_with_matching_placeholders():
    ru, en = _by_key(_po("ru")), _by_key(_po("en"))
    assert set(ru) == set(en), f"ключи ru и en различаются: {set(ru) ^ set(en)}"
    for key in ru:
        assert KEY_RE.match(key), f"в каталоге не ключ: {key!r}"
        for loc, entries in (("ru", ru), ("en", en)):
            entry = entries[key]
            text = _text(entry.string)
            assert text.strip(), f"[{loc}] пустой перевод для {key!r}"
            assert "fuzzy" not in entry.flags, f"[{loc}] fuzzy для {key!r}"
            assert text != key, f"[{loc}] не переведён: {key!r}"
        ph_ru, ph_en = set(PH_RE.findall(_text(ru[key].string))), set(PH_RE.findall(_text(en[key].string)))
        assert ph_ru == ph_en, f"несовпадение %(...)s для {key!r}: ru={ph_ru} en={ph_en}"


def test_po_in_sync_with_code_and_templates():
    from babel.messages.extract import extract_from_dir
    root = os.path.join(os.path.dirname(__file__), "..", "vision_app")
    method_map = [("**.py", "python"), ("templates/**.html", "jinja2")]
    found = set()
    for _f, _l, message, _c, _x in extract_from_dir(root, method_map, keywords=KEYWORDS):
        key = message[0] if isinstance(message, tuple) else message
        if isinstance(key, str):
            found.add(key)
    assert all(KEY_RE.match(k) for k in found), f"в коде не ключи: {[k for k in found if not KEY_RE.match(k)][:5]}"
    for loc in ("ru", "en"):
        have = set(_by_key(_po(loc)))
        assert found == have, f"[{loc}] нет в .po: {found - have}; лишние в .po: {have - found}"


def test_plural_forms_ru_and_en(app):
    from flask_babel import ngettext
    with app.test_request_context("/"):
        ru = [ngettext("plural.files.one", "plural.files.other", n, n=n) for n in (1, 2, 5, 21)]
        assert ru == ["1 файл", "2 файла", "5 файлов", "21 файл"]
    with app.test_request_context("/", headers={"Cookie": "lang=en"}):
        en = [ngettext("plural.files.one", "plural.files.other", n, n=n) for n in (1, 2)]
        assert en == ["1 file", "2 files"]


def _js_dictionary(html):
    import json
    m = re.search(r"window\.I18N = (\{.*?\});", html, re.S)
    assert m, "словарь window.I18N не найден на странице"
    return json.loads(m.group(1))


def test_js_dictionary_follows_language(app):
    c = _login(app, "alice")
    ru = _js_dictionary(c.get("/").get_data(as_text=True))
    assert ru["js.queue.cancel"] == "Убрать из очереди"
    c.set_cookie("lang", "en")
    en = _js_dictionary(c.get("/").get_data(as_text=True))
    assert en["js.queue.cancel"] == "Remove from queue"
    assert set(ru) == set(en) and all(k.startswith("js.") for k in en)


def test_flash_and_validation_messages_are_translated(app):
    c = app.test_client()
    c.set_cookie("lang", "en")
    html = c.post("/accounts/login/", data={"username": "alice", "password": "wrong"}).get_data(as_text=True)
    assert "Wrong login or password." in html
    ru = app.test_client().post("/accounts/login/", data={"username": "alice", "password": "wrong"}).get_data(as_text=True)
    assert "Неверный логин или пароль." in ru
