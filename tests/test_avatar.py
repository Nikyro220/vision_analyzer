"""Аватары: проверка и перекодирование, загрузка, права доступа, удаление."""

import io
import os

import pytest
from cryptography.fernet import Fernet
from PIL import Image

from vision_app import avatars, create_app
from vision_app.extensions import db
from vision_app.history import delete_user_account
from vision_app.models import Role, User

PASSWORD = "Sup3r-secret-pass"


def _img(fmt="PNG", size=(400, 300), color="red") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, fmt)
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


def _upload(client, data, name="avatar.png"):
    return client.post("/accounts/avatar/", data={"avatar": (io.BytesIO(data), name)},
                       content_type="multipart/form-data")


def _uid(app, name):
    with app.app_context():
        return User.query.filter_by(username=name).one().id


# ---- process_avatar ----

def test_process_makes_square_webp(app):
    with app.app_context():
        out = avatars.process_avatar(_img("JPEG", (800, 300)))
    im = Image.open(io.BytesIO(out))
    assert im.format == "WEBP" and im.size == (256, 256)


@pytest.mark.parametrize("data", [b"", b"not an image", _img("GIF"), _img("BMP"), b"<svg xmlns='http://www.w3.org/2000/svg'/>"])
def test_process_rejects_bad_input(app, data):
    with app.app_context(), pytest.raises(avatars.AvatarError):
        avatars.process_avatar(data)


def test_process_rejects_huge_dimensions(app):
    with app.app_context(), pytest.raises(avatars.AvatarError):
        avatars.process_avatar(_img("PNG", (3000, 10)))


def test_process_rejects_oversized_file(app):
    app.config["AVATAR_MAX_UPLOAD_BYTES"] = 100
    with app.app_context(), pytest.raises(avatars.AvatarError):
        avatars.process_avatar(_img("PNG", (256, 256)))


# ---- endpoints ----

def test_upload_saves_and_serves(app):
    c = _login(app, "alice")
    r = _upload(c, _img())
    assert r.status_code == 200 and r.get_json()["ok"] is True
    uid = _uid(app, "alice")
    with app.app_context():
        assert db.session.get(User, uid).has_avatar and avatars.avatar_file(uid).is_file()
    got = c.get(f"/accounts/avatar/{uid}/")
    assert got.status_code == 200 and got.mimetype == "image/webp"
    assert Image.open(io.BytesIO(got.data)).size == (256, 256)


def test_upload_bad_file_returns_json_error(app):
    c = _login(app, "alice")
    r = _upload(c, b"garbage")
    assert r.status_code == 400 and r.get_json()["ok"] is False
    assert _upload(c, b"") .status_code == 400
    assert c.post("/accounts/avatar/", data={}, content_type="multipart/form-data").status_code == 400


def test_anonymous_cannot_upload(app):
    r = _upload(app.test_client(), _img())
    assert r.status_code in (302, 401)


def test_access_rules(app):
    _upload(_login(app, "alice"), _img())
    uid = _uid(app, "alice")
    assert _login(app, "bob").get(f"/accounts/avatar/{uid}/").status_code == 404
    assert _login(app, "adm").get(f"/accounts/avatar/{uid}/").status_code == 200
    assert app.test_client().get(f"/accounts/avatar/{uid}/").status_code in (302, 401)
    assert _login(app, "bob").get(f"/accounts/avatar/{_uid(app, 'bob')}/").status_code == 404  # у bob аватара нет


def test_sidebar_shows_image_after_upload(app):
    c = _login(app, "alice")
    assert "<img" not in c.get("/history/").get_data(as_text=True).split('class="account-card"')[1].split("</a>")[0]
    _upload(c, _img())
    assert f"/accounts/avatar/{_uid(app, 'alice')}/?v=" in c.get("/history/").get_data(as_text=True)


def test_profile_has_editor(app):
    html = _login(app, "alice").get("/accounts/profile/").get_data(as_text=True)
    assert 'id="avatar-editor"' in html and "avatar-editor.js" in html


def test_delete_avatar(app):
    c = _login(app, "alice")
    _upload(c, _img())
    uid = _uid(app, "alice")
    assert c.post("/accounts/avatar/delete/").status_code == 302
    with app.app_context():
        assert not db.session.get(User, uid).has_avatar and not avatars.avatar_file(uid).exists()


def test_deleting_account_removes_file(app):
    _upload(_login(app, "alice"), _img())
    uid = _uid(app, "alice")
    with app.app_context():
        assert avatars.avatar_file(uid).is_file()
        delete_user_account(db.session.get(User, uid))
        assert not avatars.avatar_file(uid).exists()
