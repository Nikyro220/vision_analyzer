"""Аватары: проверка и перекодирование, загрузка, анимация, права доступа, ссылка в БД, удаление."""

import io
import json
import os

import pytest
from cryptography.fernet import Fernet
from PIL import Image, ImageSequence

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


def test_avatar_visible_to_every_logged_in_user(app):
    _upload(_login(app, "alice"), _img())
    uid = _uid(app, "alice")
    for viewer in ("bob", "adm", "alice"):
        got = _login(app, viewer).get(f"/accounts/avatar/{uid}/")
        assert got.status_code == 200 and got.mimetype == "image/webp", viewer
    assert app.test_client().get(f"/accounts/avatar/{uid}/").status_code in (302, 401)  # аноним — нет
    assert _login(app, "bob").get(f"/accounts/avatar/{_uid(app, 'bob')}/").status_code == 404  # у bob аватара нет
    assert _login(app, "bob").get("/accounts/avatar/9999/").status_code == 404


def test_avatar_link_is_stored_in_db_and_shown_to_others(app):
    _upload(_login(app, "alice"), _img())
    uid = _uid(app, "alice")
    with app.app_context():
        user = db.session.get(User, uid)
        link = user.avatar_url
        assert link == avatars.avatar_link(uid, user.avatar_version) and link.startswith(f"/accounts/avatar/{uid}/?v=")
        with app.test_request_context():
            from flask import url_for
            assert url_for("accounts.avatar", user_id=uid, v=user.avatar_version) == link  # ссылка действительно ведёт на маршрут
    # админ в списке пользователей видит чужую картинку по этой ссылке, и она открывается
    page = _login(app, "adm").get("/panel/users/").get_data(as_text=True)
    assert f'src="{link}"' in page
    assert _login(app, "bob").get(link).status_code == 200


def test_old_avatars_get_a_link_on_startup(app):
    _upload(_login(app, "alice"), _img())
    uid = _uid(app, "alice")
    with app.app_context():
        user = db.session.get(User, uid)
        user.avatar_url = None  # как у аватара, загруженного до появления колонки
        db.session.commit()
        assert user.avatar_src.startswith("/accounts/avatar/")  # и без бэкфилла картинка не пропадает
        from vision_app.schema import backfill_user_fields
        assert backfill_user_fields() == 1
        assert db.session.get(User, uid).avatar_url == avatars.avatar_link(uid, user.avatar_version)
        assert backfill_user_fields() == 0  # идемпотентно


def test_sidebar_shows_image_after_upload(app):
    c = _login(app, "alice")
    assert "<img" not in c.get("/history/").get_data(as_text=True).split('class="account-card"')[1].split("</a>")[0]
    _upload(c, _img())
    assert f"/accounts/avatar/{_uid(app, 'alice')}/?v=" in c.get("/history/").get_data(as_text=True)


def test_profile_has_editor(app):
    html = _login(app, "alice").get("/accounts/profile/").get_data(as_text=True)
    assert 'id="avatar-editor"' in html and "avatar-editor.js" in html


def test_editor_controls_have_value_field_reset_and_position(app):
    html = _login(app, "alice").get("/accounts/profile/").get_data(as_text=True)
    keys = ("dx", "dy", "zoom", "sx", "sy", "rot", "hue", "sat", "bri", "con")
    for k in keys:
        assert f'data-k="{k}"' in html
        assert f'id="av-{k}-range"' in html and f'id="av-{k}"' in html  # ползунок + числовое поле
        assert f'data-reset="{k}"' in html  # кнопка сброса именно этой настройки
    assert "Позиция по X" in html and "Позиция по Y" in html
    assert 'id="avatar-pick"' in html  # кружок кликабелен


def test_avatarless_user_is_painted_with_own_color(app):
    with app.app_context():
        db.session.get(User, _uid(app, "alice")).color = "#12ab34"
        db.session.commit()
    html = _login(app, "alice").get("/accounts/profile/").get_data(as_text=True)
    assert "--avatar-bg: #12ab34" in html


def test_chat_uses_user_color(app):
    with app.app_context():
        db.session.get(User, _uid(app, "alice")).color = "#12ab34"
        db.session.commit()
    c = _login(app, "alice")
    r = c.get("/chat/", follow_redirects=True)
    assert r.status_code == 200 and "--user-color: #12ab34" in r.get_data(as_text=True)


def test_delete_avatar(app):
    c = _login(app, "alice")
    _upload(c, _img())
    uid = _uid(app, "alice")
    assert c.post("/accounts/avatar/delete/").status_code == 302
    with app.app_context():
        user = db.session.get(User, uid)
        assert not user.has_avatar and user.avatar_url is None and not avatars.avatar_file(uid).exists()


def test_deleting_account_removes_file(app):
    _upload(_login(app, "alice"), _img())
    uid = _uid(app, "alice")
    with app.app_context():
        assert avatars.avatar_file(uid).is_file()
        delete_user_account(db.session.get(User, uid))
        assert not avatars.avatar_file(uid).exists()


# ---- анимированные аватары ----

def _gif(frames=3, size=(100, 80), durations=60, half=False) -> bytes:
    """GIF из однотонных кадров; half=True — левая половина красная, правая синяя (для проверки геометрии)."""
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    imgs = []
    for i in range(frames):
        im = Image.new("RGB", size, colors[i % 4])
        if half:
            im.paste((0, 0, 255), (size[0] // 2, 0, size[0], size[1]))
            im.paste((255, 0, 0), (0, 0, size[0] // 2, size[1]))
        imgs.append(im.convert("P", palette=Image.Palette.ADAPTIVE))
    buf = io.BytesIO()
    imgs[0].save(buf, "GIF", save_all=True, append_images=imgs[1:], duration=durations, loop=0)
    return buf.getvalue()


def _upload_animated(client, data, params=None, name="a.gif"):
    return client.post("/accounts/avatar/", content_type="multipart/form-data", data={
        "avatar": (io.BytesIO(data), name),
        "params": json.dumps(params if params is not None else {}),
    })


def _frames(app, uid):
    with Image.open(avatars_path(app, uid)) as im:
        return im.format, im.size, [f.convert("RGBA").copy() for f in ImageSequence.Iterator(im)], getattr(im, "is_animated", False)


def avatars_path(app, uid):
    with app.app_context():
        return avatars.avatar_file(uid)


def test_animated_gif_keeps_animation(app):
    c = _login(app, "alice")
    r = _upload_animated(c, _gif(frames=3))
    assert r.status_code == 200 and r.get_json()["ok"] is True, r.get_json()
    uid = _uid(app, "alice")
    fmt, size, frames, animated = _frames(app, uid)
    assert (fmt, size, animated, len(frames)) == ("WEBP", (256, 256), True, 3)
    # кадры действительно разные (красный, зелёный, синий), а не один кадр, повторённый трижды
    centers = [f.getpixel((128, 128))[:3] for f in frames]
    assert centers[0][0] > 200 and centers[1][1] > 200 and centers[2][2] > 200
    got = c.get(f"/accounts/avatar/{uid}/")
    assert got.status_code == 200 and got.mimetype == "image/webp"
    with app.app_context():
        assert db.session.get(User, uid).avatar_url


def test_animated_webp_and_apng_sources(app):
    frames = [Image.new("RGBA", (64, 64), c) for c in ((255, 0, 0, 255), (0, 0, 255, 255))]
    wp, ap = io.BytesIO(), io.BytesIO()
    frames[0].save(wp, "WEBP", save_all=True, append_images=frames[1:], duration=80, loop=0)
    frames[0].save(ap, "PNG", save_all=True, append_images=frames[1:], duration=80, loop=0)
    for data, name in ((wp.getvalue(), "a.webp"), (ap.getvalue(), "a.png")):
        c = _login(app, "alice")
        assert _upload_animated(c, data, name=name).status_code == 200, name
        assert _frames(app, _uid(app, "alice"))[3] is True, name


def test_single_frame_goes_through_params_as_static(app):
    c = _login(app, "alice")
    assert _upload_animated(c, _gif(frames=1)).status_code == 200
    fmt, size, frames, animated = _frames(app, _uid(app, "alice"))
    assert (fmt, size, len(frames), animated) == ("WEBP", (256, 256), 1, False)


def test_params_geometry_flip_and_position(app):
    c = _login(app, "alice")
    uid = _uid(app, "alice")
    src = _gif(frames=2, size=(100, 100), half=True)  # слева красный, справа синий
    _upload_animated(c, src, {})
    f = _frames(app, uid)[2][0]
    assert f.getpixel((20, 128))[0] > 200 and f.getpixel((236, 128))[2] > 200
    _upload_animated(c, src, {"flip": -1})  # отражение по горизонтали
    f = _frames(app, uid)[2][0]
    assert f.getpixel((20, 128))[2] > 200 and f.getpixel((236, 128))[0] > 200
    _upload_animated(c, src, {"zoom": 0.5})  # уменьшили вдвое: по краям прозрачно
    f = _frames(app, uid)[2][0]
    assert f.getpixel((5, 128))[3] == 0 and f.getpixel((128 - 40, 128))[3] == 255
    _upload_animated(c, src, {"zoom": 0.5, "dx": 100})  # сдвиг вправо
    f = _frames(app, uid)[2][0]
    assert f.getpixel((128 + 100 - 40, 128))[3] == 255 and f.getpixel((128 - 40, 128))[3] == 0
    _upload_animated(c, src, {"zoom": 0.5, "rot": 90})  # поворот на 90°: левая (красная) половина уходит наверх
    f = _frames(app, uid)[2][0]
    assert f.getpixel((128, 128 - 40))[0] > 200 and f.getpixel((128, 128 + 40))[2] > 200


def test_params_colour_adjustments(app):
    c = _login(app, "alice")
    uid = _uid(app, "alice")
    src = _gif(frames=2, size=(64, 64))  # красный / зелёный
    _upload_animated(c, src, {"sat": 0})
    r, g, b, _ = _frames(app, uid)[2][0].getpixel((128, 128))
    assert abs(r - g) < 6 and abs(g - b) < 6  # серый
    _upload_animated(c, src, {"bri": 0})
    assert _frames(app, uid)[2][0].getpixel((128, 128))[:3] == (0, 0, 0)
    _upload_animated(c, src, {"hue": 120})  # красный -> зелёный
    r, g, b, _ = _frames(app, uid)[2][0].getpixel((128, 128))
    assert g > r and g > b


def test_animated_rejections(app):
    c = _login(app, "alice")
    def bad(data, params=None, name="a.gif"):
        r = _upload_animated(c, data, params, name)
        assert r.status_code == 400 and r.get_json()["ok"] is False
        return r.get_json()["error"]
    app.config["AVATAR_MAX_FRAMES"] = 2
    assert "кадров" in bad(_gif(frames=3))
    app.config["AVATAR_MAX_FRAMES"] = 200
    app.config["AVATAR_MAX_ANIMATED_BYTES"] = 100
    assert "слишком большой" in bad(_gif(frames=3))
    app.config["AVATAR_MAX_ANIMATED_BYTES"] = 8 * 1024 * 1024
    app.config["AVATAR_MAX_ANIMATED_PIXELS"] = 1000
    assert "тяжёл" in bad(_gif(frames=3))
    app.config["AVATAR_MAX_ANIMATED_PIXELS"] = 250_000_000
    assert "Не удалось прочитать" in bad(b"GIF89a-not-really")
    assert "PNG, JPEG" in bad(_img("BMP"), name="a.bmp")
    # мусорные параметры
    for raw in ("{not json", "[]", json.dumps({"zoom": "abc"}), '{"zoom": NaN}'):
        r = c.post("/accounts/avatar/", content_type="multipart/form-data",
                   data={"avatar": (io.BytesIO(_gif()), "a.gif"), "params": raw})
        assert r.status_code == 400, raw
    with app.app_context():
        assert not db.session.get(User, _uid(app, "alice")).has_avatar  # ничего не сохранилось


def test_params_are_clamped_and_extreme_zoom_refused(app):
    with app.app_context():
        p = avatars.parse_params({"zoom": 99, "dx": -9999, "rot": 1e9, "flip": "x"})
        assert (p.zoom, p.dx, p.rot, p.flip) == (5, -256, 180, 1)
    c = _login(app, "alice")
    wide = _gif(frames=2, size=(2000, 60))  # очень вытянутый кадр + максимум масштаба и растяжения
    r = _upload_animated(c, wide, {"zoom": 5, "sx": 3})
    assert r.status_code == 400 and "увеличение" in r.get_json()["error"]


def test_static_upload_still_rejects_gif_without_params(app):
    c = _login(app, "alice")
    assert _upload(c, _gif(frames=2), "a.gif").status_code == 400  # обычный путь по-прежнему принимает только PNG/JPEG/WebP
