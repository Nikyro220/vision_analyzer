import io

import pytest
import requests

from conftest import FakeResponse, api_ok, login, logout, make_png, register
from vision_app import services
from vision_app.extensions import db
from vision_app.models import AnalysisResult, Role, User


def upload(client, data=None, name="pic.png", content_type="image/png"):
    data = data if data is not None else make_png()
    return client.post(
        "/",
        data={"image": (io.BytesIO(data), name, content_type)},
        content_type="multipart/form-data",
    )


def role_of(app, username):
    with app.app_context():
        return db.session.scalars(db.select(User).where(User.username == username)).one().role


def uid(app, username):
    with app.app_context():
        return db.session.scalars(db.select(User).where(User.username == username)).one().id


# ------------------------------------------------------------------ accounts
def test_first_user_is_head_admin_next_is_user(app, client):
    r = register(client, "boss")
    assert "Главный администратор" in r.get_data(as_text=True)
    logout(client)
    r = register(client, "worker")
    assert "Регистрация прошла успешно" in r.get_data(as_text=True)
    assert role_of(app, "boss") == Role.HEAD_ADMIN
    assert role_of(app, "worker") == Role.USER


def test_register_validation(app, client):
    register(client, "boss")
    logout(client)
    # дубликат логина без учёта регистра
    r = register(client, "BOSS")
    assert "Пользователь с таким логином уже существует." in r.get_data(as_text=True)
    # слабые пароли
    assert "слишком короткий" in register(client, "u1", password="abc").get_data(as_text=True)
    assert "только из цифр" in register(client, "u2", password="12345678901").get_data(as_text=True)
    assert "широко распространён" in register(client, "u3", password="password123").get_data(as_text=True)
    assert "слишком похож на логин" in register(client, "alexander", password="alexander99").get_data(as_text=True)
    # несовпадение паролей
    r = client.post("/accounts/register/", data={"username": "u4", "password1": "Str0ng-pass-42", "password2": "other-pass-1"})
    assert "пароли не совпадают" in r.get_data(as_text=True)


def test_login_logout_flow(app, client):
    register(client, "boss")
    r = logout(client)
    assert "Вы вышли из системы." in r.get_data(as_text=True)
    r = login(client, "boss", "wrong")
    assert "Неверный логин или пароль." in r.get_data(as_text=True)
    r = login(client, "BOSS")  # логин без учёта регистра
    assert "С возвращением, boss!" in r.get_data(as_text=True)


def test_logout_requires_post(client):
    register(client, "boss")
    assert client.get("/accounts/logout/").status_code == 405


def test_login_next_is_safe(client):
    register(client, "boss")
    logout(client)
    r = client.post("/accounts/login/?next=//evil.com", data={"username": "boss", "password": "Str0ng-pass-42"})
    assert r.headers["Location"] == "/"
    logout(client)
    r = client.post("/accounts/login/?next=/history/", data={"username": "boss", "password": "Str0ng-pass-42"})
    assert r.headers["Location"] == "/history/"


def test_anonymous_redirected_to_login(client):
    for url in ["/", "/history/", "/health/", "/panel/users/", "/accounts/profile/"]:
        r = client.get(url)
        assert r.status_code == 302 and "/accounts/login/" in r.headers["Location"], url


def test_profile_update(app, client):
    register(client, "boss")
    r = client.post("/accounts/profile/", data={"email": "b@x.kz", "first_name": "Бек", "last_name": "Ж"}, follow_redirects=True)
    assert "Профиль обновлён." in r.get_data(as_text=True)
    with app.app_context():
        u = db.session.scalars(db.select(User)).one()
        assert (u.email, u.first_name, u.last_name) == ("b@x.kz", "Бек", "Ж")
    r = client.post("/accounts/profile/", data={"email": "not-an-email"})
    assert "корректный адрес" in r.get_data(as_text=True)


# ------------------------------------------------------------------ analyzer
def test_upload_success_and_detail_page(app, client, monkeypatch):
    register(client, "boss")
    captured = {}

    def fake_post(url, data=None, headers=None, params=None, timeout=None):
        captured.update(url=url, headers=headers, params=params, timeout=timeout, size=len(data))
        return api_ok()

    monkeypatch.setattr(services.requests, "post", fake_post)
    r = upload(client)
    assert r.status_code == 302 and "/result/1/" in r.headers["Location"]

    assert captured["url"] == "http://vision.test:6769/analyze"
    assert captured["headers"] == {"Content-Type": "image/png"}
    assert captured["params"] == {"lang": "ru"}

    page = client.get("/result/1/").get_data(as_text=True)
    assert "Высокий" in page
    assert "Требует проверки человеком" in page
    assert "нечто похожее на нож" in page
    assert "Передать модератору." in page
    assert "STOP" in page and "Улица" in page
    assert "/media/uploads/" in page

    # изображение отдаётся владельцу
    with app.app_context():
        path = db.session.get(AnalysisResult, 1).image_path
    img = client.get(f"/media/{path}")
    assert img.status_code == 200 and img.mimetype == "image/png"


def test_upload_saves_error_when_backend_down(app, client, monkeypatch):
    register(client, "boss")

    def boom(*a, **k):
        raise requests.exceptions.ConnectionError()

    monkeypatch.setattr(services.requests, "post", boom)
    r = upload(client)
    page = client.get(r.headers["Location"]).get_data(as_text=True)
    assert "Ошибка анализа" in page
    assert "vision_analyzer_server.py запущен" in page
    with app.app_context():
        assert db.session.get(AnalysisResult, 1).error


def test_upload_http_error_and_timeout(app, client, monkeypatch):
    register(client, "boss")
    monkeypatch.setattr(services.requests, "post",
                        lambda *a, **k: FakeResponse({"error": "модель упала"}, status_code=500))
    page = client.get(upload(client).headers["Location"]).get_data(as_text=True)
    assert "Сервер вернул ошибку (500): модель упала" in page

    def slow(*a, **k):
        raise requests.exceptions.Timeout()

    monkeypatch.setattr(services.requests, "post", slow)
    page = client.get(upload(client).headers["Location"]).get_data(as_text=True)
    assert "не ответил вовремя" in page


def test_raw_fallback_is_shown(app, client, monkeypatch):
    register(client, "boss")
    monkeypatch.setattr(services.requests, "post", lambda *a, **k: api_ok({"_raw": "Просто текст от модели"}))
    page = client.get(upload(client).headers["Location"]).get_data(as_text=True)
    assert "Ответ модели (не JSON)" in page
    assert "Просто текст от модели" in page
    assert "Требует проверки человеком" in page
    with app.app_context():
        r = db.session.get(AnalysisResult, 1)
        assert r.risk_level == "unknown" and r.needs_human_review


def test_invalid_risk_level_becomes_unknown(app, client, monkeypatch):
    register(client, "boss")
    monkeypatch.setattr(services.requests, "post", lambda *a, **k: api_ok({"risk_level": "extreme", "description": "x"}))
    upload(client)
    with app.app_context():
        assert db.session.get(AnalysisResult, 1).risk_level == "unknown"


def test_non_image_and_empty_rejected(app, client, monkeypatch):
    register(client, "boss")
    monkeypatch.setattr(services.requests, "post", lambda *a, **k: pytest.fail("не должно вызываться"))
    r = upload(client, data=b"just text", name="fake.png")
    assert "не является изображением" in r.get_data(as_text=True)
    r = client.post("/", data={}, content_type="multipart/form-data")
    assert "Выберите файл изображения." in r.get_data(as_text=True)
    with app.app_context():
        assert db.session.scalar(db.select(db.func.count(AnalysisResult.id))) == 0


def test_file_too_large(app, client):
    app.config["MAX_CONTENT_LENGTH"] = 1024
    register(client, "boss")
    r = upload(client, data=make_png(size=(300, 300)), name="big.png")
    assert r.status_code == 302
    page = client.get("/").get_data(as_text=True)
    assert "Файл слишком большой" in page


def test_history_pagination_and_privacy(app, client, monkeypatch):
    register(client, "boss")
    logout(client)
    register(client, "alice")
    monkeypatch.setattr(services.requests, "post", lambda *a, **k: api_ok())
    for _ in range(13):
        upload(client)
    page1 = client.get("/history/").get_data(as_text=True)
    assert "Страница 1 из 2" in page1 and "Вперёд" in page1
    page2 = client.get("/history/?page=2").get_data(as_text=True)
    assert "Страница 2 из 2" in page2 and "Назад" in page2
    assert "Страница 2 из 2" in client.get("/history/?page=999").get_data(as_text=True)
    assert "Страница 1 из 2" in client.get("/history/?page=abc").get_data(as_text=True)

    # чужой результат недоступен обычному пользователю (и картинка тоже)
    logout(client)
    register(client, "bob")
    assert client.get("/result/1/").status_code == 404
    with app.app_context():
        path = db.session.get(AnalysisResult, 1).image_path
    assert client.get(f"/media/{path}").status_code == 404
    assert "История пуста" in client.get("/history/").get_data(as_text=True)

    # админ (boss) видит чужое
    logout(client)
    login(client, "boss")
    r = client.get("/result/1/")
    assert r.status_code == 200 and "пользователь alice" in r.get_data(as_text=True)
    assert client.get(f"/media/{path}").status_code == 200


def test_health_page(app, client, monkeypatch):
    register(client, "boss")
    payload = {"ok": True, "default_backend": "vllm",
               "backends": {"vllm": {"ok": True, "endpoint": "http://x:8000", "model": "qwen-vl"},
                            "ollama": {"ok": False, "endpoint": "http://y:11434", "error": "refused"}}}
    monkeypatch.setattr(services.requests, "get", lambda *a, **k: FakeResponse(payload))
    page = client.get("/health/").get_data(as_text=True)
    assert "В сети" in page and "qwen-vl" in page and "refused" in page and "Работает" in page

    def down(*a, **k):
        raise requests.exceptions.ConnectionError()

    monkeypatch.setattr(services.requests, "get", down)
    assert "Не удалось подключиться к серверу анализа изображений." in client.get("/health/").get_data(as_text=True)


# ------------------------------------------------------------------ roles / panel
def setup_three(client):
    register(client, "boss")
    logout(client)
    register(client, "adm")
    logout(client)
    register(client, "user1")
    logout(client)


def test_role_matrix(app, client):
    setup_three(client)
    with app.app_context():
        boss = db.session.scalars(db.select(User).where(User.username == "boss")).one()
        adm = db.session.scalars(db.select(User).where(User.username == "adm")).one()
        u1 = db.session.scalars(db.select(User).where(User.username == "user1")).one()
        adm.role = Role.ADMIN
        db.session.commit()
        assert boss.can_manage(adm) and boss.can_manage(u1) and not boss.can_manage(boss)
        assert adm.can_manage(u1) and not adm.can_manage(boss) and not adm.can_manage(adm)
        assert not u1.can_manage(adm)
        assert boss.assignable_roles() == ["blocked", "user", "admin", "head_admin"]
        assert adm.assignable_roles() == ["blocked", "user"]
        assert u1.assignable_roles() == []


def test_panel_access_rules(app, client):
    setup_three(client)
    with app.app_context():
        u = db.session.scalars(db.select(User).where(User.username == "adm")).one()
        u.role = Role.ADMIN
        db.session.commit()

    # обычный пользователь
    login(client, "user1")
    r = client.get("/panel/users/", follow_redirects=True)
    assert "Недостаточно прав" in r.get_data(as_text=True)
    assert "Панель управления" not in r.get_data(as_text=True)
    logout(client)

    # админ: видит пользователей, но не «Обзор»
    login(client, "adm")
    page = client.get("/panel/users/").get_data(as_text=True)
    assert "Панель управления" in page
    assert 'href="/panel/users/"' in page and 'href="/panel/analyses/"' in page
    assert 'href="/panel/"' not in page  # ссылки «Обзор» у обычного админа нет
    r = client.get("/panel/", follow_redirects=True)
    assert "Недостаточно прав" in r.get_data(as_text=True)
    logout(client)

    # главный админ
    login(client, "boss")
    page = client.get("/panel/").get_data(as_text=True)
    assert "Обзор системы" in page and "Пользователей всего" in page


def test_admin_can_block_user_but_not_admins(app, client):
    setup_three(client)
    with app.app_context():
        db.session.scalars(db.select(User).where(User.username == "adm")).one().role = Role.ADMIN
        db.session.commit()
    login(client, "adm")

    r = client.post(f"/panel/users/{uid(app, 'user1')}/toggle-block/", follow_redirects=True)
    assert "заблокирован" in r.get_data(as_text=True)
    assert role_of(app, "user1") == Role.BLOCKED

    r = client.post(f"/panel/users/{uid(app, 'user1')}/toggle-block/", follow_redirects=True)
    assert "разблокирован" in r.get_data(as_text=True)
    assert role_of(app, "user1") == Role.USER

    # нельзя трогать главного админа и себя
    r = client.post(f"/panel/users/{uid(app, 'boss')}/toggle-block/", follow_redirects=True)
    assert "нет прав" in r.get_data(as_text=True)
    assert role_of(app, "boss") == Role.HEAD_ADMIN
    client.post(f"/panel/users/{uid(app, 'adm')}/set-role/", data={"role": "user"})
    assert role_of(app, "adm") == Role.ADMIN

    # админ не может назначить admin
    client.post(f"/panel/users/{uid(app, 'user1')}/set-role/", data={"role": "admin"})
    assert role_of(app, "user1") == Role.USER


def test_head_admin_sets_roles(app, client):
    setup_three(client)
    login(client, "boss")
    r = client.post(f"/panel/users/{uid(app, 'user1')}/set-role/", data={"role": "admin"}, follow_redirects=True)
    assert "Пользователь → Администратор" in r.get_data(as_text=True)
    assert role_of(app, "user1") == Role.ADMIN
    r = client.post(f"/panel/users/{uid(app, 'user1')}/set-role/", data={"role": "nonsense"}, follow_redirects=True)
    assert "не можете назначить" in r.get_data(as_text=True)
    assert role_of(app, "user1") == Role.ADMIN
    # GET на POST-эндпоинт запрещён
    assert client.get(f"/panel/users/{uid(app, 'user1')}/set-role/").status_code == 405


def test_blocked_user_is_locked_out(app, client, monkeypatch):
    setup_three(client)
    login(client, "boss")
    client.post(f"/panel/users/{uid(app, 'user1')}/toggle-block/")
    logout(client)

    r = login(client, "user1")
    page = r.get_data(as_text=True)
    assert "Аккаунт заблокирован" in page and "Vision&nbsp;Triage" not in page
    for url in ["/", "/history/", "/health/", "/accounts/profile/", "/result/1/", "/panel/users/"]:
        resp = client.get(url)
        assert resp.status_code == 302 and resp.headers["Location"].endswith("/accounts/blocked/"), url
    # загрузка тоже недоступна
    monkeypatch.setattr(services.requests, "post", lambda *a, **k: pytest.fail("нельзя"))
    assert upload(client).status_code == 302
    # но выйти можно
    assert "Вы вышли из системы." in logout(client).get_data(as_text=True)


def test_user_search_and_filters(app, client, monkeypatch):
    setup_three(client)
    login(client, "boss")
    page = client.get("/panel/users/?q=USER").get_data(as_text=True)
    assert ">user1<" in page and ">adm<" not in page and ">boss<" not in page
    page = client.get("/panel/users/?role=head_admin").get_data(as_text=True)
    assert ">boss<" in page and ">user1<" not in page
    # спецсимволы LIKE экранируются
    page = client.get("/panel/users/?q=%25").get_data(as_text=True)
    assert "Ничего не найдено." in page

    monkeypatch.setattr(services.requests, "post", lambda *a, **k: api_ok())
    upload(client)
    monkeypatch.setattr(services.requests, "post", lambda *a, **k: api_ok({"risk_level": "low", "description": "ok"}))
    upload(client)
    page = client.get("/panel/analyses/?risk=high").get_data(as_text=True)
    assert page.count("risk-pill-high") == 1
    page = client.get("/panel/analyses/?review=1").get_data(as_text=True)
    assert page.count("<tr onclick") == 1
    page = client.get("/panel/analyses/").get_data(as_text=True)
    assert page.count("<tr onclick") == 2


def test_stats_counts(app, client, monkeypatch):
    setup_three(client)
    login(client, "boss")
    monkeypatch.setattr(services.requests, "post", lambda *a, **k: api_ok())
    upload(client); upload(client)
    monkeypatch.setattr(services.requests, "get", lambda *a, **k: FakeResponse({"ok": True, "default_backend": "vllm"}))
    page = client.get("/panel/").get_data(as_text=True)
    assert "stat-value\">3<" in page  # пользователей
    assert page.count("stat-value\">2<") == 3  # анализов, high, review


def test_csrf_enforced_when_enabled(tmp_path):
    from vision_app import create_app

    app = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path/'c.db'}",
                      "UPLOAD_FOLDER": str(tmp_path / "m"), "WTF_CSRF_ENABLED": True})
    c = app.test_client()
    r = c.post("/accounts/register/", data={"username": "x", "password1": "Str0ng-pass-42", "password2": "Str0ng-pass-42"},
               follow_redirects=True)
    assert "Сессия истекла" in r.get_data(as_text=True)
    with app.app_context():
        assert db.session.scalar(db.select(db.func.count(User.id))) == 0


def test_timezone_filter(app):
    from datetime import datetime
    from vision_app.utils import local_dt, truncate_chars

    with app.app_context():
        assert local_dt(datetime(2026, 9, 20, 6, 0), "%H:%M") == "11:00"  # UTC+5
        assert local_dt(None) == "—"
    assert truncate_chars("a" * 40, 28) == "a" * 27 + "…"
    assert truncate_chars("abc", 28) == "abc"


def test_404_page_and_cli(app, client):
    register(client, "boss")
    r = client.get("/nope/")
    assert r.status_code == 404 and "Страница не найдена" in r.get_data(as_text=True)
    runner = app.test_cli_runner()
    res = runner.invoke(args=["set-role", "boss", "user"])
    assert "Пользователь" in res.output and role_of(app, "boss") == Role.USER
    assert runner.invoke(args=["set-role", "ghost", "user"]).exit_code != 0
