"""Страница «Логи» (/panel/logs/): доступ только у head_admin, Flask-панель и сервер — раздельно."""

import os

import pytest
from cryptography.fernet import Fernet

from vision_app import create_app, log_viewer
from vision_app.extensions import db
from vision_app.models import Role, User

PASSWORD = "Sup3r-secret-pass"

APP_LOG = (
    "2026-10-14 05:00:01,100 [INFO] werkzeug: GET /panel/ 200\n"
    "2026-10-14 05:00:02,200 [WARNING] vision_app.services: backend slow\n"
    "2026-10-14 05:00:03,300 [ERROR] vision_app.queue_worker: job failed\n"
    "Traceback (most recent call last):\n"
    "  File \"x.py\", line 1, in <module>\n"
    "ValueError: boom\n"
    "2026-10-14 05:00:04,400 [INFO] vision_app: <script>alert(1)</script>\n"
)
SERVER_LOG = (
    "2026-10-14 05:00:01,000 [INFO] aiohttp.access: POST /analyze 200\n"
    "2026-10-14 05:00:02,000 [ERROR] inference.server: model crashed\n"
)


@pytest.fixture()
def app(tmp_path):
    os.environ["VISION_CREDENTIALS_KEY"] = Fernet.generate_key().decode()
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "app.log").write_text(APP_LOG, encoding="utf-8")
    (logs / "app.log.2026-10-13").write_text("2026-10-13 10:00:00,000 [INFO] old: yesterday-flask\n", encoding="utf-8")
    (logs / "analyzer.log").write_text(SERVER_LOG, encoding="utf-8")
    (logs / "secret.txt").write_text("do-not-leak", encoding="utf-8")

    app = create_app({
        "TESTING": True, "WTF_CSRF_ENABLED": False, "SECRET_KEY": "test", "LOG_DIR": str(logs),
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path}/t.db", "UPLOAD_FOLDER": str(tmp_path / "up"),
    })
    with app.app_context():
        db.create_all()
        for name, role in (("boss", Role.HEAD_ADMIN), ("adm", Role.ADMIN), ("usr", Role.USER)):
            u = User(username=name, role=role)
            u.set_password(PASSWORD)
            db.session.add(u)
        db.session.commit()
    return app


def _client(app, username=None):
    c = app.test_client()
    if username:
        c.post("/accounts/login/", data={"username": username, "password": PASSWORD})
    return c


def test_only_head_admin_can_open(app):
    assert _client(app, "boss").get("/panel/logs/").status_code == 200
    for who in ("adm", "usr", None):
        r = _client(app, who).get("/panel/logs/")
        assert r.status_code == 302
        assert b"job failed" not in r.data


def test_menu_item_only_for_head_admin(app):
    assert b'href="/panel/logs/"' in _client(app, "boss").get("/panel/").data
    assert b'href="/panel/logs/"' not in _client(app, "adm").get("/panel/users/").data


def test_flask_and_server_are_separate(app):
    c = _client(app, "boss")
    flask_page = c.get("/panel/logs/?source=flask").get_data(as_text=True)
    assert "job failed" in flask_page and "model crashed" not in flask_page
    server_page = c.get("/panel/logs/?source=server").get_data(as_text=True)
    assert "model crashed" in server_page and "job failed" not in server_page


def test_level_query_and_traceback_grouping(app):
    c = _client(app, "boss")
    page = c.get("/panel/logs/?level=ERROR").get_data(as_text=True)
    assert "job failed" in page and "ValueError: boom" in page  # traceback остаётся с записью
    assert "GET /panel/ 200" not in page and "backend slow" not in page
    page = c.get("/panel/logs/?q=SLOW").get_data(as_text=True)
    assert "backend slow" in page and "job failed" not in page


def test_html_in_logs_is_escaped(app):
    page = _client(app, "boss").get("/panel/logs/").get_data(as_text=True)
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in page


def test_cannot_read_foreign_files(app):
    c = _client(app, "boss")
    for bad in ("secret.txt", "../t.db", "../../etc/passwd", "analyzer.log"):  # analyzer.log — чужой источник
        page = c.get(f"/panel/logs/?source=flask&file={bad}").get_data(as_text=True)
        assert "do-not-leak" not in page and "model crashed" not in page
        assert "job failed" in page  # откатились к app.log


def test_rotated_file_and_fetch_fragment(app):
    c = _client(app, "boss")
    r = c.get("/panel/logs/?file=app.log.2026-10-13", headers={"X-Requested-With": "fetch"})
    assert r.is_json and "yesterday-flask" in r.get_json()["html"]
    assert "<html" not in r.get_json()["html"]


def test_missing_log_dir_is_not_an_error(app, tmp_path):
    app.config["LOG_DIR"] = str(tmp_path / "nope")
    assert _client(app, "boss").get("/panel/logs/").status_code == 200


def test_file_ordering_and_tail(tmp_path):
    for n in ("app.log", "app.error.log", "app.log.2026-10-12", "app.log.2026-10-13", "analyzer.log", "x.log"):
        (tmp_path / n).write_text("x", encoding="utf-8")
    names = [f.name for f in log_viewer.list_files(tmp_path, log_viewer.SOURCES["flask"])]
    assert names == ["app.log", "app.error.log", "app.log.2026-10-13", "app.log.2026-10-12"]

    big = tmp_path / "big.log"
    big.write_text("".join(f"2026-10-14 05:00:00,000 [INFO] n: line {i}\n" for i in range(5000)), encoding="utf-8")
    entries, truncated = log_viewer.read_entries(big, max_bytes=4096)
    assert truncated and entries[-1].text.endswith("line 4999")
    assert entries[0].level == "INFO"  # обрезанная первая строка отброшена
