"""Инструмент чата search_users: недоступен обычным пользователям (в промпте, разборе и исполнении),
у админов — поиск, разбивка по ролям и статистика анализов."""

import json
import os
from datetime import timedelta

import pytest
from cryptography.fernet import Fernet

from vision_app import create_app
from vision_app.chat_tools import runner
from vision_app.chat_tools.users import search_users
from vision_app.extensions import db
from vision_app.models import AnalysisResult, Role, RiskLevel, Status, User, utcnow


@pytest.fixture()
def app(tmp_path):
    os.environ["VISION_CREDENTIALS_KEY"] = Fernet.generate_key().decode()
    app = create_app({
        "TESTING": True, "WTF_CSRF_ENABLED": False, "SECRET_KEY": "test",
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path}/t.db", "UPLOAD_FOLDER": str(tmp_path / "up"),
    })
    with app.test_request_context():
        db.create_all()
        for name, role in (
            ("head", Role.HEAD_ADMIN), ("adm", Role.ADMIN), ("alice", Role.USER),
            ("bob", Role.USER), ("carol_x", Role.USER), ("mallory", Role.BLOCKED),
        ):
            u = User(username=name, role=role, email=f"{name}@mail.test")
            u.set_password("pw-pw-pw-pw")
            db.session.add(u)
        db.session.commit()

        def add(name, n, risk=RiskLevel.LOW, age_days=0):
            uid = User.query.filter_by(username=name).one().id
            for i in range(n):
                db.session.add(AnalysisResult(
                    user_id=uid, image_path=f"p/{name}{i}.png", status=Status.DONE, risk_level=risk,
                    created_at=utcnow() - timedelta(days=age_days),
                ))

        add("alice", 5, RiskLevel.HIGH)
        add("bob", 2)
        add("bob", 7, age_days=40)  # давние: в недельное окно не попадают
        db.session.commit()
    return app


def _u(name):
    return User.query.filter_by(username=name).one()


def _call(who, **args):
    return json.loads(search_users(_u(who), args).text)


def test_prompt_hides_users_tool_from_regular_user(app):
    with app.test_request_context():
        for who in ("alice", "mallory"):
            prompt = runner.build_tool_system_prompt(_u(who), [])
            assert "search_users" not in prompt
            assert "email" not in prompt.lower()
        for who in ("adm", "head"):
            assert "search_users" in runner.build_tool_system_prompt(_u(who), [])


def test_regular_user_cannot_call_users_tool(app):
    call = json.dumps({"tool": "search_users", "args": {"count_only": True}})
    with app.test_request_context():
        alice, adm = _u("alice"), _u("adm")
        kind, reason = runner.parse_reply(call, runner.allowed_tools(alice))
        assert kind == "bad" and "search_users" not in reason  # имя скрытого инструмента не светим
        assert runner.parse_reply(call, runner.allowed_tools(adm))[0] == "call"
        # даже если вызов добрался до исполнения — отказ, данные не отдаются
        out = runner._execute(alice, runner.ToolCall("search_users", {}), [], None)
        assert "records" not in out.text and "error" in out.text
        # и сам инструмент проверяет права
        assert "error" in _call("alice")


def test_head_admin_sees_everyone_admin_sees_only_users(app):
    with app.test_request_context():
        head = _call("head", count_only=True)
        assert head["total_matched"] == 6
        assert head["by_role"]["Пользователь"] == 3 and head["by_role"]["Заблокирован"] == 1
        adm = _call("adm", count_only=True)
        assert adm["total_matched"] == 4  # без head и adm
        names = {r["username"] for r in _call("adm", limit=25)["records"]}
        assert not names & {"head", "adm"}
        assert _call("adm", role="head_admin")["total_matched"] == 0


def test_top_analyzers_and_period(app):
    with app.test_request_context():
        top = _call("head", sort="analyses", limit=3)["records"]
        assert [(r["username"], r["analyses_count"]) for r in top[:2]] == [("bob", 9), ("alice", 5)]
        assert top[1]["high_risk_count"] == 5 and top[0]["email"] == "bob@mail.test"
        week = _call("head", sort="analyses", analyses_days=7, limit=2)["records"]
        assert [(r["username"], r["analyses_count"]) for r in week] == [("alice", 5), ("bob", 2)]


def test_has_analyses_and_search(app):
    with app.test_request_context():
        idle = {r["username"] for r in _call("head", has_analyses=False, limit=25)["records"]}
        assert idle == {"head", "adm", "carol_x", "mallory"}
        assert _call("head", query="ALICE@mail")["total_matched"] == 1  # по email, без учёта регистра
        assert _call("head", username="bob")["total_matched"] == 1  # прежнее имя аргумента работает
        assert _call("head", query="%")["total_matched"] == 0  # % — не шаблон
        assert _call("head", query="carol_")["total_matched"] == 1
        assert _call("head", query="car_l")["total_matched"] == 0  # _ — не «любой символ»


def test_secrets_not_exposed(app):
    with app.test_request_context():
        text = search_users(_u("head"), {"limit": 25}).text
        assert "password" not in text.lower() and "pbkdf2" not in text and "scrypt" not in text


def test_cards_link_to_panel_user_page(app):
    with app.test_request_context():
        res = search_users(_u("head"), {"sort": "analyses", "limit": 2})
        assert [c["label"] for c in res.references] == ["bob", "alice"]
        card = res.references[0]
        assert card["kind"] == "user" and card["analyses_count"] == 9
        assert card["url"] == f"/panel/users/{_u('bob').id}/"
        assert card["role_label"] == "Пользователь" and card["initials"] and card["avatar_color"].startswith("#")
        assert json.loads(res.text)["cards_shown"] == 2
        assert search_users(_u("head"), {"count_only": True}).references == []  # «сколько» — без карточек
        # админ получает карточки только тех, кого видит
        labels = {c["label"] for c in search_users(_u("adm"), {"limit": 25}).references}
        assert labels == {"alice", "bob", "carol_x", "mallory"}


def test_card_links_open_for_staff_only(app):
    def login(name):
        c = app.test_client()
        c.post("/accounts/login/", data={"username": name, "password": "pw-pw-pw-pw"})
        return c

    with app.app_context():
        pk = _u("bob").id
    assert login("adm").get(f"/panel/users/{pk}/").status_code == 200
    assert login("alice").get(f"/panel/users/{pk}/").status_code in (302, 403, 404)


def test_turn_passes_cards_to_chat(app, monkeypatch):
    replies = iter([json.dumps({"tool": "search_users", "args": {"sort": "analyses", "limit": 2}}), "Лидирует bob."])

    class Out:
        backend, model = "b", "m"

        def __init__(self, reply):
            self.reply = reply

    monkeypatch.setattr(runner, "chat_with_model", lambda *a, **k: Out(next(replies)))
    with app.test_request_context():
        turn = runner.run_chat_turn(_u("head"), "кто чаще всех анализирует?", [], "b", "m")
        assert turn.reply == "Лидирует bob."
        assert [r["label"] for r in turn.references] == ["bob", "alice"]
        assert all(r["kind"] == "user" for r in turn.references)

    # обычному пользователю тот же вызов карточек не даёт
    replies2 = iter([json.dumps({"tool": "search_users", "args": {}}), "У меня нет таких данных."])
    monkeypatch.setattr(runner, "chat_with_model", lambda *a, **k: Out(next(replies2)))
    with app.test_request_context():
        turn = runner.run_chat_turn(_u("alice"), "покажи всех", [], "b", "m")
        assert turn.references == []
