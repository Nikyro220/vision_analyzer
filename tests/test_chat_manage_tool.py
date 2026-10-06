"""Действия над пользователями из чата (manage_user): модель только готовит заявку, выполняет её
человек кнопкой; права — как в панели; защита от инъекций через данные."""

import json
import os
from datetime import timedelta

import pytest
from cryptography.fernet import Fernet

from vision_app import create_app
from vision_app.chat_tools import runner
from vision_app.chat_tools.analyses import search_analyses
from vision_app.chat_tools.manage import manage_user
from vision_app.extensions import db
from vision_app.models import (
    AnalysisResult, ChatAction, ChatActionStatus, Role, RiskLevel, Status, User, utcnow,
)

PW = "pw-pw-pw-pw"


@pytest.fixture()
def app(tmp_path):
    os.environ["VISION_CREDENTIALS_KEY"] = Fernet.generate_key().decode()
    app = create_app({
        "TESTING": True, "WTF_CSRF_ENABLED": False, "SECRET_KEY": "test",
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path}/t.db", "UPLOAD_FOLDER": str(tmp_path / "up"),
    })
    with app.app_context():
        db.create_all()
        for name, role in (
            ("head", Role.HEAD_ADMIN), ("head2", Role.HEAD_ADMIN), ("adm", Role.ADMIN),
            ("alice", Role.USER), ("bob", Role.USER), ("mallory", Role.BLOCKED),
        ):
            u = User(username=name, role=role, email=f"{name}@mail.test")
            u.set_password(PW)
            db.session.add(u)
        db.session.commit()
        for name, n in (("alice", 3), ("bob", 2)):
            uid = User.query.filter_by(username=name).one().id
            for i in range(n):
                db.session.add(AnalysisResult(
                    user_id=uid, image_path=f"p/{name}{i}.png", status=Status.DONE,
                    risk_level=RiskLevel.LOW, description=f"снимок {name} {i}",
                ))
        db.session.commit()
    return app


def _u(name):
    return User.query.filter_by(username=name).one()


def _login(app, name):
    c = app.test_client()
    c.post("/accounts/login/", data={"username": name, "password": PW})
    return c


def _propose(who, **args):
    res = manage_user(_u(who), args)
    return json.loads(res.text), res.references


def test_tool_exists_only_for_head_admin(app):
    call = json.dumps({"tool": "manage_user", "args": {"action": "block", "user": "bob"}})
    with app.test_request_context():
        for who in ("alice", "mallory", "adm"):
            assert "manage_user" not in runner.build_tool_system_prompt(_u(who), [])
            kind, reason = runner.parse_reply(call, runner.allowed_tools(_u(who)))
            assert kind == "bad" and "manage_user" not in reason
            assert "error" in _propose(who, action="block", user="bob")[0]
        head_prompt = runner.build_tool_system_prompt(_u("head"), [])
        assert "manage_user" in head_prompt and "инструмент ничего не выполняет" in head_prompt
        assert "  - user (строка или число)" in head_prompt  # просмотр анализов другого пользователя
        assert "  - user (строка или число)" not in runner.build_tool_system_prompt(_u("adm"), [])
        assert runner.parse_reply(call, runner.allowed_tools(_u("head")))[0] == "call"


def test_proposal_changes_nothing_until_confirmed(app):
    with app.test_request_context():
        payload, refs = _propose("head", action="block", user="bob")
        assert payload["status"] == "awaiting_confirmation" and "НИЧЕГО ЕЩЁ НЕ ИЗМЕНЕНО" in payload["note"]
        assert _u("bob").role == Role.USER  # модель ничего не выполнила
        card = refs[0]
        assert card["kind"] == "action" and card["status"] == "pending" and card["danger"] is True
        assert card["confirm_url"].endswith(f"/chat/actions/{card['id']}/confirm")
        assert _propose("head", action="block", user="bob")[1][0]["id"] == card["id"]  # дубль не плодится
        assert ChatAction.query.count() == 1


def test_confirm_and_cancel_over_http(app):
    with app.test_request_context():
        block = _propose("head", action="block", user="bob")[1][0]
        role = _propose("head", action="set_role", user="alice", role="admin")[1][0]
        cancel = _propose("head", action="unblock", user="mallory")[1][0]
    c = _login(app, "head")
    r = c.post(block["confirm_url"]).get_json()
    assert r["ok"] and r["card"]["status"] == "done"
    assert c.post(block["confirm_url"]).get_json()["ok"] is False  # повторный клик не выполняет дважды
    assert c.post(role["confirm_url"]).get_json()["ok"]
    assert c.post(cancel["cancel_url"]).get_json()["card"]["status"] == "cancelled"
    assert c.post(cancel["confirm_url"]).get_json()["ok"] is False
    with app.app_context():
        assert (_u("bob").role, _u("alice").role, _u("mallory").role) == (Role.BLOCKED, Role.ADMIN, Role.BLOCKED)
    assert c.get(block["state_url"]).get_json()["card"]["status"] == "done"


def test_only_owner_head_admin_can_confirm(app):
    with app.test_request_context():
        card = _propose("head", action="block", user="bob")[1][0]
    assert _login(app, "head2").post(card["confirm_url"]).status_code == 404  # чужая заявка
    assert _login(app, "adm").post(card["confirm_url"]).status_code == 403
    assert _login(app, "alice").post(card["confirm_url"]).status_code == 403
    assert app.test_client().post(card["confirm_url"]).status_code in (302, 401)
    with app.app_context():
        assert _u("bob").role == Role.USER


def test_delete_needs_typed_login(app):
    with app.test_request_context():
        card = _propose("head", action="delete", user="bob")[1][0]
        assert card["type_to_confirm"] == "bob"
    c = _login(app, "head")
    assert c.post(card["confirm_url"], json={"confirm_text": "BOB"}).get_json()["ok"] is False
    with app.app_context():
        assert User.query.filter_by(username="bob").count() == 1
    assert c.post(card["confirm_url"], json={"confirm_text": "bob"}).get_json()["ok"]
    with app.app_context():
        assert User.query.filter_by(username="bob").count() == 0
        assert AnalysisResult.query.filter_by(original_name="").count() == 3  # анализы bob ушли вместе с ним


def test_edit_validates_like_panel(app):
    with app.test_request_context():
        assert "error" in _propose("head", action="edit", user="bob", changes={"email": "не-почта"})[0]
        assert "error" in _propose("head", action="edit", user="bob", changes={"username": "alice"})[0]  # занят
        assert "error" in _propose("head", action="edit", user="bob", changes={"password_hash": "x"})[0]
        card = _propose("head", action="edit", user="bob", changes={"email": "new@example.com", "nickname": "Боб"})[1][0]
        assert any("new@example.com" in d for d in card["details"])
    assert _login(app, "head").post(card["confirm_url"]).get_json()["ok"]
    with app.app_context():
        assert (_u("bob").email, _u("bob").nickname) == ("new@example.com", "Боб")


def test_rules_match_panel(app):
    with app.test_request_context():
        assert "error" in _propose("head", action="block", user="head")[0]  # себя нельзя
        assert "error" in _propose("head", action="block", user="nobody")[0]
        assert "error" in _propose("head", action="block", user="mallory")[0]  # уже заблокирован
        assert "error" in _propose("head", action="unblock", user="bob")[0]
        assert "error" in _propose("head", action="set_role", user="bob", role="blocked")[0]
        assert "error" in _propose("head", action="set_role", user="bob", role="root")[0]
        assert "error" in _propose("head", action="set_role", user="bob", role="user")[0]  # уже user
        assert "error" in _propose("head", action="explode", user="bob")[0]
        assert ChatAction.query.count() == 0


def test_rechecks_rights_at_confirm_time(app):
    with app.test_request_context():
        card = _propose("head", action="set_role", user="alice", role="admin")[1][0]
    with app.app_context():
        _u("head").role = Role.ADMIN  # права отобрали после создания заявки
        db.session.commit()
    c = _login(app, "head")
    assert c.post(card["confirm_url"]).status_code == 403
    with app.app_context():
        assert _u("alice").role == Role.USER


def test_expired_and_pending_cap(app):
    with app.test_request_context():
        card = _propose("head", action="block", user="bob")[1][0]
        row = db.session.get(ChatAction, card["id"])
        row.expires_at = utcnow() - timedelta(minutes=1)
        db.session.commit()
    r = _login(app, "head").post(card["confirm_url"]).get_json()
    assert r["ok"] is False and r["card"]["status"] == "expired"
    with app.app_context():
        assert _u("bob").role == Role.USER
        app.config["CHAT_ACTION_MAX_PENDING"] = 1
    with app.test_request_context():
        assert _propose("head", action="block", user="alice")[0]["status"] == "awaiting_confirmation"  # просроченная не в счёт
        over = _propose("head", action="set_role", user="bob", role="admin")[0]
        assert "неподтверждённых" in over.get("error", "")


def _scripted(monkeypatch, *replies):
    it = iter(replies)

    class Out:
        backend, model = "b", "m"

        def __init__(self, reply):
            self.reply = reply

    monkeypatch.setattr(runner, "chat_with_model", lambda *a, **k: Out(next(it)))


def _call(tool, **args):
    return json.dumps({"tool": tool, "args": args})


def test_turn_collects_cards_for_several_actions(app, monkeypatch):
    _scripted(monkeypatch, _call("manage_user", action="block", user="bob"),
              _call("manage_user", action="block", user="alice"), "Подготовил две заявки.")
    with app.test_request_context():
        turn = runner.run_chat_turn(_u("head"), "заблокируй bob и alice", [], "b", "m")
        assert [r["kind"] for r in turn.references] == ["action", "action"]
        assert _u("bob").role == Role.USER and _u("alice").role == Role.USER


def test_no_actions_after_reading_data_in_same_turn(app, monkeypatch):
    """Данные из анализов/имён пользователей — чужой текст: после их чтения заявку создать нельзя."""
    for first in (_call("search_users", count_only=True), _call("search_analyses", count_only=True)):
        _scripted(monkeypatch, first, _call("manage_user", action="delete", user="bob"), "Нужна отдельная просьба.")
        with app.test_request_context():
            turn = runner.run_chat_turn(_u("head"), "посмотри и удали bob", [], "b", "m")
            assert not [r for r in turn.references if r.get("kind") == "action"]
            assert ChatAction.query.count() == 0


def test_no_actions_in_service_turns(app, monkeypatch):
    _scripted(monkeypatch, _call("manage_user", action="block", user="bob"), "Не могу.")
    with app.test_request_context():
        turn = runner.run_chat_turn(_u("head"), "[TOOL RESULT] ...", [], "b", "m", allow_actions=False)
        assert turn.references == [] and ChatAction.query.count() == 0


def test_head_admin_views_other_users_analyses(app):
    with app.test_request_context():
        res = json.loads(search_analyses(_u("head"), {"user": "bob"}).text)
        assert res["total_matched"] == 2 and "bob" in res["scope"]
        assert {r["user"] for r in res["records"]} == {"bob"}
        assert "error" in json.loads(search_analyses(_u("head"), {"user": "nobody"}).text)
        # обычный админ: аргумент отбрасывается, видит только своё (как и раньше)
        adm = json.loads(search_analyses(_u("adm"), {"user": "bob"}).text)
        assert adm["total_matched"] == 0 and any("user" in w for w in adm["warnings"])
