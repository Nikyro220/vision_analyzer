"""Категории из чата (manage_category): чтение сразу, изменения только заявкой с подтверждением;
новая категория — выключенный черновик; права — как в панели; защита от устаревших данных и инъекций."""

import json
import os

import pytest
from cryptography.fernet import Fernet

from vision_app import create_app
from vision_app.chat_tools import runner
from vision_app.chat_tools.categories import manage_category
from vision_app.extensions import db
from vision_app.models import Category, ChatAction, ChatActionStatus, Role, User

PW = "pw-pw-pw-pw"
NEW = dict(
    name="drone_activity", title="Беспилотники", summary="Drone visible in the frame.",
    full="Flag drones.\n\nIgnore toy drones.", compact="Flag drones.",
)


@pytest.fixture()
def app(tmp_path):
    os.environ["VISION_CREDENTIALS_KEY"] = Fernet.generate_key().decode()
    app = create_app({
        "TESTING": True, "WTF_CSRF_ENABLED": False, "SECRET_KEY": "test",
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path}/t.db", "UPLOAD_FOLDER": str(tmp_path / "up"),
    })
    with app.app_context():
        db.create_all()
        for name, role in (("head", Role.HEAD_ADMIN), ("adm", Role.ADMIN), ("adm2", Role.ADMIN), ("alice", Role.USER)):
            u = User(username=name, role=role, email=f"{name}@mail.test")
            u.set_password(PW)
            db.session.add(u)
        for i, (name, active) in enumerate((("weapons", True), ("old_rules", False))):
            db.session.add(Category(
                name=name, title=name.title(), summary=f"{name} summary", full=f"{name} full rules",
                compact=f"{name} compact", position=i, is_active=active,
            ))
        db.session.commit()
    return app


def _u(name):
    return User.query.filter_by(username=name).one()


def _cat(name):
    return Category.query.filter_by(name=name).first()


def _login(app, name):
    c = app.test_client()
    c.post("/accounts/login/", data={"username": name, "password": PW})
    return c


def _run(who, **args):
    res = manage_category(_u(who), args)
    return json.loads(res.text), res.references


def _scripted(monkeypatch, *replies):
    it = iter(replies)

    class Out:
        backend, model = "b", "m"

        def __init__(self, reply):
            self.reply = reply

    monkeypatch.setattr(runner, "chat_with_model", lambda *a, **k: Out(next(it)))


def _call(tool, **args):
    return json.dumps({"tool": tool, "args": args})


def test_tool_exists_for_panel_staff_only(app):
    with app.test_request_context():
        for who, expected in (("alice", False), ("adm", True), ("head", True)):
            assert ("manage_category" in runner.allowed_tools(_u(who))) is expected
            assert ("manage_category" in runner.build_tool_system_prompt(_u(who), [])) is expected
        assert "manage_user" not in runner.allowed_tools(_u("adm"))  # пользователи — по-прежнему только главному
        assert "error" in _run("alice", action="list")[0]


def test_read_actions_run_immediately(app):
    with app.test_request_context():
        listing, refs = _run("adm", action="list")
        assert refs == [] and listing["total"] == 2
        assert {c["name"]: c["is_active"] for c in listing["categories"]} == {"weapons": True, "old_rules": False}
        got, _ = _run("adm", action="get", category="WEAPONS")  # имя без учёта регистра
        assert got["full"] == "weapons full rules" and got["is_active"] is True
        assert "error" in _run("adm", action="get", category="weap")[0]  # частичных совпадений нет
        assert ChatAction.query.count() == 0


def test_create_is_a_disabled_draft_made_only_on_confirm(app):
    with app.test_request_context():
        payload, refs = _run("adm", action="create", **NEW)
        assert payload["status"] == "awaiting_confirmation" and refs[0]["action"] == "cat_create"
        assert _cat("drone_activity") is None  # модель ничего не создала
        assert any("ВЫКЛЮЧЕННОЙ" in line for line in refs[0]["details"])
        assert any("Ignore toy drones." in line for line in refs[0]["details"])  # текст виден целиком
        again = _run("adm", action="create", **NEW)[1][0]
        assert again["id"] == refs[0]["id"]  # дубль не плодится
    reply = _login(app, "adm").post(refs[0]["confirm_url"]).get_json()
    assert reply["ok"], reply
    with app.app_context():
        row = _cat("drone_activity")
        assert row is not None and row.is_active is False and row.position == 2
        assert row.full == NEW["full"]


def test_create_validation(app):
    with app.test_request_context():
        assert "error" in _run("adm", action="create", **{**NEW, "name": "Плохое имя!"})[0]
        assert "error" in _run("adm", action="create", **{**NEW, "name": "weapons"})[0]  # занято
        assert "error" in _run("adm", action="create", **{**NEW, "summary": ""})[0]
        assert "error" in _run("adm", action="create", **{**NEW, "full": "x" * 5000})[0]  # слишком длинный
        assert "error" in _run("adm", action="create", **{k: v for k, v in NEW.items() if k != "name"})[0]
        assert ChatAction.query.count() == 0


def test_update_shows_before_after_and_applies(app):
    with app.test_request_context():
        payload, refs = _run("adm", action="update", category="weapons",
                             changes={"summary": "New summary", "position": 5, "name": "hacked"})
        assert payload["warnings"] and "name" in payload["warnings"][0]  # переименовать нельзя
        details = "\n".join(refs[0]["details"])
        assert "weapons summary → New summary" in details and "Позиция: 0 → 5" in details
        assert _cat("weapons").summary == "weapons summary"  # пока не подтверждено
        assert "error" in _run("adm", action="update", category="weapons", changes={"summary": "weapons summary"})[0]
        assert "error" in _run("adm", action="update", category="weapons", changes={})[0]
    assert _login(app, "adm").post(refs[0]["confirm_url"]).get_json()["ok"]
    with app.app_context():
        row = _cat("weapons")
        assert (row.summary, row.position, row.name) == ("New summary", 5, "weapons")


def test_update_refuses_to_overwrite_a_category_edited_meanwhile(app):
    with app.test_request_context():
        card = _run("adm", action="update", category="weapons", changes={"summary": "From chat"})[1][0]
        _cat("weapons").summary = "Edited in panel"  # правка в панели между заявкой и подтверждением
        db.session.commit()
    reply = _login(app, "adm").post(card["confirm_url"]).get_json()
    assert not reply["ok"] and "изменилась" in reply["message"]
    with app.app_context():
        assert _cat("weapons").summary == "Edited in panel"
        assert db.session.get(ChatAction, card["id"]).status == ChatActionStatus.FAILED


def test_enable_disable(app):
    with app.test_request_context():
        assert "error" in _run("adm", action="enable", category="weapons")[0]  # уже включена
        assert "error" in _run("adm", action="disable", category="old_rules")[0]  # уже выключена
        on = _run("adm", action="enable", category="old_rules")[1][0]
        off = _run("adm", action="disable", category="weapons")[1][0]
        assert _cat("old_rules").is_active is False and _cat("weapons").is_active is True
    c = _login(app, "adm")
    assert c.post(on["confirm_url"]).get_json()["ok"] and c.post(off["confirm_url"]).get_json()["ok"]
    with app.app_context():
        assert _cat("old_rules").is_active is True and _cat("weapons").is_active is False


def test_delete_needs_typed_name(app):
    with app.test_request_context():
        card = _run("adm", action="delete", category="old_rules")[1][0]
    assert card["danger"] and card["type_to_confirm"] == "old_rules"
    c = _login(app, "adm")
    assert not c.post(card["confirm_url"], json={"confirm_text": "nope"}).get_json()["ok"]
    with app.app_context():
        assert _cat("old_rules") is not None
        assert db.session.get(ChatAction, card["id"]).status == ChatActionStatus.PENDING
    assert c.post(card["confirm_url"], json={"confirm_text": "old_rules"}).get_json()["ok"]
    with app.app_context():
        assert _cat("old_rules") is None
        assert db.session.get(ChatAction, card["id"]).params["snapshot"]["full"] == "old_rules full rules"


def test_confirmation_rights(app):
    with app.test_request_context():
        card = _run("adm", action="disable", category="weapons")[1][0]
    assert _login(app, "alice").post(card["confirm_url"]).status_code == 403  # не администратор панели
    assert _login(app, "adm2").post(card["confirm_url"]).status_code == 404  # чужая заявка
    assert _login(app, "head").post(card["confirm_url"]).status_code == 404  # даже главному
    with app.app_context():
        assert _cat("weapons").is_active is True
    assert _login(app, "adm").post(card["confirm_url"]).get_json()["ok"]
    assert not _login(app, "adm").post(card["confirm_url"]).get_json()["ok"]  # повторно не выполняется


def test_rights_are_rechecked_on_confirm(app):
    with app.test_request_context():
        card = _run("adm", action="disable", category="weapons")[1][0]
        _u("adm").role = Role.USER  # понизили между заявкой и подтверждением
        db.session.commit()
    assert _login(app, "adm").post(card["confirm_url"]).status_code == 403
    with app.app_context():
        assert _cat("weapons").is_active is True


def test_changes_are_blocked_after_reading_foreign_data_but_reading_categories_is_fine(app, monkeypatch):
    for first in (_call("search_analyses", count_only=True), _call("search_users", count_only=True)):
        _scripted(monkeypatch, first, _call("manage_category", action="create", **NEW), "Нужна отдельная просьба.")
        with app.test_request_context():
            turn = runner.run_chat_turn(_u("head"), "посмотри и создай категорию", [], "b", "m")
            assert not [r for r in turn.references if r.get("kind") == "action"]
            assert ChatAction.query.count() == 0
    # get перед update в одном ходе — штатный сценарий
    _scripted(monkeypatch, _call("manage_category", action="get", category="weapons"),
              _call("manage_category", action="update", category="weapons", changes={"summary": "S2"}),
              "Подготовил — подтвердите в карточке ниже.")
    with app.test_request_context():
        turn = runner.run_chat_turn(_u("adm"), "уточни summary у weapons", [], "b", "m")
        assert [r["action"] for r in turn.references] == ["cat_update"]


def test_no_category_changes_in_service_turns(app, monkeypatch):
    _scripted(monkeypatch, _call("manage_category", action="disable", category="weapons"), "Не могу.")
    with app.test_request_context():
        turn = runner.run_chat_turn(_u("adm"), "[TOOL RESULT] ...", [], "b", "m", allow_actions=False)
        assert turn.references == [] and ChatAction.query.count() == 0


def test_phantom_card_claim_is_retried_for_category_tool(app, monkeypatch):
    _scripted(monkeypatch, "Подготовил заявку, подтвердите в карточке ниже.",
              _call("manage_category", action="disable", category="weapons"), "Подготовил — карточка ниже.")
    with app.test_request_context():
        turn = runner.run_chat_turn(_u("adm"), "выключи weapons", [], "b", "m")
        assert [r["action"] for r in turn.references] == ["cat_disable"]
        assert _cat("weapons").is_active is True


def test_tool_prompt_follows_the_unified_template(app):
    """Единый шаблон: английский текст, XML-секции <tools>/<tool>/<protocol>/<rules>, примеры вызова;
    блоки инструментов зависят от прав, а у обычного пользователя нет ни слова про email и админские тулы."""
    with app.test_request_context():
        head = runner.build_tool_system_prompt(_u("head"), [])
        adm = runner.build_tool_system_prompt(_u("adm"), [])
        user = runner.build_tool_system_prompt(_u("alice"), [])
    for prompt in (head, adm, user):
        for tag in ("<tools>", "</tools>", "<protocol>", "</protocol>", "<rules>", "</rules>"):
            assert tag in prompt
        assert prompt.count("<tool name=") == prompt.count("</tool>")
    assert head.count("<tool name=") == 6 and adm.count("<tool name=") == 5 and user.count("<tool name=") == 3
    assert "You have 6 tools" in head and "You have 3 tools" in user
    assert "email" not in user.lower() and "manage_category" not in user and "search_users" not in user
    assert '{"tool": "manage_user"' in head and '{"tool": "manage_user"' not in adm
