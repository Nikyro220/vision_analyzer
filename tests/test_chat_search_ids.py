"""search_analyses(ids=[...]): показать анализы по номерам, права и склейка дублей не мешают."""

import json
import os

import pytest
from cryptography.fernet import Fernet

from vision_app import create_app
from vision_app.chat_tools.analyses import search_analyses
from vision_app.extensions import db
from vision_app.models import AnalysisResult, RiskLevel, Role, Status, User


@pytest.fixture()
def app(tmp_path):
    os.environ["VISION_CREDENTIALS_KEY"] = Fernet.generate_key().decode()
    app = create_app({
        "TESTING": True, "WTF_CSRF_ENABLED": False, "SECRET_KEY": "test",
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path}/t.db", "UPLOAD_FOLDER": str(tmp_path / "up"),
    })
    with app.test_request_context():
        db.create_all()
        for name, role in (("alice", Role.USER), ("bob", Role.USER)):
            u = User(username=name, role=role, email=f"{name}@mail.test")
            u.set_password("pw-pw-pw-pw")
            db.session.add(u)
        db.session.commit()
        alice, bob = (User.query.filter_by(username=n).one().id for n in ("alice", "bob"))
        for uid, h in ((alice, "same"), (alice, "same"), (bob, "other")):  # 1 и 2 — один и тот же файл
            db.session.add(AnalysisResult(
                user_id=uid, image_path="p/x.png", status=Status.DONE, risk_level=RiskLevel.LOW,
                image_hash=h, description="описание",
            ))
        db.session.commit()
    return app


def test_normalize_ids():
    from vision_app.chat_tools.analyses import _ids_arg
    assert _ids_arg([38, "#37", 0, "x", 38]) == [38, 37]
    assert _ids_arg("12, 13") == [12, 13]
    assert _ids_arg(5) == [5]
    assert _ids_arg(None) == [] and _ids_arg(True) == []


def test_ids_returns_exact_numbers_without_merging_duplicates(app):
    with app.test_request_context():
        alice = User.query.filter_by(username="alice").one()
        out = json.loads(search_analyses(alice, {"ids": [1, 2]}).text)
        assert sorted(r["id"] for r in out["records"]) == [1, 2]  # без склейки одного файла
        assert "duplicates_merged" not in out


def test_ids_respect_access_rights(app):
    with app.test_request_context():
        alice = User.query.filter_by(username="alice").one()
        out = json.loads(search_analyses(alice, {"ids": [3]}).text)  # №3 принадлежит bob
        assert out["total_matched"] == 0
