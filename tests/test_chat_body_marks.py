"""Метки на теле (татуировки, шрамы, пирсинг): поиск по body_marks из raw_report и честный счётчик
анализов, про которые «есть ли метка» сказать нельзя (старые анализы, кожа не видна)."""

import json
import os

import pytest
from cryptography.fernet import Fernet

from vision_app import create_app
from vision_app.chat_tools.analyses import search_analyses
from vision_app.extensions import db
from vision_app.models import AnalysisResult, Role, RiskLevel, Status, User

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
        u = User(username="head", role=Role.HEAD_ADMIN, email="head@mail.test")
        u.set_password(PW)
        db.session.add(u)
        db.session.commit()
        yield app


def _add(user, name, description, raw_report):
    db.session.add(AnalysisResult(
        user_id=user.id, image_path=f"uploads/{name}.png", original_name=f"{name}.png",
        risk_level=RiskLevel.LOW, description=description, raw_report=raw_report, status=Status.DONE,
    ))


def _seed(app, with_new_only=False):
    user = db.session.scalar(db.select(User).where(User.username == "head"))
    if not with_new_only:
        _add(user, "old", "Мужчина в чёрной куртке.", {"description": "Мужчина в чёрной куртке."})
        _add(user, "hidden", "Женщина в пальто, руки в карманах.", {"skin_visible": False, "body_marks": []})
    # татуировки в description нет — она есть только в body_marks
    _add(user, "tattoo", "Мужчина держит телефон.", {
        "skin_visible": True, "body_marks": ["правое предплечье: чёрная татуировка, три слова, около 6 см"],
    })
    _add(user, "clean", "Мужчина у воды.", {"skin_visible": True, "body_marks": []})
    db.session.commit()
    return user


def _search(app, user, **args):
    with app.test_request_context():  # карточки-ссылки строятся через url_for
        return json.loads(search_analyses(user, args).text)


def test_keywords_find_tattoo_that_exists_only_in_body_marks(app):
    user = _seed(app)
    payload = _search(app, user, keywords=["татуировка", "тату", "наколка"])
    assert payload["total_matched"] == 1
    rec = payload["records"][0]
    assert "татуировк" not in rec["description"]  # в description её нет — нашлось именно через body_marks
    assert "татуировка" in rec["body_marks"][0]    # и модель должна её увидеть в записи


def test_gap_counter_counts_old_and_hidden_skin(app):
    user = _seed(app)
    payload = _search(app, user, keywords=["татуировка"])
    assert payload["analyses_in_scope"] == 4
    assert payload["body_marks_unknown"] == 2  # старый анализ без ключа + кожа не видна
    assert "does NOT" in payload["body_marks_note"]


def test_gap_counter_in_count_only_and_empty_result(app):
    user = _seed(app)
    payload = _search(app, user, keywords=["пирсинг"], count_only=True)
    assert payload["total_matched"] == 0
    assert payload["body_marks_unknown"] == 2  # «ноль» не равен «нет», и модель об этом предупреждена


def test_no_gap_fields_when_every_analysis_has_data(app):
    user = _seed(app, with_new_only=True)
    payload = _search(app, user, keywords=["татуировка"])
    assert payload["total_matched"] == 1
    assert "body_marks_unknown" not in payload and "body_marks_note" not in payload


def test_malformed_report_counts_as_unknown(app):
    user = _seed(app, with_new_only=True)
    _add(user, "bad", "Что-то.", {"skin_visible": "true", "body_marks": 42})
    _add(user, "raw", "сырой текст", {"_raw": "garbage"})
    db.session.commit()
    payload = _search(app, user, count_only=True)
    assert payload["body_marks_unknown"] == 2
    assert payload["analyses_in_scope"] == 4
