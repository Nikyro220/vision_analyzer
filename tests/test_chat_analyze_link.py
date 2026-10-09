"""analyze_image со ссылкой (аргумент url): задача-ссылка в общей очереди, без вложений в чате."""

import io
import json
import os
from unittest import mock

import pytest
from cryptography.fernet import Fernet
from PIL import Image

from vision_app import chat_jobs, create_app, queue_worker
from vision_app.chat_tools import runner
from vision_app.chat_tools.images import ChatImage, analyze_chat_image
from vision_app.extensions import db
from vision_app.models import AnalysisResult, ChatAnalysisJob, ChatSession, Role, Status, User
from vision_app.services import AnalysisOutcome, LinkItem, LinkOutcome

URL = "https://example.com/p/1"


def _png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(buf, "PNG")
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
        user = User(username="alice", role=Role.USER, email="alice@mail.test")
        user.set_password("pw-pw-pw-pw")
        db.session.add(user)
        db.session.flush()
        db.session.add(ChatSession(user_id=user.id))
        db.session.commit()
    return app


def _ctx():
    return User.query.filter_by(username="alice").one(), ChatSession.query.one().id


def _call(args, images=None):
    user, sid = _ctx()
    return json.loads(analyze_chat_image(user, args, images or [], sid).text)


def test_link_is_queued_without_attachments(app):
    with app.app_context():
        payload = _call({"url": URL})
        assert payload["status"] == "queued" and payload["url"] == URL and "image" not in payload
        row = db.session.get(AnalysisResult, payload["analysis_id"])
        assert row.source_url == URL and row.image_path == "" and row.status == Status.QUEUED
        job = ChatAnalysisJob.query.one()
        assert chat_jobs.is_link_job(job) and job.analysis_id == row.id and not job.delivered


def test_same_link_is_queued_once(app):
    with app.app_context():
        first = _call({"url": URL})
        second = _call({"url": URL})
        assert second["analysis_id"] == first["analysis_id"] and "уже стоит" in second["note"]
        assert AnalysisResult.query.count() == 1 and ChatAnalysisJob.query.count() == 1


def test_bare_link_gets_scheme(app):
    with app.app_context():
        assert _call({"url": "t.me/chan/5"})["url"] == "https://t.me/chan/5"


@pytest.mark.parametrize("bad", ["ftp://x.com/a", "javascript:alert(1)", "not a link", f"{URL} https://example.com/p/2"])
def test_bad_links_rejected(app, bad):
    with app.app_context():
        assert "error" in _call({"url": bad})
        assert AnalysisResult.query.count() == 0 and ChatAnalysisJob.query.count() == 0


def test_image_and_url_together_rejected(app):
    with app.app_context():
        img = ChatImage(number=1, path="chat_uploads/x.png", name="x.png", mime="image/png")
        assert "либо image, либо url" in _call({"image": 1, "url": URL}, [img])["error"]
        assert AnalysisResult.query.count() == 0


def test_no_images_and_no_url_points_to_url(app):
    with app.app_context():
        assert "url" in _call({})["error"]


def test_caption_ignored_for_link_with_warning(app):
    with app.app_context():
        payload = _call({"url": URL, "caption": "камера у входа"})
        assert any("caption" in w for w in payload["warnings"])
        assert AnalysisResult.query.one().caption == ""


def test_queue_limit_applies(app):
    with app.app_context():
        with mock.patch("vision_app.queue_worker.queue_limit_hit", lambda *a, **k: ("user", 5, 5)):
            assert "заполнена" in _call({"url": URL})["error"]
        assert AnalysisResult.query.count() == 0


def test_tool_is_in_prompt_without_attachments(app):
    with app.test_request_context():
        user = User.query.filter_by(username="alice").one()
        prompt = runner.build_tool_system_prompt(user, [])
    assert '<tool name="analyze_image">' in prompt and "url (string)" in prompt
    assert '{"tool": "analyze_image", "args": {"url"' in prompt
    assert "[Прикреплено изображение" not in prompt  # правила про вложения — только когда они есть


def test_link_result_is_delivered_to_chat(app):
    with app.app_context():
        _call({"url": URL})
        user, sid = _ctx()

        item = LinkItem(
            name=URL, image_bytes=_png(), mime="image/png", caption="Автор: bob",
            outcome=AnalysisOutcome(
                backend="ollama", risk_level="low", needs_human_review=False,
                description="Кот на диване", raw_report={"risk_level": "low"}, is_raw_fallback=False,
            ),
        )
        with mock.patch.object(queue_worker, "analyze_link", lambda *a, **k: LinkOutcome(items=[item])), \
             mock.patch.object(queue_worker.vector_search, "index_analysis", lambda _id: None):
            assert queue_worker.drain(app) == 1

        ready = chat_jobs.claim_ready(sid)
        assert len(ready) == 1
        job, row = ready[0]
        payload = chat_jobs.status_payload(row, job)
        assert payload["status"] == "done" and payload["url"] == URL and payload["risk_level"] == "low"
        assert "ссылки" in chat_jobs.fallback_text(job, row)
        assert chat_jobs.claim_ready(sid) == []  # доставлено ровно один раз


def test_model_call_with_url_in_chat_without_images(app, monkeypatch):
    """Ход чата целиком: модель шлёт JSON-вызов с url, получает «в очереди» и отвечает текстом."""
    replies = iter([
        json.dumps({"tool": "analyze_image", "args": {"url": URL}}),
        "Ссылка поставлена в очередь, результат придёт сюда.",
    ])
    seen = []

    class Out:
        backend, model = "b", "m"

        def __init__(self, reply):
            self.reply = reply

    def fake(message, *a, **k):
        seen.append(message)
        return Out(next(replies))

    monkeypatch.setattr(runner, "chat_with_model", fake)
    with app.test_request_context():
        user, sid = _ctx()
        turn = runner.run_chat_turn(user, f"поставь в анализ {URL}", [], "b", "m", images=[], session_id=sid)
        assert turn.reply.startswith("Ссылка поставлена")
        assert "[TOOL RESULT] analyze_image" in seen[1] and '"status": "queued"' in seen[1]
        assert AnalysisResult.query.one().source_url == URL
