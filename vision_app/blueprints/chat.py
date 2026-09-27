"""Чат со свободным диалогом с моделью (POST /chat на сервере анализа)."""

from __future__ import annotations

from flask import Blueprint, abort, current_app, jsonify, redirect, render_template, request, url_for
from flask_login import current_user, login_required
from sqlalchemy import select

from ..extensions import db
from ..models import ChatMessage, ChatRole, ChatSession
from ..services import VisionApiError, chat_with_model
from ..settings_store import get_analysis_target

bp = Blueprint("chat", __name__)

MAX_MESSAGE_LEN = 8000
MAX_HISTORY_MESSAGES = 80  # сколько последних сообщений сессии отправлять модели как контекст
TITLE_LEN = 60


def _own_sessions():
    return db.session.scalars(
        select(ChatSession)
        .where(ChatSession.user_id == current_user.id)
        .order_by(ChatSession.updated_at.desc())
    ).all()


def _get_own_session(session_id: int) -> ChatSession:
    session_row = db.session.get(ChatSession, session_id)
    if session_row is None or session_row.user_id != current_user.id:
        abort(404)
    return session_row


def _make_title(message: str) -> str:
    text = " ".join(message.split())  # схлопнуть переносы строк/лишние пробелы для заголовка
    if len(text) <= TITLE_LEN:
        return text
    return text[: TITLE_LEN - 1] + "…"


@bp.route("/")
@login_required
def index():
    """Без конкретного чата — открываем последний активный или заводим первый."""
    sessions = _own_sessions()
    if sessions:
        return redirect(url_for("chat.view", session_id=sessions[0].id))

    session_row = ChatSession(user_id=current_user.id)
    db.session.add(session_row)
    db.session.commit()
    return redirect(url_for("chat.view", session_id=session_row.id))


@bp.route("/<int:session_id>")
@login_required
def view(session_id: int):
    session_row = _get_own_session(session_id)
    sessions = _own_sessions()
    target_backend, target_model = get_analysis_target()
    return render_template(
        "chat/chat.html",
        session=session_row,
        sessions=sessions,
        messages=session_row.messages,
        target_backend=target_backend,
        target_model=target_model,
    )


@bp.route("/new", methods=["POST"])
@login_required
def new_session():
    # Не плодим пустые чаты — если уже есть чат без единого сообщения, ведём в него.
    existing_empty = db.session.scalars(
        select(ChatSession)
        .where(ChatSession.user_id == current_user.id, ~ChatSession.messages.any())
        .order_by(ChatSession.updated_at.desc())
    ).first()
    if existing_empty is not None:
        return redirect(url_for("chat.view", session_id=existing_empty.id))

    session_row = ChatSession(user_id=current_user.id)
    db.session.add(session_row)
    db.session.commit()
    return redirect(url_for("chat.view", session_id=session_row.id))


@bp.route("/<int:session_id>/delete", methods=["POST"])
@login_required
def delete_session(session_id: int):
    session_row = _get_own_session(session_id)
    db.session.delete(session_row)
    db.session.commit()

    next_row = db.session.scalars(
        select(ChatSession)
        .where(ChatSession.user_id == current_user.id)
        .order_by(ChatSession.updated_at.desc())
    ).first()
    if next_row is not None:
        return redirect(url_for("chat.view", session_id=next_row.id))
    return redirect(url_for("chat.index"))


@bp.route("/<int:session_id>/send", methods=["POST"])
@login_required
def send(session_id: int):
    session_row = _get_own_session(session_id)
    body = request.get_json(silent=True) or {}

    message = (body.get("message") or "").strip()
    if not message:
        return jsonify({"error": "Пустое сообщение."}), 400
    if len(message) > MAX_MESSAGE_LEN:
        return jsonify({"error": f"Сообщение слишком длинное (максимум {MAX_MESSAGE_LEN} символов)."}), 400

    # Контекст для модели — уже сохранённые сообщения ЭТОЙ сессии (без нового,
    # оно передаётся отдельным полем 'message', как ожидает inference/chat.py).
    prior = session_row.messages[-MAX_HISTORY_MESSAGES:]
    history = [{"role": m.role, "content": m.content} for m in prior]

    target_backend, target_model = get_analysis_target()

    try:
        outcome = chat_with_model(
            message,
            history=history,
            backend=target_backend,
            model=target_model,
            lang="ru",
        )
    except VisionApiError as exc:
        current_app.logger.warning("chat: ошибка сервера анализа: %s", exc)
        return jsonify({"error": str(exc)}), 502

    db.session.add(ChatMessage(session_id=session_row.id, role=ChatRole.USER, content=message))
    db.session.add(
        ChatMessage(
            session_id=session_row.id,
            role=ChatRole.ASSISTANT,
            content=outcome.reply,
            backend=outcome.backend,
            model=outcome.model,
        )
    )
    if not session_row.title:
        session_row.title = _make_title(message)
    db.session.commit()  # onupdate=utcnow сам обновит session_row.updated_at

    return jsonify(
        {
            "reply": outcome.reply,
            "backend": outcome.backend,
            "model": outcome.model,
            "title": session_row.title,
        }
    )
