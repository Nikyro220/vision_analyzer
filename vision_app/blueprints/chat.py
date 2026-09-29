"""Чат со свободным диалогом с моделью (POST /chat на сервере анализа)."""

from __future__ import annotations

import json
from pathlib import Path

from flask import (
    Blueprint,
    abort,
    current_app,
    jsonify,
    redirect,
    render_template,
    request,
    send_file,
    url_for,
)
from flask_login import current_user, login_required
from sqlalchemy import select

from .. import chat_images, chat_jobs
from ..chat_tools import ChatImage, attachment_note, delivery_message, run_chat_turn
from ..extensions import db
from ..models import ChatMessage, ChatRole, ChatSession, utcnow
from ..services import VisionApiError
from ..settings_store import get_analysis_target
from ..thumbs import ensure_thumb

bp = Blueprint("chat", __name__)

MAX_MESSAGE_LEN = 8000
MAX_HISTORY_MESSAGES = 80  # сколько последних сообщений сессии отправлять модели как контекст
TITLE_LEN = 60
DEFAULT_IMAGE_MESSAGE = "Опиши прикреплённое изображение."  # если пользователь приложил картинку без текста


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


def _session_images(session_row: ChatSession) -> tuple[list[ChatImage], dict[int, list[ChatImage]]]:
    """Все вложения чата со сквозной нумерацией (#1, #2, …) и то же самое по сообщениям.

    Нумерация идёт по всем сообщениям сессии, а не только по последним MAX_HISTORY_MESSAGES:
    номер, названный моделью в прошлых репликах, не должен «поехать» из-за обрезки истории.
    """
    everything: list[ChatImage] = []
    by_message: dict[int, list[ChatImage]] = {}
    for m in session_row.messages:
        for item in m.images or []:
            if not isinstance(item, dict) or not item.get("path"):
                continue
            entry = ChatImage(
                number=len(everything) + 1,
                path=item["path"],
                name=chat_images.clean_name(item.get("name", "")),
                mime=str(item.get("mime") or ""),
            )
            everything.append(entry)
            by_message.setdefault(m.id, []).append(entry)
    return everything, by_message


def _apply_context_limit(all_images: list[ChatImage]) -> None:
    """Модели напрямую передаются только последние CHAT_CONTEXT_IMAGES вложений чата — каждая
    картинка занимает порядка 1–1.5 тыс. токенов контекста. У более старых in_context=False:
    в чате они остаются и их по-прежнему можно поставить в очередь анализа, но модель их не видит
    (и знает об этом из промпта). Вложения текущего сообщения всегда в контексте."""
    limit = max(int(current_app.config.get("CHAT_CONTEXT_IMAGES", 6)), chat_images.MAX_IMAGES_PER_MESSAGE)
    for img in all_images[:-limit]:
        img.in_context = False


def _data_urls(images: list[ChatImage]) -> list[str]:
    """Картинки для поля `images` POST /chat (недоступные на диске молча пропускаются)."""
    urls = (chat_images.to_data_url(img.path, img.mime) for img in images if img.in_context)
    return [u for u in urls if u]


def _history_for_model(session_row: ChatSession, by_message: dict[int, list[ChatImage]]) -> list[dict]:
    """Последние сообщения сессии как контекст для модели. К сообщениям с вложениями дописана
    пометка «[Прикреплено изображение: #N «имя»]» (в БД её нет), а сами картинки, попавшие в
    лимит контекста (см. _apply_context_limit), приложены в поле images."""
    prior = session_row.messages[-MAX_HISTORY_MESSAGES:]
    history = []
    for m in prior:
        attached = by_message.get(m.id, [])
        entry = {"role": m.role, "content": (m.content or "") + attachment_note(attached)}
        urls = _data_urls(attached)
        if urls:
            entry["images"] = urls
        history.append(entry)
    return history


def _attachment_urls(session_row: ChatSession) -> dict[int, list[dict]]:
    """message.id -> [{url, thumb_url, name}] для отрисовки вложений в шаблоне."""
    out: dict[int, list[dict]] = {}
    for m in session_row.messages:
        for idx, item in enumerate(m.images or []):
            if not isinstance(item, dict) or not item.get("path"):
                continue
            url = url_for("chat.attachment", session_id=session_row.id, message_id=m.id, index=idx)
            out.setdefault(m.id, []).append(
                {"url": url, "thumb_url": url + "?thumb=1", "name": chat_images.clean_name(item.get("name", ""))}
            )
    return out


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
        attachments=_attachment_urls(session_row),
        pending_analyses=chat_jobs.pending_count(session_row.id),
        max_images=chat_images.MAX_IMAGES_PER_MESSAGE,
        max_image_mb=chat_images.MAX_IMAGE_BYTES // (1024 * 1024),
        target_backend=target_backend,
        target_model=target_model,
    )


@bp.route("/<int:session_id>/attachment/<int:message_id>/<int:index>")
@login_required
def attachment(session_id: int, message_id: int, index: int):
    """Отдаёт вложение сообщения (?thumb=1 — миниатюра). Только владельцу чата."""
    session_row = _get_own_session(session_id)
    message = db.session.get(ChatMessage, message_id)
    if message is None or message.session_id != session_row.id:
        abort(404)
    items = message.images or []
    if index >= len(items) or not isinstance(items[index], dict):
        abort(404)
    rel = items[index].get("path") or ""

    path = chat_images.resolve_path(rel)
    if path is None:
        abort(404)

    if request.args.get("thumb"):
        thumb = ensure_thumb(Path(current_app.config["UPLOAD_FOLDER"]), rel)
        if thumb is not None:
            resp = send_file(thumb, mimetype="image/jpeg", max_age=7 * 24 * 3600, conditional=True)
            resp.cache_control.public = False
            resp.cache_control.private = True
            return resp

    resp = send_file(path, mimetype=items[index].get("mime") or None, max_age=24 * 3600, conditional=True)
    resp.cache_control.public = False
    resp.cache_control.private = True
    return resp


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
    attachment_paths = chat_images.paths_of(session_row.messages)
    db.session.delete(session_row)
    db.session.commit()
    chat_images.remove_files(attachment_paths)  # файлы — только после успешного коммита

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
    """Принимает сообщение: JSON {"message": ...} либо multipart/form-data с полями
    message и images (0..MAX_IMAGES_PER_MESSAGE файлов). Хотя бы одно из двух обязательно."""
    session_row = _get_own_session(session_id)

    if request.is_json:
        body = request.get_json(silent=True) or {}
        message = (body.get("message") or "").strip() if isinstance(body.get("message"), str) else ""
        files = []
    else:
        message = (request.form.get("message") or "").strip()
        files = [f for f in request.files.getlist("images") if f and f.filename]

    if not message and not files:
        return jsonify({"error": "Пустое сообщение."}), 400
    if len(message) > MAX_MESSAGE_LEN:
        return jsonify({"error": f"Сообщение слишком длинное (максимум {MAX_MESSAGE_LEN} символов)."}), 400

    # Вложения сохраняем на диск ДО обращения к модели: инструмент analyze_image читает их
    # оттуда. Если ход не удался — файлы удаляются, в БД ничего не попадает.
    try:
        saved = chat_images.validate_and_save(files)
    except chat_images.AttachmentError as exc:
        return jsonify({"error": str(exc)}), 400

    try:
        return _run_turn(session_row, message, saved)
    except Exception:
        db.session.rollback()
        _discard_attachments(session_row.id, saved)
        raise


def _discard_attachments(session_id: int, saved: list[dict]) -> None:
    """Ход не удался — сообщение не сохранено: убираем вложения и всё, что инструмент успел
    поставить в очередь анализа из-за них."""
    paths = [item["path"] for item in saved]
    try:
        chat_jobs.cancel_for_paths(session_id, paths)
    except Exception:  # noqa: BLE001 — откат не должен маскировать исходную ошибку
        db.session.rollback()
        current_app.logger.exception("chat: не удалось откатить задачи анализа")
    chat_images.remove_files(paths)


def _run_turn(session_row: ChatSession, message: str, saved: list[dict]):
    all_images, by_message = _session_images(session_row)
    new_images = [
        ChatImage(
            number=len(all_images) + i + 1,
            path=item["path"],
            name=item["name"],
            mime=item["mime"],
        )
        for i, item in enumerate(saved)
    ]
    all_images = all_images + new_images
    _apply_context_limit(all_images)

    # Контекст для модели — уже сохранённые сообщения ЭТОЙ сессии (без нового,
    # оно передаётся отдельным полем 'message', как ожидает inference/chat.py).
    history = _history_for_model(session_row, by_message)

    model_message = message or (DEFAULT_IMAGE_MESSAGE if new_images else "")
    model_message += attachment_note(new_images)
    # Картинки текущего сообщения уходят модели напрямую (POST /chat, поле images): на вопросы о них
    # она отвечает сама. В очередь анализа изображение попадает только по явной просьбе — через
    # инструмент analyze_image.
    message_images = _data_urls(new_images)

    target_backend, target_model = get_analysis_target()

    # Модель сама решает, нужны ли ей данные из истории анализов или (только если пользователь
    # явно попросил) постановка изображения в очередь анализа: при необходимости она присылает
    # JSON-вызов инструмента, мы его выполняем (с проверкой прав) и повторно вызываем модель —
    # см. chat_tools/runner.py. Результат анализа из очереди придёт позже, отдельным ходом (pending() ниже).
    try:
        turn = run_chat_turn(
            current_user, model_message, history, target_backend, target_model, lang="ru",
            images=all_images, session_id=session_row.id, message_images=message_images,
        )
    except VisionApiError as exc:
        current_app.logger.warning("chat: ошибка сервера анализа: %s", exc)
        _discard_attachments(session_row.id, saved)
        return jsonify({"error": str(exc)}), 502

    user_message = ChatMessage(session_id=session_row.id, role=ChatRole.USER, content=message, images=saved)
    db.session.add(user_message)
    db.session.add(
        ChatMessage(
            session_id=session_row.id,
            role=ChatRole.ASSISTANT,
            content=turn.reply,
            backend=turn.backend,
            model=turn.model,
            refs=turn.references,
        )
    )
    if not session_row.title:
        session_row.title = _make_title(message) if message else _make_title(f"Изображение: {saved[0]['name']}")
    db.session.commit()  # onupdate=utcnow сам обновит session_row.updated_at

    return jsonify(
        {
            "reply": turn.reply,
            "backend": turn.backend,
            "model": turn.model,
            "title": session_row.title,
            "refs": turn.references,
            "pending": chat_jobs.pending_count(session_row.id),
        }
    )


@bp.route("/<int:session_id>/pending")
@login_required
def pending(session_id: int):
    """Опрос из chat.js, пока в чате есть изображения, ждущие анализа в очереди.

    Для каждой задачи, анализ которой завершился, модель получает результат, а её ответ
    сохраняется в чат и возвращается здесь (по одному сообщению на изображение). Каждая
    задача доставляется ровно один раз (chat_jobs.claim_ready).
    """
    session_row = _get_own_session(session_id)
    delivered = [_deliver(session_row, job, row) for job, row in chat_jobs.claim_ready(session_row.id)]
    response = jsonify({"messages": delivered, "pending": chat_jobs.pending_count(session_row.id)})
    response.headers["Cache-Control"] = "no-store"
    return response


def _deliver(session_row: ChatSession, job, row) -> dict:
    """Передаёт модели готовый результат анализа и сохраняет её ответ в чат.
    Если модель недоступна — сохраняет короткое сообщение без неё, чтобы результат не потерялся."""
    reply, backend, model, references = "", "", "", []
    if row is not None:
        references = [chat_jobs.card(row)]

    if row is not None:
        all_images, by_message = _session_images(session_row)
        _apply_context_limit(all_images)
        history = _history_for_model(session_row, by_message)
        payload = json.dumps(chat_jobs.status_payload(row, job), ensure_ascii=False)
        target_backend, target_model = get_analysis_target()
        try:
            turn = run_chat_turn(
                current_user, delivery_message(payload), history, target_backend, target_model, lang="ru",
                images=all_images, session_id=session_row.id,
            )
            reply, backend, model = turn.reply, turn.backend, turn.model
            references = references + [r for r in turn.references if r.get("id") != row.id]
        except VisionApiError as exc:
            current_app.logger.warning("chat: не удалось передать результат анализа модели: %s", exc)
        except Exception:  # noqa: BLE001 — результат уже помечен доставленным, теряться нельзя
            current_app.logger.exception("chat: сбой при доставке результата анализа")
    if not reply:
        reply = chat_jobs.fallback_text(job, row)

    message = ChatMessage(
        session_id=session_row.id,
        role=ChatRole.ASSISTANT,
        content=reply,
        backend=backend,
        model=model,
        refs=references,
    )
    db.session.add(message)
    session_row.updated_at = utcnow()  # чат поднимается в списке: пришёл результат
    db.session.commit()
    return {"reply": reply, "backend": backend, "model": model, "refs": references}
