"""Чат со свободным диалогом с моделью (POST /chat на сервере анализа)."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
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
from sqlalchemy import func, or_, select, update

from .. import chat_images, chat_jobs
from ..chat_tools import ChatImage, attachment_note, delivery_message, run_chat_turn
from ..config import conf
from ..extensions import db
from ..models import ChatMessage, ChatRole, ChatSession, utcnow
from ..services import VisionApiError
from ..settings_store import get_analysis_target
from ..thumbs import ensure_thumb
from ..utils import local_dt

bp = Blueprint("chat", __name__)

# Лимиты чата (длина сообщения, глубина истории, длина названия, ...) — в config.py: CHAT_*.
DEFAULT_IMAGE_MESSAGE = "Опиши прикреплённое изображение."  # если пользователь приложил картинку без текста


def _aware(value: datetime) -> datetime:
    """SQLite отдаёт DateTime без часового пояса; в БД мы всегда кладём UTC."""
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


def _turn_busy(session_row: ChatSession) -> bool:
    """Модель сейчас отвечает на сообщение этого чата."""
    started = session_row.turn_started_at
    if started is None:
        return False
    return (utcnow() - _aware(started)).total_seconds() < conf("CHAT_TURN_STALE_SECONDS")


def _claim_turn(session_id: int) -> bool:
    """Атомарно помечает чат «модель отвечает». False — предыдущий ход ещё не закончен
    (например, сообщение отправили из двух вкладок сразу)."""
    now = utcnow()
    cutoff = now - timedelta(seconds=conf("CHAT_TURN_STALE_SECONDS"))
    claimed = db.session.execute(
        update(ChatSession)
        .where(
            ChatSession.id == session_id,
            or_(ChatSession.turn_started_at.is_(None), ChatSession.turn_started_at < cutoff),
        )
        .values(turn_started_at=now)
        .execution_options(synchronize_session=False)
    )
    db.session.commit()
    return claimed.rowcount == 1


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
    title_len = conf("CHAT_TITLE_LEN")
    if len(text) <= title_len:
        return text
    return text[: title_len - 1] + "…"


def _session_images(session_row: ChatSession) -> tuple[list[ChatImage], dict[int, list[ChatImage]]]:
    """Все вложения чата со сквозной нумерацией (#1, #2, …) и то же самое по сообщениям.

    Нумерация идёт по всем сообщениям сессии, а не только по последним CHAT_MAX_HISTORY_MESSAGES:
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
    limit = max(int(conf("CHAT_CONTEXT_IMAGES")), int(conf("CHAT_MAX_IMAGES_PER_MESSAGE")))
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
    prior = session_row.messages[-conf("CHAT_MAX_HISTORY_MESSAGES"):]
    history = []
    for m in prior:
        attached = by_message.get(m.id, [])
        entry = {"role": m.role, "content": (m.content or "") + attachment_note(attached)}
        urls = _data_urls(attached)
        if urls:
            entry["images"] = urls
        history.append(entry)
    return history


def _attachment_urls(session_row: ChatSession, messages=None) -> dict[int, list[dict]]:
    """message.id -> [{url, thumb_url, name}] для отрисовки вложений в шаблоне."""
    out: dict[int, list[dict]] = {}
    for m in session_row.messages if messages is None else messages:
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

    # Модель ещё отвечает (страницу перезагрузили посреди хода): сообщение пользователя уже в БД,
    # chat.js покажет «печатает…» и дождётся ответа через GET .../state. busy_since — id, после
    # которого искать сообщения этого хода (само сообщение пользователя и всё за ним).
    busy = _turn_busy(session_row)
    busy_since = 0
    if busy:
        last_user = next((m for m in reversed(session_row.messages) if m.role == ChatRole.USER), None)
        busy_since = last_user.id - 1 if last_user is not None else 0

    return render_template(
        "chat/chat.html",
        session=session_row,
        sessions=sessions,
        messages=session_row.messages,
        attachments=_attachment_urls(session_row),
        busy=busy,
        busy_since=busy_since,
        pending_analyses=chat_jobs.pending_count(session_row.id),
        max_images=conf("CHAT_MAX_IMAGES_PER_MESSAGE"),
        max_image_mb=conf("CHAT_MAX_IMAGE_BYTES") // (1024 * 1024),
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
            resp = send_file(thumb, mimetype="image/jpeg", max_age=conf("THUMB_CACHE_SECONDS"), conditional=True)
            resp.cache_control.public = False
            resp.cache_control.private = True
            return resp

    resp = send_file(path, mimetype=items[index].get("mime") or None, max_age=conf("CHAT_ATTACHMENT_CACHE_SECONDS"), conditional=True)
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
    message и images (0..CHAT_MAX_IMAGES_PER_MESSAGE файлов). Хотя бы одно из двух обязательно.

    Сообщение пользователя сохраняется в БД сразу, ДО обращения к модели, — так оно не пропадает,
    если страницу перезагрузили, пока модель отвечает. Если ход не удался, сообщение (и вложения)
    откатываются, а клиент получает ошибку и возвращает текст с картинками в поле ввода."""
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
    max_len = conf("CHAT_MAX_MESSAGE_LEN")
    if len(message) > max_len:
        return jsonify({"error": f"Сообщение слишком длинное (максимум {max_len} символов)."}), 400

    # Вложения сохраняем на диск ДО обращения к модели: инструмент analyze_image читает их
    # оттуда. Если ход не удался — файлы удаляются, в БД ничего не остаётся.
    try:
        saved = chat_images.validate_and_save(files)
    except chat_images.AttachmentError as exc:
        return jsonify({"error": str(exc)}), 400

    if not _claim_turn(session_row.id):
        _discard_attachments(session_row.id, saved)
        return jsonify({"error": "Модель ещё отвечает на предыдущее сообщение — дождитесь ответа."}), 409

    progress: dict = {}  # что успел сделать ход — нужно, чтобы аккуратно откатить его при сбое
    try:
        return _run_turn(session_row, message, saved, progress)
    except Exception:
        _abort_turn(session_row.id, progress, saved)
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


def _abort_turn(session_id: int, progress: dict, saved: list[dict]) -> None:
    """Откат неудавшегося хода: удаляем уже сохранённое сообщение пользователя, снимаем метку
    «модель отвечает», возвращаем чату пустой заголовок (если его дало это сообщение) и убираем
    вложения. Безопасно вызывать, даже если ход упал до сохранения сообщения."""
    try:
        db.session.rollback()
        message_id = progress.get("user_message_id")
        if message_id:
            message = db.session.get(ChatMessage, message_id)
            if message is not None:
                db.session.delete(message)
                db.session.flush()
        row = db.session.get(ChatSession, session_id)
        if row is not None:
            row.turn_started_at = None
            if progress.get("title_set"):
                left = db.session.scalar(
                    select(func.count(ChatMessage.id)).where(ChatMessage.session_id == session_id)
                )
                if not left:
                    row.title = ""
        db.session.commit()
    except Exception:  # noqa: BLE001 — откат не должен маскировать исходную ошибку
        db.session.rollback()
        current_app.logger.exception("chat: не удалось откатить неудавшийся ход")
    _discard_attachments(session_id, saved)


def _run_turn(session_row: ChatSession, message: str, saved: list[dict], progress: dict):
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
    # Собираем ДО сохранения нового сообщения ниже.
    history = _history_for_model(session_row, by_message)

    model_message = message or (DEFAULT_IMAGE_MESSAGE if new_images else "")
    model_message += attachment_note(new_images)
    # Картинки текущего сообщения уходят модели напрямую (POST /chat, поле images): на вопросы о них
    # она отвечает сама. В очередь анализа изображение попадает только по явной просьбе — через
    # инструмент analyze_image.
    message_images = _data_urls(new_images)

    target_backend, target_model = get_analysis_target()

    # Сообщение пользователя сохраняем сразу: после перезагрузки страницы оно на месте, а ответ
    # модели дорисуется, когда будет готов (GET .../state). При сбое хода оно откатывается (_abort_turn).
    user_message = ChatMessage(session_id=session_row.id, role=ChatRole.USER, content=message, images=saved)
    db.session.add(user_message)
    if not session_row.title:
        session_row.title = _make_title(message) if message else _make_title(f"Изображение: {saved[0]['name']}")
        progress["title_set"] = True
    db.session.commit()
    progress["user_message_id"] = user_message.id

    # Модель сама решает, нужны ли ей данные из истории анализов или (только если пользователь
    # явно попросил) постановка изображения в очередь анализа: при необходимости она присылает
    # JSON-вызов инструмента, мы его выполняем (с проверкой прав) и повторно вызываем модель —
    # см. chat_tools/runner.py. Результат анализа из очереди придёт позже, отдельным ходом (pending() ниже).
    try:
        turn = run_chat_turn(
            current_user, model_message, history, target_backend, target_model, lang=conf("DEFAULT_LANG"),
            images=all_images, session_id=session_row.id, message_images=message_images,
        )
    except VisionApiError as exc:
        current_app.logger.warning("chat: ошибка сервера анализа: %s", exc)
        _abort_turn(session_row.id, progress, saved)
        return jsonify({"error": str(exc)}), 502

    assistant_message = ChatMessage(
        session_id=session_row.id,
        role=ChatRole.ASSISTANT,
        content=turn.reply,
        backend=turn.backend,
        model=turn.model,
        refs=turn.references,
    )
    db.session.add(assistant_message)
    session_row.turn_started_at = None  # ответ и снятие метки — одним коммитом
    db.session.commit()  # onupdate=utcnow сам обновит session_row.updated_at

    return jsonify(
        {
            "reply": turn.reply,
            "backend": turn.backend,
            "model": turn.model,
            "title": session_row.title,
            "refs": turn.references,
            "pending": chat_jobs.pending_count(session_row.id),
            "user_message_id": progress["user_message_id"],
            "assistant_message_id": assistant_message.id,
        }
    )


def _message_payload(message: ChatMessage, attachments: dict[int, list[dict]]) -> dict:
    return {
        "id": message.id,
        "role": message.role,
        "content": message.content or "",
        "backend": message.backend or "",
        "model": message.model or "",
        "refs": message.refs or [],
        "images": attachments.get(message.id, []),
        "time": local_dt(message.created_at, "time"),
    }


@bp.route("/<int:session_id>/state")
@login_required
def state(session_id: int):
    """Опрос из chat.js, пока модель отвечает на сообщение (в том числе после перезагрузки страницы
    или обрыва соединения): идёт ли ход ещё и какие сообщения с id > since уже есть в чате.

    Если ход не удался, сообщение пользователя удалено на сервере — в ответе его уже нет, и
    страница возвращает текст в поле ввода."""
    session_row = _get_own_session(session_id)
    since = request.args.get("since", 0, type=int) or 0
    messages = db.session.scalars(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_row.id, ChatMessage.id > since)
        .order_by(ChatMessage.id)
    ).all()
    attachments = _attachment_urls(session_row, messages)
    response = jsonify(
        {
            "busy": _turn_busy(session_row),
            "title": session_row.title,
            "messages": [_message_payload(m, attachments) for m in messages],
            "pending": chat_jobs.pending_count(session_row.id),
        }
    )
    response.headers["Cache-Control"] = "no-store"
    return response


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
                current_user, delivery_message(payload), history, target_backend, target_model, lang=conf("DEFAULT_LANG"),
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
    return {"id": message.id, "reply": reply, "backend": backend, "model": model, "refs": references}
