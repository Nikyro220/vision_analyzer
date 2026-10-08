"""Анализ изображений из чата через общую очередь.

Инструмент analyze_image (chat_tools/images.py) ничего не анализирует сам — он только
ставит вложение чата в ту же очередь, что и загрузка на странице «Анализ»:

  1. enqueue()      — копирует файл в uploads/, создаёт AnalysisResult(status=queued) и
                      ChatAnalysisJob, будит обработчик очереди. Модель получает статус «в очереди».
  2. Обработчик очереди (queue_worker.py) анализирует изображение как обычно, по одному,
     и сохраняет результат в AnalysisResult — он же попадает в историю анализов.
  3. claim_ready()  — когда анализ завершён, чат забирает результат (blueprints/chat.py:
                      GET .../pending), передаёт его модели и добавляет её ответ в переписку.

Здесь только слой данных; обращение к модели — в blueprints/chat.py, чтобы не было
циклического импорта chat_tools <-> chat_jobs.
"""

from __future__ import annotations

import logging
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import current_app, url_for
from sqlalchemy import delete, func, select, update

from . import chat_images, image_dedup
from .config import Config, conf
from .extensions import db
from .history import remove_image_files
from .models import RISK_LABELS, AnalysisResult, ChatAnalysisJob, Status
from .utils import local_dt

log = logging.getLogger("vision_app.chat_jobs")

# Длины подписи и полей отчёта, которые уходят модели, — в config.py (CAPTION_MAX_CHARS, CHAT_JOB_*).


class JobError(Exception):
    """Не удалось поставить изображение в очередь; текст можно отдать модели/пользователю."""


def _clip(value, limit: int | None = None) -> str:
    if isinstance(value, (list, tuple)):
        value = "; ".join(str(v) for v in value if v not in (None, ""))
    elif isinstance(value, dict):
        import json

        value = json.dumps(value, ensure_ascii=False)
    text = " ".join(str(value or "").split())
    if limit is None:
        limit = conf("CHAT_JOB_FIELD_CHARS")
    return text if len(text) <= limit else text[: limit - 1] + "…"


# ----------------------------------------------------------------------------
# Постановка в очередь
# ----------------------------------------------------------------------------
def _find_job(session_id: int, image_path: str) -> ChatAnalysisJob | None:
    return db.session.scalars(
        select(ChatAnalysisJob)
        .where(ChatAnalysisJob.session_id == session_id, ChatAnalysisJob.image_path == image_path)
        .order_by(ChatAnalysisJob.id.desc())
    ).first()


def enqueue(user, session_id: int, image, caption: str = "") -> dict:
    """Ставит вложение чата (chat_tools.images.ChatImage) в общую очередь анализа.

    Повторный вызов для того же файла в том же чате новую задачу не создаёт — возвращает
    статус уже существующей. Возвращает словарь для модели (см. status_payload).
    Бросает JobError, если поставить не удалось.
    """
    existing = _find_job(session_id, image.path)
    if existing is not None:
        row = db.session.get(AnalysisResult, existing.analysis_id) if existing.analysis_id else None
        if row is not None:
            return status_payload(row, existing, repeated=True)
        # задачу успели отменить из очереди — ниже поставим заново

    from .queue_worker import queue_limit_hit, wake_worker

    hit = queue_limit_hit(user.id)
    if hit:
        scope, now_count, limit = hit
        who = "очередь пользователя заполнена" if scope == "user" else "общая очередь заполнена"
        raise JobError(f"{who} ({now_count} из {limit}) — попробуйте позже")

    data = chat_images.read_bytes(image.path)
    if data is None:
        raise JobError(f"файл изображения #{image.number} недоступен (удалён с диска)")

    # У анализа своя копия файла: история, миниатюры и удаление записей работают с uploads/
    # как обычно, а удаление чата не ломает уже сохранённый анализ (и наоборот).
    now = datetime.now(timezone.utc)
    root = Path(current_app.config["UPLOAD_FOLDER"])
    rel = f"{Config.UPLOADS_DIR}/{now:%Y/%m/%d}/{uuid.uuid4().hex}{Path(image.path).suffix.lower()}"
    target = root / rel
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        row = AnalysisResult(
            user_id=user.id,
            image_path=rel,
            original_name=image.name[:255],
            image_mime=image.mime,
            image_hash=image_dedup.sha256_bytes(data),
            caption=(caption or "")[: conf("CAPTION_MAX_CHARS")],
            status=Status.QUEUED,
        )
        db.session.add(row)
        db.session.flush()
        job = ChatAnalysisJob(
            session_id=session_id,
            analysis_id=row.id,
            image_path=image.path,
            image_name=image.name[:255],
            image_number=image.number,
        )
        db.session.add(job)
        db.session.commit()
    except Exception:
        db.session.rollback()
        target.unlink(missing_ok=True)
        raise

    wake_worker(current_app)
    return status_payload(row, job)


def cancel_for_paths(session_id: int, chat_paths: list[str]) -> None:
    """Откат: сообщение с этими вложениями не сохранилось (ход чата не удался) —
    убираем связанные задачи и, если анализ ещё не начался, сам анализ из очереди."""
    if not chat_paths:
        return
    jobs = db.session.scalars(
        select(ChatAnalysisJob).where(
            ChatAnalysisJob.session_id == session_id, ChatAnalysisJob.image_path.in_(chat_paths)
        )
    ).all()
    to_remove: list[str] = []
    for job in jobs:
        if job.analysis_id:
            row = db.session.get(AnalysisResult, job.analysis_id)
            path = row.image_path if row is not None else ""
            cancelled = db.session.execute(
                delete(AnalysisResult)
                .where(AnalysisResult.id == job.analysis_id, AnalysisResult.status == Status.QUEUED)
                .execution_options(synchronize_session=False)
            )
            if cancelled.rowcount and path:
                to_remove.append(path)
        db.session.delete(job)
    db.session.commit()
    remove_image_files(to_remove)


# ----------------------------------------------------------------------------
# Статус и результат
# ----------------------------------------------------------------------------
def _queue_position(row: AnalysisResult) -> int:
    """Сколько анализов (всех пользователей) стоит в очереди перед этим."""
    return db.session.scalar(
        select(func.count(AnalysisResult.id)).where(
            AnalysisResult.status != Status.DONE, AnalysisResult.id < row.id
        )
    ) or 0


def report_payload(row: AnalysisResult) -> dict:
    """Готовый отчёт анализа в компактном виде для модели."""
    report = row.raw_report if isinstance(row.raw_report, dict) else {}
    if "_raw" in report:
        return {
            "warning": "модель вернула не структурированный отчёт, а свободный текст — результат приблизительный",
            "raw_text": _clip(report.get("_raw"), conf("CHAT_JOB_ANSWER_CHARS")),
        }
    signals = []
    raw_signals = report.get("signals")
    for item in (raw_signals if isinstance(raw_signals, list) else [])[: conf("CHAT_JOB_MAX_SIGNALS")]:
        if isinstance(item, dict):
            signals.append({
                "category": _clip(item.get("category"), conf("CHAT_JOB_CATEGORY_CHARS")),
                "detail": _clip(item.get("detail"), conf("CHAT_JOB_DETAIL_CHARS")),
            })
        elif item not in (None, ""):
            signals.append({"category": "", "detail": _clip(item, conf("CHAT_JOB_DETAIL_CHARS"))})
    return {
        "risk_level": row.risk_level,
        "risk_label": RISK_LABELS.get(row.risk_level, row.risk_level),
        "needs_human_review": bool(row.needs_human_review),
        "description": _clip(row.description),
        "signals": signals,
        "rationale": _clip(report.get("rationale")),
        "recommendation": _clip(report.get("recommendation")),
        "text_on_image": _clip(report.get("text_on_image")),
        "context": _clip(report.get("context")),
    }


def status_payload(row: AnalysisResult, job: ChatAnalysisJob, repeated: bool = False) -> dict:
    """Что инструмент отвечает модели: статус задачи, а для завершённой — ещё и отчёт."""
    payload: dict = {
        "image": job.image_number,
        "name": job.image_name,
        "analysis_id": row.id,
    }
    if row.status == Status.DONE:
        if row.is_error:
            payload.update({"status": "error", "error": _clip(row.error)})
        else:
            payload.update({"status": "done", **report_payload(row)})
        return payload

    payload["status"] = "processing" if row.status == Status.PROCESSING else "queued"
    payload["ahead_in_queue"] = _queue_position(row)
    note = (
        "изображение поставлено в очередь анализа; результат придёт в этот чат автоматически, когда "
        "анализ завершится — сообщи об этом пользователю и не гадай, что на картинке"
    )
    if repeated:
        note = "это изображение уже стоит в очереди — повторно ставить не нужно; " + note
    payload["note"] = note
    return payload


def card(row: AnalysisResult) -> dict:
    """Карточка-ссылка на анализ под ответом (тот же формат, что у search_analyses)."""
    return {
        "id": row.id,
        "url": url_for("analyzer.result_detail", pk=row.id),
        "thumb_url": url_for("analyzer.thumb", filename=row.image_path) if row.image_path else "",
        "label": row.original_name or f"Анализ #{row.id}",
        "risk_level": row.risk_level,
        "risk_label": row.risk_level_display,
        "date": local_dt(row.created_at),
    }


def fallback_text(job: ChatAnalysisJob, row: AnalysisResult | None) -> str:
    """Сообщение без участия модели — если она недоступна, а результат уже готов."""
    name = job.image_name or f"#{job.image_number}"
    if row is None:
        return f"Анализ изображения «{name}» отменён — записи в очереди больше нет."
    if row.is_error:
        return f"Анализ изображения «{name}» завершился ошибкой: {_clip(row.error, conf("CHAT_JOB_DETAIL_CHARS"))}"
    text = f"Анализ изображения «{name}» завершён. Уровень риска: {row.risk_level_display.lower()}."
    if row.needs_human_review:
        text += " Требуется проверка человеком."
    if row.description:
        text += f"\n\n{_clip(row.description)}"
    return text + "\n\nПодробный отчёт — по ссылке ниже."


# ----------------------------------------------------------------------------
# Доставка результата в чат
# ----------------------------------------------------------------------------
def claim_ready(session_id: int) -> list[tuple[ChatAnalysisJob, AnalysisResult | None]]:
    """Забирает задачи чата, анализ которых завершён (или отменён), и помечает их доставленными.

    Пометка — атомарный UPDATE ... WHERE delivered = 0: если результат одновременно
    запросили две вкладки, каждую задачу получит ровно одна. Если после этого доставить
    не получится, повторно она не придёт — вызывающая сторона обязана отдать хотя бы fallback_text.
    """
    jobs = db.session.scalars(
        select(ChatAnalysisJob)
        .where(ChatAnalysisJob.session_id == session_id, ChatAnalysisJob.delivered.is_(False))
        .order_by(ChatAnalysisJob.id)
    ).all()
    ready: list[tuple[ChatAnalysisJob, AnalysisResult | None]] = []
    for job in jobs:
        row = db.session.get(AnalysisResult, job.analysis_id) if job.analysis_id else None
        if row is not None and row.status != Status.DONE:
            continue
        claimed = db.session.execute(
            update(ChatAnalysisJob)
            .where(ChatAnalysisJob.id == job.id, ChatAnalysisJob.delivered.is_(False))
            .values(delivered=True)
            .execution_options(synchronize_session=False)
        )
        db.session.commit()
        if claimed.rowcount == 1:
            ready.append((job, db.session.get(AnalysisResult, job.analysis_id) if job.analysis_id else None))
    return ready


def pending_count(session_id: int) -> int:
    """Сколько задач чата ещё ждут доставки результата."""
    return db.session.scalar(
        select(func.count(ChatAnalysisJob.id)).where(
            ChatAnalysisJob.session_id == session_id, ChatAnalysisJob.delivered.is_(False)
        )
    ) or 0
