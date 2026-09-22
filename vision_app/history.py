"""Удаление записей истории вместе с файлами изображений."""

from __future__ import annotations

import logging
from pathlib import Path

from flask import current_app
from sqlalchemy import delete, func, select

from .extensions import db
from .models import AnalysisResult, Status

log = logging.getLogger("vision_app.history")

_CHUNK = 500  # не упираемся в лимит числа параметров SQL


def remove_image_files(paths: list[str]) -> None:
    """Удаляет файлы загрузок и пустые папки с датой. Никогда не выходит за UPLOAD_FOLDER."""
    root = Path(current_app.config["UPLOAD_FOLDER"]).resolve()
    uploads_root = root / "uploads"

    for rel in paths:
        if not rel:
            continue
        # файл может быть общим для нескольких записей — тогда его не трогаем
        if db.session.scalar(select(func.count(AnalysisResult.id)).where(AnalysisResult.image_path == rel)):
            continue
        target = (root / rel).resolve()
        if root not in target.parents:  # защита от «../»
            log.warning("Пропускаю путь вне UPLOAD_FOLDER: %s", rel)
            continue
        try:
            target.unlink(missing_ok=True)
        except OSError as exc:
            log.warning("Не удалось удалить файл %s: %s", target, exc)
            continue
        # подчищаем пустые папки uploads/ГГГГ/ММ/ДД
        parent = target.parent
        while uploads_root in parent.parents:
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent


def delete_finished(*conditions) -> int:
    """Удаляет ЗАВЕРШЁННЫЕ записи (status=done), подходящие под условия, и их файлы.

    Записи в очереди/обработке не удаляются: очередь отменяется отдельно,
    а запись, которую прямо сейчас обрабатывает обработчик, трогать нельзя.
    Возвращает количество удалённых записей.
    """
    rows = db.session.execute(
        select(AnalysisResult.id, AnalysisResult.image_path).where(
            AnalysisResult.status == Status.DONE, *conditions
        )
    ).all()
    if not rows:
        return 0

    ids = [r.id for r in rows]
    for i in range(0, len(ids), _CHUNK):
        db.session.execute(
            delete(AnalysisResult)
            .where(AnalysisResult.id.in_(ids[i : i + _CHUNK]), AnalysisResult.status == Status.DONE)
            .execution_options(synchronize_session=False)
        )
    db.session.commit()

    # файлы удаляем только после успешного коммита
    remove_image_files([r.image_path for r in rows])
    return len(ids)
