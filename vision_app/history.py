"""Удаление записей истории вместе с файлами изображений."""

from __future__ import annotations

import logging
from pathlib import Path

from flask import current_app
from sqlalchemy import delete, func, select

from .extensions import db
from .models import AnalysisEmbedding, AnalysisResult, Status, User
from .thumbs import THUMBS_DIR, thumb_rel

log = logging.getLogger("vision_app.history")

_CHUNK = 500  # не упираемся в лимит числа параметров SQL


def _prune_empty_dirs(parent: Path, stop_root: Path) -> None:
    """Убирает пустые папки ГГГГ/ММ/ДД вверх до stop_root (сам stop_root не трогаем)."""
    while stop_root in parent.parents:
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


def remove_image_files(paths: list[str]) -> None:
    """Удаляет файлы загрузок, их миниатюры и пустые папки с датой. Никогда не выходит за UPLOAD_FOLDER."""
    root = Path(current_app.config["UPLOAD_FOLDER"]).resolve()
    uploads_root = root / "uploads"
    thumbs_root = root / THUMBS_DIR

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
        _prune_empty_dirs(target.parent, uploads_root)  # uploads/ГГГГ/ММ/ДД

        thumb = (root / thumb_rel(rel)).resolve()
        if thumbs_root in thumb.parents:
            try:
                thumb.unlink(missing_ok=True)
            except OSError as exc:
                log.warning("Не удалось удалить миниатюру %s: %s", thumb, exc)
            else:
                _prune_empty_dirs(thumb.parent, thumbs_root)


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
        # SQLite не выполняет ON DELETE CASCADE без PRAGMA foreign_keys — чистим векторы сами.
        db.session.execute(
            delete(AnalysisEmbedding)
            .where(AnalysisEmbedding.analysis_id.in_(ids[i : i + _CHUNK]))
            .execution_options(synchronize_session=False)
        )
        db.session.execute(
            delete(AnalysisResult)
            .where(AnalysisResult.id.in_(ids[i : i + _CHUNK]), AnalysisResult.status == Status.DONE)
            .execution_options(synchronize_session=False)
        )
    db.session.commit()

    # файлы удаляем только после успешного коммита
    remove_image_files([r.image_path for r in rows])
    return len(ids)


def delete_user_account(user: User) -> None:
    """Удаляет пользователя целиком: все его анализы (любого статуса) вместе с
    файлами изображений, а затем саму учётную запись.

    Записи AnalysisResult (и чаты) удалились бы каскадом на уровне БД (ondelete="CASCADE"),
    но файлы изображений — загрузки и вложения чатов — так не подчистить, поэтому собираем пути заранее.
    """
    from . import chat_images  # локально: chat_images сам импортирует этот модуль

    paths = db.session.scalars(
        select(AnalysisResult.image_path).where(AnalysisResult.user_id == user.id)
    ).all()
    chat_paths = chat_images.user_paths(user.id)  # вложения чатов — собираем до удаления записей

    db.session.execute(
        delete(AnalysisEmbedding)
        .where(AnalysisEmbedding.analysis_id.in_(select(AnalysisResult.id).where(AnalysisResult.user_id == user.id)))
        .execution_options(synchronize_session=False)
    )
    db.session.delete(user)
    db.session.commit()

    remove_image_files([p for p in paths if p])
    chat_images.remove_files(chat_paths)
