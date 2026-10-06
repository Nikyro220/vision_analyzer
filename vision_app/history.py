"""Удаление записей истории вместе с файлами изображений."""

from __future__ import annotations

import logging
from pathlib import Path

from flask import current_app
from sqlalchemy import delete, func, or_, select

from .config import Config, conf
from .extensions import db
from .models import (
    AnalysisEmbedding,
    AnalysisResult,
    ChatAnalysisJob,
    ChatMessage,
    ChatSession,
    Status,
    User,
)
from .thumbs import THUMBS_DIR, thumb_rel

log = logging.getLogger("vision_app.history")


def _chunk() -> int:
    """Размер пачки id в IN (...) — не упираемся в лимит числа параметров SQL (SQL_IN_CHUNK)."""
    return max(int(conf("SQL_IN_CHUNK")), 1)


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
    uploads_root = root / Config.UPLOADS_DIR
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
    chunk = _chunk()
    for i in range(0, len(ids), chunk):
        # SQLite не выполняет ON DELETE CASCADE без PRAGMA foreign_keys — чистим векторы сами.
        db.session.execute(
            delete(AnalysisEmbedding)
            .where(AnalysisEmbedding.analysis_id.in_(ids[i : i + chunk]))
            .execution_options(synchronize_session=False)
        )
        db.session.execute(
            delete(AnalysisResult)
            .where(AnalysisResult.id.in_(ids[i : i + chunk]), AnalysisResult.status == Status.DONE)
            .execution_options(synchronize_session=False)
        )
    db.session.commit()

    # файлы удаляем только после успешного коммита
    remove_image_files([r.image_path for r in rows])
    return len(ids)


def _purge_owned(analysis_where, session_where) -> tuple[list[str], list[str], int, int]:
    """Удаляет анализы и чаты, подходящие под условия, вместе со ВСЕМИ зависимыми строками
    (векторы, задачи анализа из чата, сообщения). Без коммита.

    Возвращает (пути загрузок анализов, пути вложений чатов, число анализов, число чатов) —
    файлы удаляет вызывающий, уже после коммита.

    Всё делается явно, а не через ON DELETE CASCADE: связи User -> AnalysisResult / ChatSession
    объявлены с ondelete="CASCADE" и passive_deletes=True, то есть ORM рассчитывает, что каскад
    выполнит БД. SQLite же не проверяет внешние ключи, пока не включён PRAGMA foreign_keys (в
    приложении он выключен), — и строки анализов и чатов оставались «сиротами» без владельца
    (в панели админа у них пропадал ник, файлы на диске не чистились). Порядок удаления — от
    дочерних таблиц к родительским, чтобы то же работало и с включёнными внешними ключами.
    """
    from . import chat_images  # локально: chat_images сам импортирует этот модуль

    analysis_ids = select(AnalysisResult.id).where(analysis_where)
    session_ids = select(ChatSession.id).where(session_where)
    opts = {"synchronize_session": False}

    paths = [p for p in db.session.scalars(select(AnalysisResult.image_path).where(analysis_where)) if p]
    chat_paths = chat_images.paths_of(
        db.session.scalars(select(ChatMessage).where(ChatMessage.session_id.in_(session_ids))).all()
    )  # вложения чатов — собираем до удаления записей

    db.session.execute(
        delete(AnalysisEmbedding).where(AnalysisEmbedding.analysis_id.in_(analysis_ids)).execution_options(**opts)
    )
    db.session.execute(
        delete(ChatAnalysisJob)
        .where(or_(ChatAnalysisJob.session_id.in_(session_ids), ChatAnalysisJob.analysis_id.in_(analysis_ids)))
        .execution_options(**opts)
    )
    db.session.execute(delete(ChatMessage).where(ChatMessage.session_id.in_(session_ids)).execution_options(**opts))
    chats = db.session.execute(delete(ChatSession).where(session_where).execution_options(**opts))
    analyses = db.session.execute(delete(AnalysisResult).where(analysis_where).execution_options(**opts))
    return paths, chat_paths, analyses.rowcount or 0, chats.rowcount or 0


def delete_user_account(user: User) -> None:
    """Удаляет пользователя целиком: все его анализы (любого статуса), чаты и сообщения вместе с
    файлами изображений, а затем саму учётную запись.

    Записей «от имени» удалённого пользователя не остаётся: в истории админа не бывает анализов
    без указания, чей он.
    """
    from . import avatars, chat_images

    user_id = user.id
    paths, chat_paths, _, _ = _purge_owned(AnalysisResult.user_id == user_id, ChatSession.user_id == user_id)
    db.session.execute(delete(User).where(User.id == user_id).execution_options(synchronize_session=False))
    db.session.commit()
    db.session.expire_all()

    remove_image_files(paths)
    chat_images.remove_files(chat_paths)
    avatars.remove_avatar(user_id)


def purge_orphans() -> dict[str, int]:
    """Удаляет данные, чей владелец уже удалён (остались от версий, где каскад на SQLite не
    срабатывал): анализы, чаты, сообщения и файлы. Возвращает {"analyses": N, "chats": M}.
    Нужен app context."""
    from . import chat_images

    known_users = select(User.id)
    paths, chat_paths, analyses, chats = _purge_owned(
        ~AnalysisResult.user_id.in_(known_users), ~ChatSession.user_id.in_(known_users)
    )
    db.session.commit()
    remove_image_files(paths)
    chat_images.remove_files(chat_paths)
    return {"analyses": analyses, "chats": chats}
