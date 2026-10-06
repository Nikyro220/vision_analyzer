"""Мини-обновление схемы: добавляет в существующие таблицы недостающие колонки.

db.create_all() создаёт только отсутствующие ТАБЛИЦЫ и не трогает уже существующие.
Когда в модель добавляется новая колонка, в старой базе её нет — и приложение падает
с «no such column». Чтобы не заставлять вас настраивать миграции, при старте мы
добавляем недостающие колонки командой ALTER TABLE ... ADD COLUMN.

Ограничения: добавляет только колонки (не меняет типы, не удаляет, не создаёт индексы).
Если нужны полноценные миграции — используйте Flask-Migrate (AUTO_CREATE_DB=0).
"""

from __future__ import annotations

import logging

from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError

from .extensions import db

log = logging.getLogger("vision_app.schema")


def ensure_schema(engine) -> list[str]:
    """Возвращает список добавленных колонок вида 'таблица.колонка'."""
    added: list[str] = []
    inspector = inspect(engine)
    quote = engine.dialect.identifier_preparer.quote

    for table in db.metadata.sorted_tables:
        if not inspector.has_table(table.name):
            continue
        existing = {c["name"] for c in inspector.get_columns(table.name)}

        for column in table.columns:
            if column.name in existing:
                continue

            default_sql = ""
            if column.server_default is not None:
                arg = getattr(column.server_default, "arg", None)
                if isinstance(arg, str):
                    default_sql = " DEFAULT '" + arg.replace("'", "''") + "'"

            if not column.nullable and not default_sql:
                log.warning(
                    "Не могу автоматически добавить NOT NULL колонку %s.%s без значения по умолчанию "
                    "— используйте миграции.", table.name, column.name,
                )
                continue

            ddl = (
                f"ALTER TABLE {quote(table.name)} ADD COLUMN {quote(column.name)} "
                f"{column.type.compile(dialect=engine.dialect)}{default_sql}"
                f"{'' if column.nullable else ' NOT NULL'}"
            )
            with engine.begin() as conn:
                conn.execute(text(ddl))
            log.info("Схема БД обновлена: добавлена колонка %s.%s", table.name, column.name)
            added.append(f"{table.name}.{column.name}")

    return added


# Порядок важен: из этих колонок собирается «Фамилия Имя».
_LEGACY_NAME_COLUMNS = ("last_name", "first_name")


def migrate_legacy_names(engine) -> int:
    """Разовый перенос ФИО (users.last_name/first_name) в users.nickname с удалением старых колонок.

    Старые колонки NOT NULL и без значения по умолчанию в самой БД, поэтому оставлять их нельзя:
    после удаления из модели любая вставка нового пользователя упала бы. Сначала копируем
    «Фамилия Имя» в nickname (только тем, у кого он ещё пуст), затем удаляем колонки. Повторный
    запуск ничего не делает. Возвращает число скопированных имён.

    DROP COLUMN нужен SQLite >= 3.35 (2021 г.). Если колонку удалить не удалось, пишем ошибку в лог:
    имена к этому моменту уже скопированы, ничего не потеряно.
    """
    inspector = inspect(engine)
    if not inspector.has_table("users"):
        return 0
    existing = {c["name"] for c in inspector.get_columns("users")}
    legacy = [c for c in _LEGACY_NAME_COLUMNS if c in existing]
    if not legacy or "nickname" not in existing:
        return 0  # нечего переносить (или nickname ещё не создан — тогда ничего не удаляем)

    quote = engine.dialect.identifier_preparer.quote
    copied = 0
    with engine.begin() as conn:
        columns = ", ".join(quote(c) for c in ("id", "nickname", *legacy))
        for row in conn.execute(text(f"SELECT {columns} FROM users")).mappings().all():
            if (row["nickname"] or "").strip():
                continue
            parts = ((row[c] or "").strip() for c in legacy)
            name = " ".join(p for p in parts if p)[:150]
            if name:
                conn.execute(text("UPDATE users SET nickname = :n WHERE id = :i"), {"n": name, "i": row["id"]})
                copied += 1

    for column in legacy:
        try:
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE users DROP COLUMN {quote(column)}"))
        except SQLAlchemyError as exc:
            log.error(
                "Не удалось удалить устаревшую колонку users.%s (%s). Имена уже скопированы в nickname; "
                "удалите колонку вручную: ALTER TABLE users DROP COLUMN %s;", column, exc, column,
            )
            break
    log.info("ФИО перенесено в nickname: %d, устаревшие колонки удалены", copied)
    return copied
