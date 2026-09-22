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
