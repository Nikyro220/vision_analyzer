"""Категории оценивания: сидирование стартового набора и сборка payload для
сервера анализа (см. models.Category, services.analyze_image, категории —
единственное место хранения — таблица categories в БД vision_app)."""

from __future__ import annotations

import logging

from sqlalchemy import func, select

from .category_seed import DEFAULT_CATEGORIES
from .extensions import db
from .models import Category, _split_wrapper

log = logging.getLogger("vision_app.categories")


def seed_default_categories() -> None:
    """Заполняет таблицу categories стартовым набором — один раз, только
    если она ещё пуста (например, самый первый запуск на чистой БД).
    Дальше единственный источник правды — то, что админы изменят в
    /panel/categories/, этот сидер больше не трогается."""
    if db.session.scalar(select(func.count(Category.id))):
        return

    for position, item in enumerate(DEFAULT_CATEGORIES):
        db.session.add(
            Category(
                name=item["name"],
                title=item.get("title") or item["name"],
                summary=item.get("summary", ""),
                full=item.get("full", ""),
                compact=item.get("compact", ""),
                full_extra=item.get("full_extra", ""),
                compact_extra=item.get("compact_extra", ""),
                example_en=item.get("example_en", ""),
                example_ru=item.get("example_ru", ""),
                position=position,
                is_active=True,
            )
        )
    db.session.commit()
    log.info("categories: загружен стартовый набор из %d категорий", len(DEFAULT_CATEGORIES))


def normalize_legacy_wrappers() -> None:
    """Разовая миграция для категорий, созданных до появления «конструктора
    правил» в форме: если в full/compact всё ещё лежит целиком готовый блок
    <signal_category name="...">...</signal_category> (так писала самая первая
    версия формы — целиком руками), разносит его на тело + full_extra/
    compact_extra. Идемпотентно — на уже нормализованных строках ничего не
    делает, так что безопасно вызывать при каждом старте."""
    changed = False
    for cat in db.session.scalars(select(Category)).all():
        full_body, full_extra = _split_wrapper(cat.full)
        if full_body != cat.full:
            cat.full = full_body
            cat.full_extra = f"{full_extra}\n\n{cat.full_extra}".strip() if cat.full_extra else full_extra
            changed = True

        compact_body, compact_extra = _split_wrapper(cat.compact)
        if compact_body != cat.compact:
            cat.compact = compact_body
            cat.compact_extra = (
                f"{compact_extra}\n\n{cat.compact_extra}".strip() if cat.compact_extra else compact_extra
            )
            changed = True

    if changed:
        db.session.commit()
        log.info("categories: старые записи с обёрткой <signal_category> приведены к новому формату")


def active_categories() -> list[Category]:
    """Включённые категории в порядке отображения/промпта."""
    return db.session.scalars(
        select(Category).where(Category.is_active.is_(True)).order_by(Category.position, Category.id)
    ).all()


def build_categories_payload() -> list[dict] | None:
    """Список включённых категорий в формате, который ждёт /analyze
    (поле "categories" — см. Category.to_overlay_dict). None, если ни одной
    включённой категории нет — тогда вызывающая сторона просто не передаёт
    параметр, и сервер анализа работает без разовых категорий."""
    rows = active_categories()
    return [row.to_overlay_dict() for row in rows] or None
