"""Категории оценивания: сидирование стартового набора и сборка payload для
сервера анализа (см. models.Category, services.analyze_image, категории —
единственное место хранения — таблица categories в БД vision_app)."""

from __future__ import annotations

import logging

from sqlalchemy import func, select

from .extensions import db
from .models import Category, _split_wrapper
from .services import VisionApiError, fetch_server_categories

log = logging.getLogger("vision_app.categories")


def _title_from_name(name: str) -> str:
    """weapons_and_dangerous_objects -> "Weapons and dangerous objects" (сервер название не отдаёт)."""
    return name.replace("_", " ").strip().capitalize() or name


def category_from_server(item: dict, position: int) -> Category:
    """Ответ GET /categories/<имя> -> строка таблицы categories.

    Сервер отдаёт full/compact уже завёрнутыми в <signal_category name="...">; в БД хранится
    только тело + «хвост» после закрывающего тега (см. models._split_wrapper)."""
    full, full_extra = _split_wrapper(item.get("full") or "")
    compact, compact_extra = _split_wrapper(item.get("compact") or "")
    examples = item.get("examples") if isinstance(item.get("examples"), dict) else {}
    return Category(
        name=item["name"],
        title=_title_from_name(item["name"]),
        summary=item.get("summary") or "",
        full=full,
        compact=compact,
        full_extra=full_extra,
        compact_extra=compact_extra,
        example_en=examples.get("en") or "",
        example_ru=examples.get("ru") or "",
        position=position,
        is_active=True,
    )


def seed_default_categories() -> bool:
    """Заполняет таблицу categories набором, который сервер анализа отдаёт по
    GET /categories и GET /categories/<имя> — один раз, только если таблица
    ещё пуста (например, самый первый запуск на чистой БД). Дальше единственный
    источник правды — то, что админы изменят в /panel/categories/, этот сидер
    больше не трогается.

    Сервер может быть недоступен при старте — это не ошибка: анализ без
    категорий из БД работает на категориях самого сервера, а загрузить набор
    можно позже (повторная попытка при открытии /panel/categories/, команда
    `flask seed-categories`). Возвращает True, если категории были добавлены."""
    if db.session.scalar(select(func.count(Category.id))):
        return False

    try:
        items = fetch_server_categories()
    except VisionApiError as exc:
        log.warning("categories: не удалось получить стартовый набор с сервера анализа: %s", exc)
        return False
    if not items:
        log.warning("categories: сервер анализа не вернул ни одной категории")
        return False

    for position, item in enumerate(items):
        db.session.add(category_from_server(item, position))
    db.session.commit()
    log.info("categories: загружен стартовый набор с сервера анализа: %d категорий", len(items))
    return True


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
