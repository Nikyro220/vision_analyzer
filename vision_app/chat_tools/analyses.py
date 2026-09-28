"""Инструмент `search_analyses`: чтение истории анализов для чата.

Это слой данных для системы «тулзов» (см. runner.py): модель сама решает,
нужна ли ей история анализов, и присылает JSON-вызов с аргументами; здесь эти
аргументы валидируются, применяются права доступа, выполняется SQL-запрос и
собирается результат (JSON для модели + карточки-ссылки для интерфейса).

Принципы:

  - Аргументы от модели — НЕДОВЕРЕННЫЙ ввод. Всё приводится к безопасным типам
    и диапазонам, неизвестное игнорируется (с предупреждением в результате).
  - Права применяются здесь, на сервере, и модель на них повлиять не может:
    user_id в аргументах не принимается вообще; `own_only` умеет только СУЖАТЬ
    выборку. Полный доступ ко всей истории — только у head_admin, остальные
    видят исключительно свои анализы.
  - Инструмент только читает. Худший исход злоупотребления (например, текст на
    картинке, который пытается «управлять» моделью) — пользователь увидит свои
    же данные.
  - Карточки строятся из тех же строк БД, что вернулись модели, а не из её
    текста, поэтому ссылка не может ускользнуть на несуществующий анализ.

Почему не векторный поиск: данные структурированы (risk_level, категории,
даты, user_id), вопросы сводятся к фильтрам — SQL точнее и проще. Векторный
поиск по description имеет смысл добавить позже отдельным аргументом
инструмента, если понадобится «нечёткий» поиск (см. TODO внизу).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from flask import current_app, url_for
from sqlalchemy import select

from ..extensions import db
from ..models import RISK_LABELS, AnalysisResult, Category, RiskLevel, Status, User
from ..utils import local_dt

TOOL_NAME = "search_analyses"

_SCAN_LIMIT = 500  # сколько последних строк максимум разбираем (защита от полного скана)
_DEFAULT_LIMIT = 5
_MAX_LIMIT = 15
_MAX_CARDS = 8  # больше карточек под одним ответом — визуальный шум
_DESC_LEN = 160
_MAX_SINCE_DAYS = 365
_VALID_RISKS = {RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.UNKNOWN}


@dataclass
class ToolResult:
    text: str  # JSON-строка, которая уходит модели
    references: list = field(default_factory=list)  # карточки для фронта


# ---------------------------------------------------------------------------
# Категории (нужны и для описания инструмента в промпте, и для валидации)
# ---------------------------------------------------------------------------


def active_categories() -> list[tuple[str, str]]:
    """[(name, title)] активных категорий — что модель может подставить в `categories`."""
    rows = db.session.scalars(select(Category).where(Category.is_active.is_(True))).all()
    return [(c.name, c.title) for c in rows]


# ---------------------------------------------------------------------------
# Валидация аргументов
# ---------------------------------------------------------------------------


def _int_arg(value, lo: int, hi: int) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number < lo:
        return None  # заведомо некорректное значение (0, отрицательное) — игнорируем, а не подгоняем
    return min(hi, number)


def _bool_arg(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "да"}
    return bool(value)


def normalize_args(raw) -> tuple[dict, list[str]]:
    """Приводит сырые аргументы от модели к безопасному виду. Возвращает
    (очищенные аргументы, список предупреждений для модели)."""
    warnings: list[str] = []
    if not isinstance(raw, dict):
        warnings.append("args должен быть объектом — использованы значения по умолчанию")
        raw = {}

    known = {"limit", "since_days", "risk_level", "categories", "needs_review", "own_only", "count_only"}
    unknown = sorted(str(k) for k in raw if k not in known)
    if unknown:
        warnings.append("неизвестные аргументы проигнорированы: " + ", ".join(unknown))

    args: dict = {
        "limit": _int_arg(raw.get("limit"), 1, _MAX_LIMIT) or _DEFAULT_LIMIT,
        "since_days": _int_arg(raw.get("since_days"), 1, _MAX_SINCE_DAYS),
        "risk_level": None,
        "categories": [],
        "needs_review": _bool_arg(raw.get("needs_review")),
        "own_only": _bool_arg(raw.get("own_only")),
        "count_only": _bool_arg(raw.get("count_only")),
    }

    risk = raw.get("risk_level")
    if risk not in (None, ""):
        risk = str(risk).strip().lower()
        if risk in _VALID_RISKS:
            args["risk_level"] = risk
        else:
            warnings.append(f"risk_level '{risk}' не распознан и проигнорирован")

    cats = raw.get("categories")
    if isinstance(cats, str):
        cats = [cats]
    if isinstance(cats, list) and cats:
        lookup: dict[str, str] = {}
        for name, title in active_categories():
            lookup[name.strip().lower()] = name
            lookup[title.strip().lower()] = name
        for item in cats[:10]:
            found = lookup.get(str(item).strip().lower())
            if found:
                if found not in args["categories"]:
                    args["categories"].append(found)
            else:
                warnings.append(f"категория '{item}' не найдена и проигнорирована")

    return args, warnings


# ---------------------------------------------------------------------------
# Выборка + права доступа
# ---------------------------------------------------------------------------


def _in_scope_condition(user: User, own_only: bool):
    """Полный доступ ко всей истории — только у главного администратора.
    Обычный admin по видимости в чате приравнен к рядовому пользователю.
    own_only может только сузить выборку до своих анализов."""
    if user.is_head_admin and not own_only:
        return None
    return AnalysisResult.user_id == user.id


def _row_categories(row: AnalysisResult) -> set[str]:
    signals = (row.raw_report or {}).get("signals") or []
    return {s.get("category") for s in signals if isinstance(s, dict) and s.get("category")}


def _fetch_rows(user: User, args: dict) -> list[AnalysisResult]:
    conditions = [AnalysisResult.status == Status.DONE]
    scope = _in_scope_condition(user, args["own_only"])
    if scope is not None:
        conditions.append(scope)
    if args["risk_level"]:
        conditions.append(AnalysisResult.risk_level == args["risk_level"])
    if args["needs_review"]:
        conditions.append(AnalysisResult.needs_human_review.is_(True))
    if args["since_days"]:
        conditions.append(AnalysisResult.created_at >= datetime.now(timezone.utc) - timedelta(days=args["since_days"]))

    stmt = (
        select(AnalysisResult)
        .where(*conditions)
        .order_by(AnalysisResult.created_at.desc())
        .limit(_SCAN_LIMIT)
    )
    rows = list(db.session.scalars(stmt).all())

    if args["categories"]:
        wanted = set(args["categories"])
        rows = [r for r in rows if _row_categories(r) & wanted]
    return rows


# ---------------------------------------------------------------------------
# Результат: JSON для модели + карточки для интерфейса
# ---------------------------------------------------------------------------


def _clean(text: str) -> str:
    """Описания генерирует модель по картинке (а значит, их содержимое может
    быть подсунуто текстом на изображении) — не даём им имитировать служебные
    маркеры протокола."""
    return (text or "").replace("[TOOL", "(TOOL").replace("[/TOOL", "(/TOOL")


def _record(row: AnalysisResult, show_user: bool) -> dict:
    desc = _clean((row.description or "").strip().replace("\n", " "))
    if len(desc) > _DESC_LEN:
        desc = desc[: _DESC_LEN - 1] + "…"
    rec = {
        "id": row.id,
        "date": local_dt(row.created_at, "%d.%m.%Y %H:%M"),
        "risk": row.risk_level_display,
        "needs_review": bool(row.needs_human_review),
        "categories": sorted(_row_categories(row)),
        "description": desc,
    }
    if show_user and row.user:
        rec["user"] = row.user.username
    return rec


def _build_reference_cards(rows: list[AnalysisResult], show_user: bool) -> list[dict]:
    cards = []
    for row in rows[:_MAX_CARDS]:
        card = {
            "id": row.id,
            "url": url_for("analyzer.result_detail", pk=row.id),
            "thumb_url": url_for("analyzer.thumb", filename=row.image_path) if row.image_path else "",
            "label": row.original_name or f"Анализ #{row.id}",
            "risk_level": row.risk_level,
            "risk_label": row.risk_level_display,
            "date": local_dt(row.created_at, "%d.%m.%Y %H:%M"),
        }
        if show_user and row.user:
            card["username"] = row.user.username
        cards.append(card)
    return cards


def search_analyses(user: User, raw_args) -> ToolResult:
    """Точка входа инструмента: аргументы от модели -> результат."""
    args, warnings = normalize_args(raw_args)

    # Категории запрошены, но ни одна не распознана: молча отдать всё без фильтра
    # было бы обманом («были ли анализы с X?» -> «вот всё»). Возвращаем ошибку со
    # списком доступных категорий — модель может повторить вызов.
    if isinstance(raw_args, dict) and raw_args.get("categories") and not args["categories"]:
        return ToolResult(
            json.dumps(
                {
                    "error": "указанные категории не найдены",
                    "available_categories": [name for name, _ in active_categories()],
                },
                ensure_ascii=False,
            )
        )

    try:
        rows = _fetch_rows(user, args)
    except Exception:  # noqa: BLE001 — сбой инструмента не должен ронять чат
        current_app.logger.exception("search_analyses: ошибка выборки")
        return ToolResult(json.dumps({"error": "не удалось получить данные из базы"}, ensure_ascii=False))

    show_user = user.is_head_admin and not args["own_only"]
    by_risk: dict[str, int] = {}
    for row in rows:
        label = RISK_LABELS.get(row.risk_level, row.risk_level)
        by_risk[label] = by_risk.get(label, 0) + 1

    payload: dict = {
        "scope": "все пользователи" if show_user else "только анализы самого пользователя",
        "total_matched": len(rows),
        "by_risk": by_risk,
        "needs_review_count": sum(1 for r in rows if r.needs_human_review),
    }
    if len(rows) >= _SCAN_LIMIT:
        payload["note"] = f"учтены только {_SCAN_LIMIT} самых свежих записей"
    if warnings:
        payload["warnings"] = warnings

    references: list = []
    if not args["count_only"]:
        shown = rows[: args["limit"]]
        payload["records"] = [_record(r, show_user) for r in shown]
        payload["records_shown"] = len(shown)
        references = _build_reference_cards(shown, show_user)

    return ToolResult(json.dumps(payload, ensure_ascii=False), references)


# ---------------------------------------------------------------------------
# TODO (не реализовано):
#
# 1. Семантический поиск по description — отдельный аргумент инструмента
#    (например `similar_to`), если появятся вопросы «покажи анализы, похожие
#    на Х». Понадобится embedding description при сохранении AnalysisResult и
#    векторное хранилище (sqlite-vec / pgvector).
# 2. Если analysis_results вырастет настолько, что скан signals в Python
#    станет узким местом — вынести категории в таблицу analysis_categories
#    (analysis_id, category_name) с индексом и фильтровать JOIN'ом в SQL.
# ---------------------------------------------------------------------------
