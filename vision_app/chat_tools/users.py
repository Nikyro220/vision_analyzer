"""Инструмент `search_users`: поиск, просмотр и статистика пользователей системы.

Доступен только staff-ролям (admin, head_admin) — на ТРЁХ уровнях:
  1. runner.build_tool_system_prompt: обычному пользователю инструмент не описывается;
  2. runner.parse_reply: для него имя инструмента — «неизвестное», вызов не разбирается;
  3. search_users (ниже): финальная проверка прав на сервере.
Обычный admin видит только рядовых пользователей (user + blocked); head_admin — всех.

Что умеет: поиск по логину / email / ФИО, фильтры (роль, активность, дата регистрации,
наличие анализов), разбивка по ролям, сортировка (в том числе «кто больше всех анализирует»,
в т.ч. за период), по каждому пользователю — контакты и сводка по его анализам.
Инструмент только читает; хэши паролей и прочие секреты наружу не отдаются.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from flask import url_for
from sqlalchemy import case, func, or_, select

from ..config import conf
from ..extensions import db
from ..models import AnalysisResult, ChatSession, Role, ROLE_LABELS, RiskLevel, Status, User
from ..utils import local_dt

TOOL_NAME = "search_users"

# Лимиты живут в config.py (USER_SEARCH_*); публичные имена используются и в промпте (runner.py).
def default_limit() -> int:
    return conf("USER_SEARCH_DEFAULT_LIMIT")


def max_limit() -> int:
    return conf("USER_SEARCH_MAX_LIMIT")


def max_since_days() -> int:
    return conf("USER_SEARCH_MAX_SINCE_DAYS")


@dataclass
class ToolResult:
    text: str
    references: list = field(default_factory=list)


def _error(message: str, **extra) -> ToolResult:
    return ToolResult(json.dumps({"error": message, **extra}, ensure_ascii=False))


def _int_arg(value, lo: int, hi: int) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    if n < lo:
        return None
    return min(hi, n)


def _bool_arg(value):
    """None — аргумент не передан; True/False — значение."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"true", "1", "yes", "да"}:
            return True
        if lower in {"false", "0", "no", "нет"}:
            return False
    return None


_VALID_ROLES = {Role.USER, Role.ADMIN, Role.HEAD_ADMIN, Role.BLOCKED}

SORT_NEWEST = "newest"
SORT_OLDEST = "oldest"
SORT_ANALYSES = "analyses"  # больше всего анализов — первыми
SORT_NAME = "name"
_VALID_SORTS = {SORT_NEWEST, SORT_OLDEST, SORT_ANALYSES, SORT_NAME}


def normalize_args(raw) -> tuple[dict, list[str]]:
    warnings: list[str] = []
    if not isinstance(raw, dict):
        warnings.append("args должен быть объектом — использованы значения по умолчанию")
        raw = {}

    known = {
        "query", "username", "role", "active", "limit", "count_only", "since_days",
        "analyses_days", "sort", "has_analyses",
    }
    unknown = sorted(str(k) for k in raw if k not in known)
    if unknown:
        warnings.append("неизвестные аргументы проигнорированы: " + ", ".join(unknown))

    args: dict = {
        "query": None,
        "role": None,
        "active": _bool_arg(raw.get("active")),
        "has_analyses": _bool_arg(raw.get("has_analyses")),
        "limit": _int_arg(raw.get("limit"), 1, max_limit()) or default_limit(),
        "count_only": bool(_bool_arg(raw.get("count_only"))),
        "since_days": _int_arg(raw.get("since_days"), 1, max_since_days()),
        "analyses_days": _int_arg(raw.get("analyses_days"), 1, max_since_days()),
        "sort": SORT_NEWEST,
    }

    # Поиск: частичный регистронезависимый по логину, email и ФИО. `username` — прежнее имя аргумента.
    text = raw.get("query") if raw.get("query") not in (None, "") else raw.get("username")
    if isinstance(text, str) and text.strip():
        args["query"] = text.strip()[:150]

    role = raw.get("role")
    if role is not None and role != "":
        role_str = str(role).strip().lower()
        if role_str in _VALID_ROLES:
            args["role"] = role_str
        else:
            warnings.append(f"роль '{role}' не распознана; доступные: " + ", ".join(sorted(_VALID_ROLES)))

    sort = raw.get("sort")
    if sort is not None and sort != "":
        sort_str = str(sort).strip().lower()
        if sort_str in _VALID_SORTS:
            args["sort"] = sort_str
        else:
            warnings.append(f"sort '{sort}' не распознан; доступные: " + ", ".join(sorted(_VALID_SORTS)))

    return args, warnings


def _like_pattern(text: str) -> str:
    """Подстрока для LIKE: служебные % и _ из ввода модели теряют особый смысл."""
    escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _analyses_subquery(days: int | None):
    """Число анализов, из них с высоким риском (готовые), на пользователя — за период или за всё время."""
    conditions = []
    if days:
        conditions.append(AnalysisResult.created_at >= datetime.now(timezone.utc) - timedelta(days=days))
    high = case(((AnalysisResult.status == Status.DONE) & (AnalysisResult.risk_level == RiskLevel.HIGH), 1), else_=0)
    return (
        select(
            AnalysisResult.user_id.label("uid"),
            func.count(AnalysisResult.id).label("cnt"),
            func.coalesce(func.sum(high), 0).label("high"),
        )
        .where(*conditions)
        .group_by(AnalysisResult.user_id)
        .subquery()
    )


def _display_name(u: User) -> str:
    return u.full_name or ""


def user_card(u: User, analyses_count: int) -> dict:
    """Карточка-ссылка на страницу пользователя в панели под ответом (рисует static/js/chat.js
    по kind == "user"). Ссылка ведёт на panel.user_detail — страницу, доступную только staff."""
    return {
        "kind": "user",
        "id": u.id,
        "url": url_for("panel.user_detail", pk=u.id),
        "label": u.username,
        "subtitle": u.full_name or "",
        "role": u.role,
        "role_label": ROLE_LABELS.get(u.role, u.role),
        "active": bool(u.active),
        "avatar_url": u.avatar_src,
        "avatar_color": u.color_hex,
        "avatar_fg": u.color_fg,
        "initials": u.initials,
        "analyses_count": int(analyses_count),
        "date": local_dt(u.created_at, "date") if u.created_at else "",
    }


def search_users(user: User, raw_args) -> ToolResult:
    """Точка входа: аргументы от модели → результат."""
    # Права: только staff (повторная проверка — промпт и разбор вызова уже отсекают остальных)
    if not user.is_panel_staff:
        return _error("инструмент search_users доступен только администраторам")

    args, warnings = normalize_args(raw_args)
    stats = _analyses_subquery(args["analyses_days"])
    analyses_cnt = func.coalesce(stats.c.cnt, 0)

    # Права: head_admin — все пользователи; admin — только user и blocked.
    conditions = []
    if not user.is_head_admin:
        conditions.append(User.role.in_([Role.USER, Role.BLOCKED]))

    if args["query"]:
        pattern = _like_pattern(args["query"])
        conditions.append(
            or_(*(col.ilike(pattern, escape="\\") for col in (
                User.username, User.email, User.nickname, User.family_name, User.given_name,
            )))
        )
    if args["role"]:
        conditions.append(User.role == args["role"])
    if args["active"] is not None:
        conditions.append(User.active == args["active"])
    if args["since_days"]:
        conditions.append(User.created_at >= datetime.now(timezone.utc) - timedelta(days=args["since_days"]))
    if args["has_analyses"] is True:
        conditions.append(analyses_cnt > 0)
    elif args["has_analyses"] is False:
        conditions.append(analyses_cnt == 0)

    base = User.__table__.outerjoin(stats, stats.c.uid == User.id)

    try:
        total = int(db.session.scalar(select(func.count()).select_from(base).where(*conditions)) or 0)
        by_role_rows = db.session.execute(
            select(User.role, func.count()).select_from(base).where(*conditions).group_by(User.role)
        ).all()
    except Exception:  # noqa: BLE001
        return _error("не удалось получить данные пользователей из базы")

    payload: dict = {
        "scope": "все пользователи" if user.is_head_admin else "только пользователи и заблокированные (без администраторов)",
        "total_matched": total,
        "by_role": {ROLE_LABELS.get(role, role): cnt for role, cnt in sorted(by_role_rows)},
    }
    if args["analyses_days"]:
        payload["analyses_period_days"] = args["analyses_days"]
    if warnings:
        payload["warnings"] = warnings

    if args["count_only"]:
        return ToolResult(json.dumps(payload, ensure_ascii=False))

    order = {
        SORT_NEWEST: (User.created_at.desc(), User.id.desc()),
        SORT_OLDEST: (User.created_at.asc(), User.id.asc()),
        SORT_ANALYSES: (analyses_cnt.desc(), User.created_at.asc(), User.id.asc()),
        SORT_NAME: (func.lower(User.username).asc(), User.id.asc()),
    }[args["sort"]]
    try:
        rows = db.session.execute(
            select(User, analyses_cnt.label("cnt"), func.coalesce(stats.c.high, 0).label("high"))
            .select_from(base)
            .where(*conditions)
            .order_by(*order)
            .limit(args["limit"])
        ).all()

        # Остальное — одним батч-запросом на всех показанных (не N+1)
        ids = [r.User.id for r in rows]
        last_analysis: dict[int, datetime] = {}
        chats: dict[int, int] = {}
        if ids:
            last_analysis = dict(db.session.execute(
                select(AnalysisResult.user_id, func.max(AnalysisResult.created_at))
                .where(AnalysisResult.user_id.in_(ids)).group_by(AnalysisResult.user_id)
            ).all())
            chats = dict(db.session.execute(
                select(ChatSession.user_id, func.count(ChatSession.id))
                .where(ChatSession.user_id.in_(ids)).group_by(ChatSession.user_id)
            ).all())
    except Exception:  # noqa: BLE001
        return _error("не удалось получить список пользователей из базы")

    payload["sorted_by"] = args["sort"]
    payload["records"] = [
        {
            "id": u.id,
            "username": u.username,
            "email": u.email or "",
            "full_name": _display_name(u),
            "role": u.role,
            "role_label": ROLE_LABELS.get(u.role, u.role),
            "active": u.active,
            "registered": local_dt(u.created_at, "date") if u.created_at else "неизвестно",
            "analyses_count": int(cnt),  # за analyses_period_days, если он указан, иначе за всё время
            "high_risk_count": int(high),
            "last_analysis": local_dt(last_analysis[u.id], "short") if u.id in last_analysis else "нет",
            "chats_count": chats.get(u.id, 0),
        }
        for u, cnt, high in rows
    ]
    payload["records_shown"] = len(rows)

    cards = [user_card(u, cnt) for u, cnt, _ in rows[: conf("USER_SEARCH_MAX_CARDS")]]
    if cards:
        payload["cards_shown"] = len(cards)  # интерфейс сам нарисует карточки со ссылками на страницы
    return ToolResult(json.dumps(payload, ensure_ascii=False), references=cards)
