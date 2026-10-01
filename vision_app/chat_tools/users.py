"""Инструмент `search_users`: поиск и просмотр пользователей системы.

Доступен только staff-ролям (admin, head_admin). Права применяются здесь,
на сервере: обычный admin видит только рядовых пользователей (user + blocked);
head_admin видит всех включая других admin/head_admin.

Инструмент только читает. Сортировка — от новых к старым.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select

from ..config import conf
from ..extensions import db
from ..models import AnalysisResult, Role, ROLE_LABELS, User
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


def normalize_args(raw) -> tuple[dict, list[str]]:
    warnings: list[str] = []
    if not isinstance(raw, dict):
        warnings.append("args должен быть объектом — использованы значения по умолчанию")
        raw = {}

    known = {"username", "role", "active", "limit", "count_only", "since_days"}
    unknown = sorted(str(k) for k in raw if k not in known)
    if unknown:
        warnings.append("неизвестные аргументы проигнорированы: " + ", ".join(unknown))

    args: dict = {
        "username": None,
        "role": None,
        "active": None,
        "limit": _int_arg(raw.get("limit"), 1, max_limit()) or default_limit(),
        "count_only": bool(raw.get("count_only")),
        "since_days": _int_arg(raw.get("since_days"), 1, max_since_days()),
    }

    # Имя пользователя: частичный регистронезависимый поиск (LIKE)
    username = raw.get("username")
    if isinstance(username, str) and username.strip():
        args["username"] = username.strip()[:150]

    # Роль: строгое совпадение, только допустимые значения
    role = raw.get("role")
    if role is not None and role != "":
        role_str = str(role).strip().lower()
        if role_str in _VALID_ROLES:
            args["role"] = role_str
        else:
            warnings.append(
                f"роль '{role}' не распознана; доступные: "
                + ", ".join(sorted(_VALID_ROLES))
            )

    active = _bool_arg(raw.get("active"))
    if active is not None:
        args["active"] = active

    return args, warnings


def search_users(user: User, raw_args) -> ToolResult:
    """Точка входа: аргументы от модели → результат."""
    # Права: только staff
    if not user.is_panel_staff:
        return _error("инструмент search_users доступен только администраторам")

    args, warnings = normalize_args(raw_args)

    # Базовый запрос + ограничение прав:
    #   head_admin — все пользователи
    #   admin — только user и blocked (не другие админы)
    stmt = select(User)
    if not user.is_head_admin:
        stmt = stmt.where(User.role.in_([Role.USER, Role.BLOCKED]))

    # --- Фильтры ---
    if args["username"]:
        stmt = stmt.where(User.username.ilike(f"%{args['username']}%"))
    if args["role"]:
        # Дополнительное сужение по роли (права уже ограничили сверху)
        stmt = stmt.where(User.role == args["role"])
    if args["active"] is not None:
        stmt = stmt.where(User.active == args["active"])
    if args["since_days"]:
        cutoff = datetime.now(timezone.utc) - timedelta(days=args["since_days"])
        stmt = stmt.where(User.created_at >= cutoff)

    # Общий счётчик (без LIMIT)
    count_stmt = stmt.with_only_columns(func.count()).order_by(None)
    try:
        total = int(db.session.scalar(count_stmt) or 0)
    except Exception:  # noqa: BLE001
        return _error("не удалось получить данные пользователей из базы")

    payload: dict = {"total_matched": total}
    if warnings:
        payload["warnings"] = warnings

    if not args["count_only"]:
        try:
            users = db.session.scalars(
                stmt.order_by(User.created_at.desc()).limit(args["limit"])
            ).all()
        except Exception:  # noqa: BLE001
            return _error("не удалось получить список пользователей из базы")

        # Количество анализов одним батч-запросом (не N+1)
        user_ids = [u.id for u in users]
        analysis_counts: dict[int, int] = {}
        if user_ids:
            rows = db.session.execute(
                select(AnalysisResult.user_id, func.count().label("cnt"))
                .where(AnalysisResult.user_id.in_(user_ids))
                .group_by(AnalysisResult.user_id)
            ).all()
            analysis_counts = {r.user_id: r.cnt for r in rows}

        payload["records"] = [
            {
                "id": u.id,
                "username": u.username,
                "role": u.role,
                "role_label": ROLE_LABELS.get(u.role, u.role),
                "active": u.active,
                "registered": local_dt(u.created_at, "date") if u.created_at else "неизвестно",
                "analyses_count": analysis_counts.get(u.id, 0),
            }
            for u in users
        ]
        payload["records_shown"] = len(users)

    return ToolResult(json.dumps(payload, ensure_ascii=False))
