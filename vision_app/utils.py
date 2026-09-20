from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from flask import current_app, request, url_for

from .extensions import db


def paginate(stmt, per_page: int):
    """Пагинация по ?page=N. Некорректная страница -> первая, за пределами -> последняя."""
    page = max(request.args.get("page", 1, type=int) or 1, 1)
    result = db.paginate(stmt, page=page, per_page=per_page, error_out=False)
    if result.pages and page > result.pages:
        result = db.paginate(stmt, page=result.pages, per_page=per_page, error_out=False)
    return result


def page_url(page_number: int) -> str:
    """URL текущей страницы с другим номером страницы (остальные параметры сохраняются)."""
    args = request.args.to_dict(flat=True)
    args["page"] = page_number
    return url_for(request.endpoint, **{**(request.view_args or {}), **args})


def is_safe_next(target: str | None) -> bool:
    """Разрешаем редирект только на относительные пути внутри сайта."""
    return bool(
        target
        and target.startswith("/")
        and not target.startswith("//")
        and "\\" not in target
    )


def local_dt(value: datetime | None, fmt: str = "%d.%m.%Y %H:%M") -> str:
    """Фильтр Jinja: UTC из БД -> локальное время (APP_TIMEZONE) -> строка."""
    if value is None:
        return "—"
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    tz = ZoneInfo(current_app.config.get("TIMEZONE", "UTC"))
    return value.astimezone(tz).strftime(fmt)


def truncate_chars(value, length: int = 30) -> str:
    """Аналог Django truncatechars: итоговая длина не больше length, с «…» на конце."""
    text = "" if value is None else str(value)
    if len(text) <= length:
        return text
    return text[: max(length - 1, 0)] + "…"
