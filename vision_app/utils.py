from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from flask import request, url_for

from .config import conf
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


def query_to_id(query: str | None) -> int | None:
    """«42», «#42» или «№42» -> 42; всё остальное -> None. Нужен, чтобы в поиске по истории
    можно было набрать номер анализа — тот самый, который пользователь видит в интерфейсе."""
    text = (query or "").strip().lstrip("#№").strip()
    if text.isdigit() and len(text) <= 9:
        return int(text)
    return None


def is_safe_next(target: str | None) -> bool:
    """Разрешаем редирект только на относительные пути внутри сайта."""
    return bool(
        target
        and target.startswith("/")
        and not target.startswith("//")
        and "\\" not in target
    )


# Именованные форматы дат: в коде и шаблонах пишем localdt("short"), а сами строки
# формата живут в Config (DATETIME_FORMAT, DATETIME_SHORT_FORMAT, DATE_FORMAT, ...).
_DT_FORMAT_KEYS = {
    "full": "DATETIME_FORMAT",
    "short": "DATETIME_SHORT_FORMAT",
    "date": "DATE_FORMAT",
    "chat": "DATETIME_CHAT_FORMAT",
    "time": "TIME_FORMAT",
}


def local_dt(value: datetime | None, fmt: str | None = None) -> str:
    """Фильтр Jinja: UTC из БД -> локальное время (APP_TIMEZONE) -> строка.

    fmt — имя формата ("full" по умолчанию, "short", "date", "chat") либо готовая
    строка strftime."""
    if value is None:
        return "—"
    fmt = conf(_DT_FORMAT_KEYS[fmt or "full"]) if (fmt or "full") in _DT_FORMAT_KEYS else fmt
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    tz = ZoneInfo(conf("TIMEZONE"))
    return value.astimezone(tz).strftime(fmt)


def truncate_chars(value, length: int = 30) -> str:
    """Аналог Django truncatechars: итоговая длина не больше length, с «…» на конце."""
    text = "" if value is None else str(value)
    if len(text) <= length:
        return text
    return text[: max(length - 1, 0)] + "…"


def plural(n: int, forms: tuple[str, str, str]) -> str:
    """Русское склонение: plural(1, ("файл","файла","файлов")) -> "1 файл"."""
    n_abs = abs(n)
    if n_abs % 10 == 1 and n_abs % 100 != 11:
        word = forms[0]
    elif 2 <= n_abs % 10 <= 4 and not 12 <= n_abs % 100 <= 14:
        word = forms[1]
    else:
        word = forms[2]
    return f"{n} {word}"
