"""Выбор бэкенда/модели для анализа. Хранится в БД, поэтому переживает перезапуск
и действует для всех пользователей.

Сервер vision_analyzer_server.py не хранит «выбранную модель» глобально — модель
передаётся в каждом запросе (?backend=...&model=...). Поэтому приложение само помнит
выбор и подставляет его при каждом вызове /analyze.
"""

from __future__ import annotations

from .extensions import db
from .models import Setting
from .services import BACKENDS

KEY_BACKEND = "analysis_backend"
KEY_MODEL = "analysis_model"


def _get(key: str) -> str:
    row = db.session.get(Setting, key)
    return row.value if row else ""


def _set(key: str, value: str) -> None:
    row = db.session.get(Setting, key)
    if row is None:
        db.session.add(Setting(key=key, value=value))
    else:
        row.value = value


def get_analysis_target() -> tuple[str, str]:
    """(backend, model). Пустые строки — «как решит сервер» (бэкенд по умолчанию, модель авто)."""
    backend = _get(KEY_BACKEND)
    if backend not in BACKENDS:
        return "", ""
    return backend, _get(KEY_MODEL)


def set_analysis_target(backend: str, model: str = "") -> None:
    _set(KEY_BACKEND, backend)
    _set(KEY_MODEL, model)
    db.session.commit()


def clear_analysis_target() -> None:
    _set(KEY_BACKEND, "")
    _set(KEY_MODEL, "")
    db.session.commit()
