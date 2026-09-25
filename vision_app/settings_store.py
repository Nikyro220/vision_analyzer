"""Выбор бэкенда/модели для анализа. Хранится в БД, поэтому переживает перезапуск
и действует для всех пользователей.

Сервер vision_analyzer_server.py не хранит «выбранную модель» глобально — модель
передаётся в каждом запросе (?backend=...&model=...). Поэтому приложение само помнит
выбор и подставляет его при каждом вызове /analyze.
"""

from __future__ import annotations

from dataclasses import dataclass

from flask import current_app

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


# ----------------------------------------------------------------------------
# Настройки, изменяемые главным админом на странице /panel/settings/.
#
# Значение хранится в Setting (ключ "cfg:<ИМЯ>") и переопределяет одноимённую
# переменную окружения / вход в vision_app.config.Config. Без переопределения
# (запись в БД отсутствует) в ход идёт обычный app.config — то есть до первого
# сохранения в этой панели всё работает ровно как раньше, через .env.
# Изменения действуют СРАЗУ, без перезапуска: каждое место, где нужно текущее
# значение, читает его заново (см. вызовы get_runtime_setting ниже по коду).
# ----------------------------------------------------------------------------
_CFG_PREFIX = "cfg:"


@dataclass(frozen=True)
class RuntimeSetting:
    key: str            # то же имя, что у переменной окружения / ключа app.config
    label: str           # подпись в форме
    kind: str             # "bool" | "int" | "float" | "str"
    hint: str = ""
    min: float | None = None
    max: float | None = None


RUNTIME_SETTINGS: list[RuntimeSetting] = [
    RuntimeSetting(
        "QUEUE_WORKER_ENABLED", "Обработка очереди включена", "bool",
        "Если выключить, загруженные файлы будут копиться в очереди без обработки.",
    ),
    RuntimeSetting(
        "QUEUE_WORKERS", "Потоков-обработчиков", "int",
        "Сколько анализов выполнять параллельно.", min=1, max=8,
    ),
    RuntimeSetting(
        "QUEUE_POLL_SECONDS", "Интервал опроса очереди, сек", "float",
        "Как часто обработчик проверяет пустую очередь.", min=1, max=300,
    ),
    RuntimeSetting(
        "QUEUE_MAX_FILES_PER_UPLOAD", "Файлов за одну загрузку", "int",
        min=1, max=500,
    ),
    RuntimeSetting(
        "QUEUE_MAX_PENDING_PER_USER", "Задач в очереди на пользователя", "int",
        min=1, max=1000,
    ),
    RuntimeSetting(
        "VISION_API_BASE_URL", "Адрес сервера анализа", "str",
        "Например http://127.0.0.1:6769",
    ),
    RuntimeSetting(
        "VISION_API_TIMEOUT", "Таймаут запроса к /analyze, сек", "int",
        min=5, max=3600,
    ),
]
RUNTIME_SETTINGS_BY_KEY = {s.key: s for s in RUNTIME_SETTINGS}


def _cast(spec: RuntimeSetting, raw: str):
    if spec.kind == "bool":
        return raw == "1"
    if spec.kind == "int":
        return int(raw)
    if spec.kind == "float":
        return float(raw)
    return raw


def get_runtime_setting(key: str):
    """Текущее действующее значение: из БД, если оно переопределено, иначе из app.config
    (то есть из переменной окружения или дефолта в Config/queue_worker.py)."""
    spec = RUNTIME_SETTINGS_BY_KEY[key]
    row = db.session.get(Setting, _CFG_PREFIX + key)
    if row is not None and row.value != "":
        return _cast(spec, row.value)
    return current_app.config.get(key)


def is_runtime_setting_overridden(key: str) -> bool:
    row = db.session.get(Setting, _CFG_PREFIX + key)
    return row is not None and row.value != ""


def set_runtime_setting(key: str, raw_value: str) -> None:
    """Валидирует и сохраняет переопределение. Бросает ValueError с русским сообщением."""
    spec = RUNTIME_SETTINGS_BY_KEY[key]
    raw = raw_value.strip()

    if spec.kind == "bool":
        value_str = "1" if raw_value else "0"
    else:
        try:
            value = _cast(spec, raw.replace(",", "."))
        except ValueError:
            kind_hint = "целое число" if spec.kind == "int" else "число"
            raise ValueError(f"«{spec.label}»: введите {kind_hint}.") from None
        if spec.kind in ("int", "float"):
            if spec.min is not None and value < spec.min:
                raise ValueError(f"«{spec.label}»: значение не может быть меньше {spec.min:g}.")
            if spec.max is not None and value > spec.max:
                raise ValueError(f"«{spec.label}»: значение не может быть больше {spec.max:g}.")
        elif spec.kind == "str":
            if not value:
                raise ValueError(f"«{spec.label}»: значение не может быть пустым.")
            if len(value) > 300:
                raise ValueError(f"«{spec.label}»: слишком длинное значение.")
        value_str = str(value)

    _set(_CFG_PREFIX + key, value_str)


def reset_runtime_setting(key: str) -> None:
    """Убирает переопределение — настройка возвращается к значению из .env/дефолта."""
    row = db.session.get(Setting, _CFG_PREFIX + key)
    if row is not None:
        db.session.delete(row)