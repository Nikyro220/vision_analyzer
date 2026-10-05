"""Выбор бэкенда/модели для анализа. Хранится в БД, поэтому переживает перезапуск
и действует для всех пользователей.

Сервер vision_analyzer_server.py не хранит «выбранную модель» глобально — модель
передаётся в каждом запросе (?backend=...&model=...). Поэтому приложение само помнит
выбор и подставляет его при каждом вызове /analyze.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from flask import current_app

from . import config as _config
from .config import NO_OVERRIDE, conf
from .extensions import db
from .models import Setting
from .services import is_known_backend

log = logging.getLogger("vision_app.settings_store")

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
    # Список бэкендов не зашит в панели: его отдаёт сервер анализа (services.get_providers).
    # False — сервер такого провайдера уже не знает (убрали из inference/providers/): выбор
    # недействителен. None — сервер не ответил: сохранённый выбор не трогаем.
    if not backend or is_known_backend(backend) is False:
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
# Значение хранится в Setting (ключ "cfg:<ИМЯ>") и переопределяет одноимённую константу из
# vision_app/config.py (или переменную окружения, из которой она берётся). Нет записи в БД —
# действует обычное значение из config.py/.env, то есть до первого сохранения всё работает как
# раньше.
#
# Изменения применяются СРАЗУ, без перезапуска: config.conf() при каждом чтении смотрит в
# переопределения (они кэшируются на _CACHE_TTL секунд, чтобы не ходить в БД на каждое чтение;
# после сохранения кэш процесса сбрасывается, остальные процессы подхватят за ≤ _CACHE_TTL с).
# Ключи самого Flask (лимит запроса, сессия, cookie, CSRF) дополнительно переносятся в
# app.config перед каждым запросом — см. sync_flask_config().
#
# В список НЕ входят настройки, которые нельзя или опасно менять на ходу: SECRET_KEY, строка
# подключения к БД, пути и имена каталогов с файлами, логирование, параметры подключения SQLite.
# ----------------------------------------------------------------------------
_CFG_PREFIX = "cfg:"
_CACHE_TTL = 2.0


@dataclass(frozen=True)
class RuntimeSetting:
    key: str             # то же имя, что у константы в config.py
    label: str           # подпись в форме
    kind: str            # "bool" | "int" | "float" | "str" | "choice" | "days" | "opt_int"
    hint: str = ""
    min: float | None = None
    max: float | None = None
    min_key: str | None = None   # границы, заданные другими настройками (QUEUE_WORKERS_MIN и т. п.)
    max_key: str | None = None
    scale: int = 1               # int в форме = значение / scale (например, МБ при scale=1024*1024)
    choices: tuple[str, ...] = ()
    validator: str = ""          # "url" | "timezone" | "strftime"


def _s(key, label, kind="int", lo=None, hi=None, hint="", **kw) -> RuntimeSetting:
    return RuntimeSetting(key, label, kind, hint, lo, hi, **kw)


_MB = 1024 * 1024

# (название группы, пояснение, настройки) — в том порядке, в каком показываем на странице.
SETTING_GROUPS: list[tuple[str, str, list[RuntimeSetting]]] = [
    ("Сервер анализа", "Клиент vision-сервера: адрес, таймауты, язык ответов.", [
        _s("VISION_API_BASE_URL", "Адрес сервера анализа", "str", hint="Например http://127.0.0.1:6769", validator="url"),
        _s("VISION_API_TIMEOUT", "Таймаут /analyze и /chat, сек", min_key="VISION_API_TIMEOUT_MIN", max_key="VISION_API_TIMEOUT_MAX"),
        _s("VISION_API_HEALTH_TIMEOUT", "Таймаут /health и GET /embeddings, сек", lo=1, hi=300),
        _s("VISION_API_CALL_TIMEOUT", "Таймаут вспомогательных запросов, сек", lo=1, hi=600,
           hint="/models, /sampling, /categories и т. п."),
        _s("VISION_API_MODELS_TIMEOUT", "Таймаут списка моделей облачного провайдера, сек", lo=1, hi=600,
           hint="/models облачных провайдеров (Gemini, Anthropic); отдельно от общего таймаута вспомогательных запросов."),
        _s("MODELS_CACHE_TTL", "Кэш списка моделей облачного провайдера, сек", lo=0, hi=2592000,
           hint="Сколько список моделей облачного провайдера хранится в БД до автообновления. 0 — не кэшировать."),
        _s("VISION_API_EMBED_TIMEOUT", "Таймаут POST /embeddings, сек", lo=1, hi=600),
        _s("VISION_API_ERROR_CHARS", "Длина текста ошибки сервера, симв.", lo=20, hi=5000,
           hint="Сколько символов ответа-ошибки показывать пользователю."),
        _s("VISION_API_RAW_TEXT_CHARS", "Усечение «сырого» ответа модели, симв.", lo=100, hi=100000,
           hint="Для ответов без валидного JSON."),
        _s("VISION_API_BACKEND_NAME_CHARS", "Длина названия бэкенда, симв.", lo=1, hi=32,
           hint="Не больше 32 — столько влезает в колонку БД."),
        _s("VISION_API_MODEL_NAME_CHARS", "Длина названия модели, симв.", lo=1, hi=120,
           hint="Не больше 120 — столько влезает в колонку БД."),
        _s("DEFAULT_LANG", "Язык ответов модели", "choice", choices=("ru", "en"),
           hint="Передаётся серверу анализа как lang=..."),
    ]),
    ("Очередь анализов", "Фоновая обработка загруженных изображений.", [
        _s("QUEUE_WORKER_ENABLED", "Обработка очереди включена", "bool",
           hint="Если выключить, загруженные файлы будут копиться в очереди без обработки."),
        _s("QUEUE_WORKERS", "Потоков-обработчиков", hint="Сколько анализов выполнять параллельно.",
           min_key="QUEUE_WORKERS_MIN", max_key="QUEUE_WORKERS_MAX"),
        _s("QUEUE_POLL_SECONDS", "Интервал опроса очереди, сек", "float",
           hint="Как часто обработчик проверяет пустую очередь.",
           min_key="QUEUE_POLL_SECONDS_MIN", max_key="QUEUE_POLL_SECONDS_MAX"),
        _s("QUEUE_MAX_FILES_PER_UPLOAD", "Файлов за одну загрузку",
           min_key="QUEUE_MAX_FILES_PER_UPLOAD_MIN", max_key="QUEUE_MAX_FILES_PER_UPLOAD_MAX"),
        _s("QUEUE_MAX_PENDING_PER_USER", "Задач в очереди на пользователя",
           min_key="QUEUE_MAX_PENDING_PER_USER_MIN", max_key="QUEUE_MAX_PENDING_PER_USER_MAX"),
        _s("QUEUE_ERROR_PAUSE_SECONDS", "Пауза после ошибки цикла, сек", lo=0, hi=300),
        _s("QUEUE_CLAIM_RETRIES", "Попыток захвата задачи при гонке", lo=1, hi=100),
        _s("QUEUE_STOP_JOIN_TIMEOUT", "Ожидание остановки потока, сек", lo=0, hi=300),
    ]),
    ("Границы допустимых значений", "Пределы, в которых можно выбирать значения выше. Нижняя граница не может быть больше верхней.", [
        _s("QUEUE_WORKERS_MIN", "Потоков-обработчиков: минимум", lo=1, hi=64),
        _s("QUEUE_WORKERS_MAX", "Потоков-обработчиков: максимум", lo=1, hi=64,
           hint="Потолок защищает от опечатки вроде 100 потоков."),
        _s("QUEUE_POLL_SECONDS_MIN", "Интервал опроса: минимум, сек", lo=1, hi=3600),
        _s("QUEUE_POLL_SECONDS_MAX", "Интервал опроса: максимум, сек", lo=1, hi=3600),
        _s("QUEUE_MAX_FILES_PER_UPLOAD_MIN", "Файлов за загрузку: минимум", lo=1, hi=10000),
        _s("QUEUE_MAX_FILES_PER_UPLOAD_MAX", "Файлов за загрузку: максимум", lo=1, hi=10000),
        _s("QUEUE_MAX_PENDING_PER_USER_MIN", "Задач на пользователя: минимум", lo=1, hi=100000),
        _s("QUEUE_MAX_PENDING_PER_USER_MAX", "Задач на пользователя: максимум", lo=1, hi=100000),
        _s("VISION_API_TIMEOUT_MIN", "Таймаут /analyze: минимум, сек", lo=1, hi=86400),
        _s("VISION_API_TIMEOUT_MAX", "Таймаут /analyze: максимум, сек", lo=1, hi=86400),
        _s("SETTING_STR_MAX_LEN", "Максимальная длина строковой настройки", lo=10, hi=2000),
    ]),
    ("Поиск по смыслу", "Эмбеддинги и фоновая индексация описаний анализов.", [
        _s("EMBEDDING_MIN_SIMILARITY", "Порог сходства (cosine)", "float", 0, 1,
           hint="Ниже этого значения результат в выдачу не попадает. Если выдача шумная — поднимите."),
        _s("EMBEDDING_RELATIVE_MARGIN", "Отступ от лучшего результата", "float", 0, 1,
           hint="В выдачу идёт только то, что не ниже «лучший скор − отступ». 0 — выключить."),
        _s("EMBEDDING_SCAN_LIMIT", "Сколько векторов сравнивать за запрос", lo=1, hi=1000000),
        _s("EMBEDDING_MEMORY_CACHE", "Кэш векторов в памяти", "bool",
           hint="Не читать векторы из БД на каждый поиск. Если выключить — поиск работает как раньше, но медленнее."),
        _s("EMBEDDING_BATCH_SIZE", "Размер батча для сервера", lo=1, hi=64, hint="Сервер принимает максимум 64 текста за запрос."),
        _s("EMBEDDING_MAX_TEXT_CHARS", "Максимальная длина текста для эмбеддинга, симв.", lo=100, hi=8000,
           hint="Серверный лимит — 8000."),
        _s("EMBEDDING_MAX_MATCHED_IDS", "Сколько id выше порога держать для статистики", lo=1, hi=100000),
        _s("EMBEDDING_IDLE_BATCHES", "Батчей за один холостой проход", lo=1, hi=1000),
        _s("EMBEDDING_IDLE_PAUSE_UNAVAILABLE", "Пауза: эмбеддинги выключены/недоступны, сек", lo=1, hi=86400),
        _s("EMBEDDING_IDLE_PAUSE_NOTHING", "Пауза: индексировать нечего, сек", lo=1, hi=86400),
        _s("EMBEDDING_IDLE_PAUSE_RETRY", "Пауза: модель качается / сбой, сек", lo=1, hi=86400),
    ]),
    ("Чат", "Сообщения, история и изображения в чате с моделью.", [
        _s("CHAT_MAX_MESSAGE_LEN", "Максимальная длина сообщения, симв.", lo=10, hi=200000),
        _s("CHAT_MAX_HISTORY_MESSAGES", "Сообщений истории, уходящих модели", lo=1, hi=1000),
        _s("CHAT_TITLE_LEN", "Длина автоназвания чата, симв.", lo=5, hi=200),
        _s("CHAT_TURN_STALE_SECONDS", "«Протухание» ответа модели, сек", lo=30, hi=86400,
           hint="Если сервер упал посреди ответа, чат снова примет сообщения через это время."),
        _s("CHAT_MAX_TOOL_CALLS", "Обращений к инструментам за один ход", lo=1, hi=20),
        _s("CHAT_PROMPT_MAX_CHARS", "Длина системного промпта чата, симв.", lo=200, hi=50000,
           hint="Ограничение для поля «Системный промпт чата» в разделе «Промпты»."),
        _s("CHAT_CONTEXT_IMAGES", "Вложений, которые модель «видит» напрямую", lo=0, hi=50,
           hint="Каждая картинка — порядка 1–1,5 тыс. токенов контекста."),
        _s("CHAT_MAX_IMAGES_PER_MESSAGE", "Изображений в одном сообщении", lo=1, hi=50),
        _s("CHAT_MAX_IMAGE_BYTES", "Максимальный размер одного изображения, МБ", lo=1, hi=500, scale=_MB,
           hint="Общий лимит запроса — «Максимальный размер запроса»."),
        _s("CHAT_ATTACHMENT_NAME_MAX", "Длина имени вложения, симв.", lo=10, hi=255),
        _s("CHAT_ATTACHMENT_CACHE_SECONDS", "Кэш вложений в браузере, сек", lo=0, hi=31536000),
        _s("CHAT_MODEL_MAX_SIDE", "Макс. сторона изображения для модели, px", lo=64, hi=8192),
        _s("CHAT_MODEL_PASSTHROUGH_BYTES", "Файлы до этого размера уходят модели как есть, байт", lo=1000, hi=100000000),
        _s("CHAT_MODEL_JPEG_QUALITY", "Качество JPEG для модели", lo=1, hi=100),
    ]),
    ("Инструменты чата", "Результаты анализа для модели и поиск по анализам и пользователям. Числа попадают в промпт модели.", [
        _s("CHAT_JOB_FIELD_CHARS", "Результат анализа: длина поля, симв.", lo=50, hi=20000),
        _s("CHAT_JOB_ANSWER_CHARS", "Результат анализа: длина ответа, симв.", lo=100, hi=100000),
        _s("CHAT_JOB_CATEGORY_CHARS", "Результат анализа: длина категории, симв.", lo=10, hi=500),
        _s("CHAT_JOB_DETAIL_CHARS", "Результат анализа: длина детали, симв.", lo=20, hi=5000),
        _s("CHAT_JOB_MAX_SIGNALS", "Результат анализа: число сигналов", lo=1, hi=100),
        _s("SEARCH_SCAN_LIMIT", "Поиск анализов: строк для разбора", lo=10, hi=100000),
        _s("SEARCH_DEFAULT_LIMIT", "Поиск анализов: записей по умолчанию", lo=1, hi=100),
        _s("SEARCH_MAX_LIMIT", "Поиск анализов: максимум записей", lo=1, hi=500),
        _s("SEARCH_MAX_CARDS", "Поиск анализов: карточек под ответом", lo=0, hi=50),
        _s("SEARCH_DESC_CHARS", "Поиск анализов: длина описания, симв.", lo=20, hi=2000),
        _s("SEARCH_MAX_SINCE_DAYS", "Поиск анализов: глубина, дней", lo=1, hi=36500),
        _s("SEARCH_MAX_QUERY_CHARS", "Поиск анализов: длина запроса, симв.", lo=20, hi=5000),
        _s("SEARCH_MAX_KEYWORDS", "Поиск анализов: ключевых слов", lo=1, hi=50),
        _s("USER_SEARCH_DEFAULT_LIMIT", "Поиск пользователей: записей по умолчанию", lo=1, hi=100),
        _s("USER_SEARCH_MAX_LIMIT", "Поиск пользователей: максимум записей", lo=1, hi=500),
        _s("USER_SEARCH_MAX_SINCE_DAYS", "Поиск пользователей: глубина, дней", lo=1, hi=36500),
    ]),
    ("Загрузка, файлы и миниатюры", "Размеры, кэш и фоновый подсчёт хешей файлов.", [
        _s("MAX_CONTENT_LENGTH", "Максимальный размер запроса, МБ", lo=1, hi=2048, scale=_MB,
           hint="Жёсткий лимит: файлы больше отклоняются (HTTP 413)."),
        _s("CAPTION_MAX_CHARS", "Длина подписи к снимку, симв.", lo=10, hi=20000),
        _s("THUMB_SIZE", "Размер миниатюры, px", lo=16, hi=1024,
           hint="Уже созданные миниатюры остаются прежнего размера."),
        _s("THUMB_QUALITY", "Качество JPEG миниатюр", lo=1, hi=100),
        _s("THUMB_CACHE_SECONDS", "Кэш миниатюр в браузере, сек", lo=0, hi=31536000),
        _s("FILE_HASH_CHUNK_BYTES", "Кусок при чтении файла для SHA-256, байт", lo=4096, hi=104857600),
        _s("DEDUP_IDLE_BATCH", "Хеши: записей за холостой проход", lo=1, hi=100000),
        _s("DEDUP_DEFAULT_BATCH", "Хеши: записей по умолчанию", lo=1, hi=100000),
        _s("DEDUP_CLI_BATCH", "Хеши: записей за проход в CLI", lo=1, hi=100000),
        _s("DEDUP_IDLE_PAUSE", "Хеши: пауза, когда считать нечего, сек", lo=1, hi=86400),
    ]),
    ("Списки и пагинация", "Сколько записей показывать на страницах.", [
        _s("HISTORY_PER_PAGE", "«Моя история»: записей на странице", lo=1, hi=500),
        _s("PANEL_PER_PAGE", "Админ-панель: записей на странице", lo=1, hi=500),
        _s("DASHBOARD_RECENT_LIMIT", "Главная: последних анализов", lo=1, hi=100),
        _s("USER_RECENT_ANALYSES_LIMIT", "Профиль и карточка пользователя: последних анализов", lo=1, hi=200),
        _s("REJECTED_FILES_SHOWN", "Отклонённых файлов в сообщении", lo=1, hi=100),
        _s("BULK_DELETE_MAX_IDS", "Записей за одно массовое удаление", lo=1, hi=100000),
        _s("MODEL_NAME_MAX_LEN", "Длина названия модели в «Статусе сервера»", lo=1, hi=500),
    ]),
    ("Дата и время", "Часовой пояс и форматы дат (синтаксис strftime: %d.%m.%Y %H:%M).", [
        _s("TIMEZONE", "Часовой пояс", "str", hint="Название из базы IANA, например Asia/Aqtobe.", validator="timezone"),
        _s("DATETIME_FORMAT", "Дата и время", "str", validator="strftime"),
        _s("DATETIME_SHORT_FORMAT", "Дата и время без года", "str", validator="strftime"),
        _s("DATE_FORMAT", "Дата", "str", validator="strftime"),
        _s("TIME_FORMAT", "Время", "str", validator="strftime"),
        _s("DATETIME_CHAT_FORMAT", "Дата в списке чатов", "str", validator="strftime"),
    ]),
    ("Безопасность и сессии", "Пароли, сессии, cookie и CSRF.", [
        _s("PASSWORD_MIN_LENGTH", "Минимальная длина пароля", lo=1, hi=128),
        _s("PASSWORD_USERNAME_MIN_LEN", "Логины короче не сравниваются с паролем", lo=1, hi=50),
        _s("PASSWORD_USERNAME_SIMILARITY", "Порог похожести пароля на логин", "float", 0, 1),
        _s("PERMANENT_SESSION_LIFETIME", "Время жизни сессии, дней", "days", 0.01, 3650,
           hint="Действует на сессии, которые будут продлены после изменения."),
        _s("SESSION_COOKIE_HTTPONLY", "Cookie сессии недоступна для JavaScript (HttpOnly)", "bool"),
        _s("SESSION_COOKIE_SAMESITE", "Cookie сессии: SameSite", "choice", choices=("Lax", "Strict", "None"),
           hint="None работает только вместе с «Cookie только по HTTPS»."),
        _s("SESSION_COOKIE_SECURE", "Cookie сессии только по HTTPS", "bool",
           hint="Включайте, только если сайт открывается по HTTPS: иначе вход перестанет работать у всех, включая вас. "
                "Вернуть можно командой `flask reset-settings SESSION_COOKIE_SECURE`."),
        _s("WTF_CSRF_TIME_LIMIT", "Время жизни CSRF-токена, сек", "opt_int", 0, 31536000,
           hint="0 — без ограничения (токен живёт столько же, сколько сессия)."),
    ]),
    ("База данных", "", [
        _s("SQL_IN_CHUNK", "Размер пачки id в DELETE/IN", lo=10, hi=900,
           hint="Лимит числа параметров SQL (в SQLite — 999)."),
    ]),
]

RUNTIME_SETTINGS: list[RuntimeSetting] = [spec for _, _, specs in SETTING_GROUPS for spec in specs]
RUNTIME_SETTINGS_BY_KEY = {s.key: s for s in RUNTIME_SETTINGS}

# Пары «нижняя граница ≤ верхняя» среди самих настроек.
BOUND_PAIRS = [
    (s.min_key, s.max_key) for s in RUNTIME_SETTINGS if s.min_key and s.max_key
]

# Ключи, которые читает сам Flask (а не наш conf()) — переносим их в app.config перед запросом.
FLASK_KEYS = (
    "MAX_CONTENT_LENGTH",
    "PERMANENT_SESSION_LIFETIME",
    "SESSION_COOKIE_HTTPONLY",
    "SESSION_COOKIE_SAMESITE",
    "SESSION_COOKIE_SECURE",
    "WTF_CSRF_TIME_LIMIT",
)


# ---- разбор и форматирование значений -------------------------------------
def _fmt_number(x: float) -> str:
    return "%.10g" % x


def _number(spec: RuntimeSetting, raw: str):
    """Строка из формы -> число в единицах формы (МБ, дни, ...). ValueError — если не число."""
    text = raw.strip().replace(",", ".")
    if spec.kind in ("int", "opt_int"):
        return int(text)
    number = float(text)
    if not math.isfinite(number):
        raise ValueError
    return number


def _value(spec: RuntimeSetting, number):
    """Число в единицах формы -> значение, которое получит конфигурация."""
    if spec.kind == "days":
        return timedelta(days=number)
    if spec.kind == "opt_int":
        return None if number == 0 else number
    if spec.kind == "int":
        return number * spec.scale
    return number


def _kind_hint(spec: RuntimeSetting) -> str:
    return "целое число" if spec.kind in ("int", "opt_int") else "число"


def display_value(spec: RuntimeSetting, value) -> str | bool:
    """Значение конфигурации -> то, что показываем в форме (и что сравниваем с data-default)."""
    if spec.kind == "bool":
        return bool(value)
    if spec.kind == "days":
        return _fmt_number(value.total_seconds() / 86400)
    if spec.kind == "opt_int":
        return "0" if value is None else str(value)
    if spec.kind == "int":
        return str(int(value) // spec.scale) if spec.scale > 1 else str(int(value))
    if spec.kind == "float":
        return _fmt_number(float(value))
    return "" if value is None else str(value)


def _check_string(spec: RuntimeSetting, value: str) -> None:
    label = spec.label
    if spec.kind == "choice":
        if value not in spec.choices:
            raise ValueError(f"«{label}»: допустимо одно из значений — {', '.join(spec.choices)}.")
        return
    if not value:
        raise ValueError(f"«{label}»: значение не может быть пустым.")
    if len(value) > conf("SETTING_STR_MAX_LEN"):
        raise ValueError(f"«{label}»: слишком длинное значение.")
    if spec.validator == "url":
        if not value.lower().startswith(("http://", "https://")) or "://" not in value or not value.split("://", 1)[1]:
            raise ValueError(f"«{label}»: адрес должен начинаться с http:// или https://.")
    elif spec.validator == "timezone":
        try:
            ZoneInfo(value)
        except Exception:  # noqa: BLE001 — ZoneInfoNotFoundError, ValueError, OSError
            raise ValueError(f"«{label}»: неизвестный часовой пояс «{value}».") from None
    elif spec.validator == "strftime":
        try:
            ok = bool(datetime(2000, 1, 2, 3, 4).strftime(value).strip())
        except (ValueError, TypeError):
            ok = False
        if not ok:
            raise ValueError(f"«{label}»: некорректный формат даты.")


# ---- чтение переопределений (с коротким кэшем) -----------------------------
_cache_lock = threading.Lock()
_cache: dict = {"at": -1e9, "data": {}}


def invalidate_cache() -> None:
    with _cache_lock:
        _cache["at"] = -1e9


def _load_overrides() -> dict[str, str]:
    """{ИМЯ: сырая строка} всех действующих переопределений. Отдельное соединение, чтобы не
    трогать транзакцию вызывающего кода (conf() читают откуда угодно, в том числе посреди записи)."""
    now = time.monotonic()
    with _cache_lock:
        if now - _cache["at"] < _CACHE_TTL:
            return _cache["data"]
    try:
        with db.engine.connect() as conn:
            rows = conn.execute(
                db.select(Setting.key, Setting.value).where(Setting.key.like(_CFG_PREFIX + "%"))
            ).all()
        data = {key[len(_CFG_PREFIX):]: value for key, value in rows if value != ""}
    except Exception:  # noqa: BLE001 — таблицы ещё нет (первый запуск) или БД недоступна
        log.debug("settings_store: не удалось прочитать переопределения", exc_info=True)
        with _cache_lock:
            data = _cache["data"]
    with _cache_lock:
        _cache["at"] = now
        _cache["data"] = data
    return data


def _provider(key: str):
    spec = RUNTIME_SETTINGS_BY_KEY.get(key)
    if spec is None:
        return NO_OVERRIDE
    raw = _load_overrides().get(key)
    if raw is None:
        return NO_OVERRIDE
    try:
        if spec.kind == "bool":
            return raw == "1"
        if spec.kind in ("str", "choice"):
            return raw
        return _value(spec, _number(spec, raw))
    except ValueError:
        return NO_OVERRIDE  # мусор в БД не должен ронять приложение


_config.set_override_provider(_provider)


def get_runtime_setting(key: str):
    """Текущее действующее значение (переопределение из БД или значение из config.py/.env)."""
    return conf(key)


def is_runtime_setting_overridden(key: str) -> bool:
    return key in _load_overrides()


def default_value(key: str):
    """Значение «по умолчанию» — без переопределения из БД (то, что даёт config.py/.env)."""
    if key in FLASK_KEYS:
        base = current_app.extensions.get("vision_config_base")
        if base is not None:
            return base[key]
    return current_app.config.get(key, getattr(_config.Config, key))


def effective_bounds(spec: RuntimeSetting, candidates: dict | None = None) -> tuple[float | None, float | None]:
    """(нижняя, верхняя) граница в единицах формы. Если границы сами меняются в этой же форме —
    берём новые значения из candidates (ключ -> значение конфигурации)."""
    def pick(key):
        if candidates and key in candidates:
            return candidates[key]
        return conf(key)

    lo = pick(spec.min_key) if spec.min_key else spec.min
    hi = pick(spec.max_key) if spec.max_key else spec.max
    return lo, hi


# ---- сохранение -----------------------------------------------------------
def runtime_setting_unchanged(key: str, raw_value: str) -> bool:
    """True, если значение из формы совпадает с действующим. Нужно, чтобы сохранение формы не
    создавало переопределение у полей, которые не трогали (иначе «переопределено» стало бы
    всё сразу, а не только изменённое)."""
    spec = RUNTIME_SETTINGS_BY_KEY[key]
    current = conf(key)
    if spec.kind == "bool":
        return (raw_value == "1") == bool(current)
    raw = raw_value.strip()
    try:
        if spec.kind in ("str", "choice"):
            return raw == current
        return _value(spec, _number(spec, raw)) == current
    except ValueError:
        return False  # пусть валидация сообщит об ошибке


def _store(key: str, raw: str) -> None:
    _set(_CFG_PREFIX + key, raw)


def _reset_rows(keys) -> None:
    for key in keys:
        row = db.session.get(Setting, _CFG_PREFIX + key)
        if row is not None:
            db.session.delete(row)


def reset_runtime_setting(key: str) -> None:
    """Убирает переопределение — настройка возвращается к значению из config.py/.env."""
    _reset_rows([key])
    db.session.commit()
    invalidate_cache()


def reset_all_runtime_settings() -> int:
    rows = db.session.scalars(db.select(Setting).where(Setting.key.like(_CFG_PREFIX + "%"))).all()
    for row in rows:
        db.session.delete(row)
    db.session.commit()
    invalidate_cache()
    return len(rows)


def save_runtime_settings(form) -> tuple[list[str], list[str]]:
    """Сохраняет форму /panel/settings/ целиком или не сохраняет ничего.

    Возвращает (ошибки, предупреждения). Если ошибки есть — в БД ничего не изменено.
    """
    errors: list[str] = []
    resets: list[str] = []
    changes: dict[str, str] = {}          # ключ -> строка для БД
    candidates: dict[str, object] = {}    # ключ -> значение конфигурации после сохранения
    numbers: dict[str, object] = {}       # ключ -> число в единицах формы (для проверки диапазона)

    for spec in RUNTIME_SETTINGS:
        key = spec.key
        if form.get(f"reset_{key}"):
            resets.append(key)
            candidates[key] = default_value(key)
            continue
        # Невыбранный чекбокс браузер не отправляет вовсе — это и значит «выключено».
        raw = form.get(key, "")
        if runtime_setting_unchanged(key, raw):
            continue
        try:
            if spec.kind == "bool":
                candidates[key] = raw == "1"
                changes[key] = "1" if raw == "1" else "0"
            elif spec.kind in ("str", "choice"):
                value = raw.strip()
                _check_string(spec, value)
                candidates[key] = value
                changes[key] = value
            else:
                number = _number(spec, raw)
                numbers[key] = number
                candidates[key] = _value(spec, number)
                changes[key] = _fmt_number(number) if spec.kind != "int" and spec.kind != "opt_int" else str(number)
        except ValueError as exc:
            message = str(exc)
            errors.append(message if message.startswith("«") else f"«{spec.label}»: введите {_kind_hint(spec)}.")

    # Диапазоны — после разбора всех полей: границы могли поменяться в этой же форме.
    for key, number in numbers.items():
        spec = RUNTIME_SETTINGS_BY_KEY[key]
        lo, hi = effective_bounds(spec, candidates)
        shown_lo = None if lo is None else (lo / spec.scale if spec.kind == "int" and spec.min_key else lo)
        shown_hi = None if hi is None else (hi / spec.scale if spec.kind == "int" and spec.max_key else hi)
        if shown_lo is not None and number < shown_lo:
            errors.append(f"«{spec.label}»: значение не может быть меньше {_fmt_number(shown_lo)}.")
        if shown_hi is not None and number > shown_hi:
            errors.append(f"«{spec.label}»: значение не может быть больше {_fmt_number(shown_hi)}.")

    for lo_key, hi_key in BOUND_PAIRS:
        if lo_key in candidates or hi_key in candidates:
            lo = candidates.get(lo_key, conf(lo_key))
            hi = candidates.get(hi_key, conf(hi_key))
            if lo > hi:
                errors.append(
                    f"«{RUNTIME_SETTINGS_BY_KEY[lo_key].label}» не может быть больше, чем "
                    f"«{RUNTIME_SETTINGS_BY_KEY[hi_key].label}»."
                )

    if errors:
        return errors, []

    _reset_rows(resets)
    for key, raw in changes.items():
        _store(key, raw)
    db.session.commit()
    invalidate_cache()

    # Если сдвинули границы, а само значение осталось снаружи — скажем об этом, а не промолчим.
    warnings = []
    for spec in RUNTIME_SETTINGS:
        if spec.min_key or spec.max_key:
            lo, hi = effective_bounds(spec)
            current = conf(spec.key)
            if (lo is not None and current < lo) or (hi is not None and current > hi):
                warnings.append(f"«{spec.label}»: текущее значение {display_value(spec, current)} вне новых границ.")
    return [], warnings


# ---- перенос ключей Flask в app.config -------------------------------------
def install(app) -> None:
    """Запоминает исходные значения ключей Flask (после create_app(config=...)) — к ним
    возвращаемся при сбросе переопределения."""
    app.extensions["vision_config_base"] = {key: app.config.get(key) for key in FLASK_KEYS}


def sync_flask_config(app) -> None:
    """Перед каждым запросом переносит переопределения в app.config: эти ключи читает сам Flask
    (лимит запроса, сессия, cookie, CSRF), а не conf(). Дёшево — значения в кэше."""
    base = app.extensions.get("vision_config_base")
    if base is None:
        return
    with app.app_context():
        for key in FLASK_KEYS:
            override = _provider(key)
            app.config[key] = base[key] if override is NO_OVERRIDE else override
