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
from flask_babel import gettext, lazy_gettext

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


# Бэкенд/модель для /chat. Пусто — «как для анализа» (get_analysis_target).
KEY_CHAT_BACKEND = "chat_backend"
KEY_CHAT_MODEL = "chat_model"


def get_chat_target() -> tuple[str, str]:
    """(backend, model) для чата: свой выбор, а если его нет (или провайдера уже нет на
    сервере) — тот же, что у анализа."""
    backend = _get(KEY_CHAT_BACKEND)
    if not backend or is_known_backend(backend) is False:
        return get_analysis_target()
    return backend, _get(KEY_CHAT_MODEL)


def has_chat_target() -> bool:
    """True, если для чата выбран отдельный бэкенд (а не «как для анализа»)."""
    return bool(_get(KEY_CHAT_BACKEND))


def set_chat_target(backend: str, model: str = "") -> None:
    _set(KEY_CHAT_BACKEND, backend)
    _set(KEY_CHAT_MODEL, model)
    db.session.commit()


def clear_chat_target() -> None:
    _set(KEY_CHAT_BACKEND, "")
    _set(KEY_CHAT_MODEL, "")
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
    (lazy_gettext("settings.group.g1.title"), lazy_gettext("settings.group.g1.desc"), [
        _s("VISION_API_BASE_URL", lazy_gettext("settings.vision_api_base_url.label"), "str", hint=lazy_gettext("settings.vision_api_base_url.hint"), validator="url"),
        _s("VISION_API_TIMEOUT", lazy_gettext("settings.vision_api_timeout.label"), min_key="VISION_API_TIMEOUT_MIN", max_key="VISION_API_TIMEOUT_MAX"),
        _s("VISION_API_HEALTH_TIMEOUT", lazy_gettext("settings.vision_api_health_timeout.label"), lo=1, hi=300),
        _s("VISION_API_CALL_TIMEOUT", lazy_gettext("settings.vision_api_call_timeout.label"), lo=1, hi=600,
           hint=lazy_gettext("settings.vision_api_call_timeout.hint")),
        _s("VISION_API_MODELS_TIMEOUT", lazy_gettext("settings.vision_api_models_timeout.label"), lo=1, hi=600,
           hint=lazy_gettext("settings.vision_api_models_timeout.hint")),
        _s("MODELS_CACHE_TTL", lazy_gettext("settings.models_cache_ttl.label"), lo=0, hi=2592000,
           hint=lazy_gettext("settings.models_cache_ttl.hint")),
        _s("VISION_API_EMBED_TIMEOUT", lazy_gettext("settings.vision_api_embed_timeout.label"), lo=1, hi=600),
        _s("VISION_API_ERROR_CHARS", lazy_gettext("settings.vision_api_error_chars.label"), lo=20, hi=5000,
           hint=lazy_gettext("settings.vision_api_error_chars.hint")),
        _s("VISION_API_RAW_TEXT_CHARS", lazy_gettext("settings.vision_api_raw_text_chars.label"), lo=100, hi=100000,
           hint=lazy_gettext("settings.vision_api_raw_text_chars.hint")),
        _s("VISION_API_BACKEND_NAME_CHARS", lazy_gettext("settings.vision_api_backend_name_chars.label"), lo=1, hi=32,
           hint=lazy_gettext("settings.vision_api_backend_name_chars.hint")),
        _s("VISION_API_MODEL_NAME_CHARS", lazy_gettext("settings.vision_api_model_name_chars.label"), lo=1, hi=120,
           hint=lazy_gettext("settings.vision_api_model_name_chars.hint")),
        _s("VISION_API_FALLBACK", lazy_gettext("settings.vision_api_fallback.label"), "bool",
           hint=lazy_gettext("settings.vision_api_fallback.hint")),
        _s("DEFAULT_LANG", lazy_gettext("settings.default_lang.label"), "choice", choices=("ru", "en"),
           hint=lazy_gettext("settings.default_lang.hint")),
    ]),
    (lazy_gettext("settings.group.g2.title"), lazy_gettext("settings.group.g2.desc"), [
        _s("QUEUE_WORKER_ENABLED", lazy_gettext("settings.queue_worker_enabled.label"), "bool",
           hint=lazy_gettext("settings.queue_worker_enabled.hint")),
        _s("QUEUE_WORKERS", lazy_gettext("settings.queue_workers.label"), hint=lazy_gettext("settings.queue_workers.hint"),
           min_key="QUEUE_WORKERS_MIN", max_key="QUEUE_WORKERS_MAX"),
        _s("QUEUE_POLL_SECONDS", lazy_gettext("settings.queue_poll_seconds.label"), "float",
           hint=lazy_gettext("settings.queue_poll_seconds.hint"),
           min_key="QUEUE_POLL_SECONDS_MIN", max_key="QUEUE_POLL_SECONDS_MAX"),
        _s("QUEUE_MAX_FILES_PER_UPLOAD", lazy_gettext("settings.queue_max_files_per_upload.label"),
           min_key="QUEUE_MAX_FILES_PER_UPLOAD_MIN", max_key="QUEUE_MAX_FILES_PER_UPLOAD_MAX"),
        _s("QUEUE_MAX_PENDING_PER_USER", lazy_gettext("settings.queue_max_pending_per_user.label"),
           min_key="QUEUE_MAX_PENDING_PER_USER_MIN", max_key="QUEUE_MAX_PENDING_PER_USER_MAX"),
        _s("QUEUE_MAX_PENDING_TOTAL", lazy_gettext("settings.queue_max_pending_total.label"),
           hint=lazy_gettext("settings.queue_max_pending_total.hint"),
           min_key="QUEUE_MAX_PENDING_TOTAL_MIN", max_key="QUEUE_MAX_PENDING_TOTAL_MAX"),
        _s("QUEUE_ERROR_PAUSE_SECONDS", lazy_gettext("settings.queue_error_pause_seconds.label"), lo=0, hi=300),
        _s("QUEUE_CLAIM_RETRIES", lazy_gettext("settings.queue_claim_retries.label"), lo=1, hi=100),
        _s("QUEUE_STOP_JOIN_TIMEOUT", lazy_gettext("settings.queue_stop_join_timeout.label"), lo=0, hi=300),
    ]),
    (lazy_gettext("settings.group.g3.title"), lazy_gettext("settings.group.g3.desc"), [
        _s("QUEUE_WORKERS_MIN", lazy_gettext("settings.queue_workers_min.label"), lo=1, hi=64),
        _s("QUEUE_WORKERS_MAX", lazy_gettext("settings.queue_workers_max.label"), lo=1, hi=64,
           hint=lazy_gettext("settings.queue_workers_max.hint")),
        _s("QUEUE_POLL_SECONDS_MIN", lazy_gettext("settings.queue_poll_seconds_min.label"), lo=1, hi=3600),
        _s("QUEUE_POLL_SECONDS_MAX", lazy_gettext("settings.queue_poll_seconds_max.label"), lo=1, hi=3600),
        _s("QUEUE_MAX_FILES_PER_UPLOAD_MIN", lazy_gettext("settings.queue_max_files_per_upload_min.label"), lo=1, hi=10000),
        _s("QUEUE_MAX_FILES_PER_UPLOAD_MAX", lazy_gettext("settings.queue_max_files_per_upload_max.label"), lo=1, hi=10000),
        _s("QUEUE_MAX_PENDING_PER_USER_MIN", lazy_gettext("settings.queue_max_pending_per_user_min.label"), lo=1, hi=100000),
        _s("QUEUE_MAX_PENDING_PER_USER_MAX", lazy_gettext("settings.queue_max_pending_per_user_max.label"), lo=1, hi=100000),
        _s("QUEUE_MAX_PENDING_TOTAL_MIN", lazy_gettext("settings.queue_max_pending_total_min.label"), lo=1, hi=100000),
        _s("QUEUE_MAX_PENDING_TOTAL_MAX", lazy_gettext("settings.queue_max_pending_total_max.label"), lo=1, hi=100000),
        _s("VISION_API_TIMEOUT_MIN", lazy_gettext("settings.vision_api_timeout_min.label"), lo=1, hi=86400),
        _s("VISION_API_TIMEOUT_MAX", lazy_gettext("settings.vision_api_timeout_max.label"), lo=1, hi=86400),
        _s("SETTING_STR_MAX_LEN", lazy_gettext("settings.setting_str_max_len.label"), lo=10, hi=2000),
    ]),
    (lazy_gettext("settings.group.g4.title"), lazy_gettext("settings.group.g4.desc"), [
        _s("EMBEDDING_MIN_SIMILARITY", lazy_gettext("settings.embedding_min_similarity.label"), "float", 0, 1,
           hint=lazy_gettext("settings.embedding_min_similarity.hint")),
        _s("EMBEDDING_RELATIVE_MARGIN", lazy_gettext("settings.embedding_relative_margin.label"), "float", 0, 1,
           hint=lazy_gettext("settings.embedding_relative_margin.hint")),
        _s("EMBEDDING_SCAN_LIMIT", lazy_gettext("settings.embedding_scan_limit.label"), lo=1, hi=1000000),
        _s("EMBEDDING_MEMORY_CACHE", lazy_gettext("settings.embedding_memory_cache.label"), "bool",
           hint=lazy_gettext("settings.embedding_memory_cache.hint")),
        _s("EMBEDDING_BATCH_SIZE", lazy_gettext("settings.embedding_batch_size.label"), lo=1, hi=64, hint=lazy_gettext("settings.embedding_batch_size.hint")),
        _s("EMBEDDING_MAX_TEXT_CHARS", lazy_gettext("settings.embedding_max_text_chars.label"), lo=100, hi=8000,
           hint=lazy_gettext("settings.embedding_max_text_chars.hint")),
        _s("EMBEDDING_MAX_MATCHED_IDS", lazy_gettext("settings.embedding_max_matched_ids.label"), lo=1, hi=100000),
        _s("EMBEDDING_IDLE_BATCHES", lazy_gettext("settings.embedding_idle_batches.label"), lo=1, hi=1000),
        _s("EMBEDDING_IDLE_PAUSE_UNAVAILABLE", lazy_gettext("settings.embedding_idle_pause_unavailable.label"), lo=1, hi=86400),
        _s("EMBEDDING_IDLE_PAUSE_NOTHING", lazy_gettext("settings.embedding_idle_pause_nothing.label"), lo=1, hi=86400),
        _s("EMBEDDING_IDLE_PAUSE_RETRY", lazy_gettext("settings.embedding_idle_pause_retry.label"), lo=1, hi=86400),
    ]),
    (lazy_gettext("settings.group.g5.title"), lazy_gettext("settings.group.g5.desc"), [
        _s("CHAT_MAX_MESSAGE_LEN", lazy_gettext("settings.chat_max_message_len.label"), lo=10, hi=200000),
        _s("CHAT_MAX_HISTORY_MESSAGES", lazy_gettext("settings.chat_max_history_messages.label"), lo=1, hi=1000),
        _s("CHAT_TITLE_LEN", lazy_gettext("settings.chat_title_len.label"), lo=5, hi=200),
        _s("CHAT_TURN_STALE_SECONDS", lazy_gettext("settings.chat_turn_stale_seconds.label"), lo=30, hi=86400,
           hint=lazy_gettext("settings.chat_turn_stale_seconds.hint")),
        _s("CHAT_MAX_TOOL_CALLS", lazy_gettext("settings.chat_max_tool_calls.label"), lo=1, hi=20),
        _s("CHAT_PROMPT_MAX_CHARS", lazy_gettext("settings.chat_prompt_max_chars.label"), lo=200, hi=50000,
           hint=lazy_gettext("settings.chat_prompt_max_chars.hint")),
        _s("CHAT_CONTEXT_IMAGES", lazy_gettext("settings.chat_context_images.label"), lo=0, hi=50,
           hint=lazy_gettext("settings.chat_context_images.hint")),
        _s("CHAT_MAX_IMAGES_PER_MESSAGE", lazy_gettext("settings.chat_max_images_per_message.label"), lo=1, hi=50),
        _s("CHAT_MAX_IMAGE_BYTES", lazy_gettext("settings.chat_max_image_bytes.label"), lo=1, hi=500, scale=_MB,
           hint=lazy_gettext("settings.chat_max_image_bytes.hint")),
        _s("CHAT_ATTACHMENT_NAME_MAX", lazy_gettext("settings.chat_attachment_name_max.label"), lo=10, hi=255),
        _s("CHAT_ATTACHMENT_CACHE_SECONDS", lazy_gettext("settings.chat_attachment_cache_seconds.label"), lo=0, hi=31536000),
        _s("CHAT_MODEL_MAX_SIDE", lazy_gettext("settings.chat_model_max_side.label"), lo=64, hi=8192),
        _s("CHAT_MODEL_PASSTHROUGH_BYTES", lazy_gettext("settings.chat_model_passthrough_bytes.label"), lo=1000, hi=100000000),
        _s("CHAT_MODEL_JPEG_QUALITY", lazy_gettext("settings.chat_model_jpeg_quality.label"), lo=1, hi=100),
    ]),
    (lazy_gettext("settings.group.g6.title"), lazy_gettext("settings.group.g6.desc"), [
        _s("CHAT_JOB_FIELD_CHARS", lazy_gettext("settings.chat_job_field_chars.label"), lo=50, hi=20000),
        _s("CHAT_JOB_ANSWER_CHARS", lazy_gettext("settings.chat_job_answer_chars.label"), lo=100, hi=100000),
        _s("CHAT_JOB_CATEGORY_CHARS", lazy_gettext("settings.chat_job_category_chars.label"), lo=10, hi=500),
        _s("CHAT_JOB_DETAIL_CHARS", lazy_gettext("settings.chat_job_detail_chars.label"), lo=20, hi=5000),
        _s("CHAT_JOB_MAX_SIGNALS", lazy_gettext("settings.chat_job_max_signals.label"), lo=1, hi=100),
        _s("SEARCH_SCAN_LIMIT", lazy_gettext("settings.search_scan_limit.label"), lo=10, hi=100000),
        _s("SEARCH_DEFAULT_LIMIT", lazy_gettext("settings.search_default_limit.label"), lo=1, hi=100),
        _s("SEARCH_MAX_LIMIT", lazy_gettext("settings.search_max_limit.label"), lo=1, hi=500),
        _s("SEARCH_MAX_CARDS", lazy_gettext("settings.search_max_cards.label"), lo=0, hi=50),
        _s("SEARCH_DESC_CHARS", lazy_gettext("settings.search_desc_chars.label"), lo=20, hi=2000),
        _s("SEARCH_MAX_SINCE_DAYS", lazy_gettext("settings.search_max_since_days.label"), lo=1, hi=36500),
        _s("SEARCH_MAX_QUERY_CHARS", lazy_gettext("settings.search_max_query_chars.label"), lo=20, hi=5000),
        _s("SEARCH_MAX_KEYWORDS", lazy_gettext("settings.search_max_keywords.label"), lo=1, hi=50),
        _s("BODY_MARKS_CHECK_DEFAULT_LIMIT", lazy_gettext("settings.body_marks_check_default_limit.label"), lo=1, hi=50),
        _s("BODY_MARKS_CHECK_MAX_LIMIT", lazy_gettext("settings.body_marks_check_max_limit.label"), lo=1, hi=100),
        _s("BODY_MARKS_CHECK_TIME_BUDGET", lazy_gettext("settings.body_marks_check_time_budget.label"), lo=10, hi=1800),
        _s("USER_SEARCH_DEFAULT_LIMIT", lazy_gettext("settings.user_search_default_limit.label"), lo=1, hi=100),
        _s("USER_SEARCH_MAX_LIMIT", lazy_gettext("settings.user_search_max_limit.label"), lo=1, hi=500),
        _s("USER_SEARCH_MAX_CARDS", lazy_gettext("settings.user_search_max_cards.label"), lo=0, hi=50),
        _s("CHAT_ACTION_TTL_MINUTES", lazy_gettext("settings.chat_action_ttl_minutes.label"), lo=1, hi=1440),
        _s("CHAT_ACTION_MAX_PENDING", lazy_gettext("settings.chat_action_max_pending.label"), lo=1, hi=50),
        _s("USER_SEARCH_MAX_SINCE_DAYS", lazy_gettext("settings.user_search_max_since_days.label"), lo=1, hi=36500),
    ]),
    (lazy_gettext("settings.group.g7.title"), lazy_gettext("settings.group.g7.desc"), [
        _s("MAX_CONTENT_LENGTH", lazy_gettext("settings.max_content_length.label"), lo=1, hi=2048, scale=_MB,
           hint=lazy_gettext("settings.max_content_length.hint")),
        _s("CAPTION_MAX_CHARS", lazy_gettext("settings.caption_max_chars.label"), lo=10, hi=20000),
        _s("THUMB_SIZE", lazy_gettext("settings.thumb_size.label"), lo=16, hi=1024,
           hint=lazy_gettext("settings.thumb_size.hint")),
        _s("THUMB_QUALITY", lazy_gettext("settings.thumb_quality.label"), lo=1, hi=100),
        _s("THUMB_CACHE_SECONDS", lazy_gettext("settings.thumb_cache_seconds.label"), lo=0, hi=31536000),
        _s("FILE_HASH_CHUNK_BYTES", lazy_gettext("settings.file_hash_chunk_bytes.label"), lo=4096, hi=104857600),
        _s("DEDUP_IDLE_BATCH", lazy_gettext("settings.dedup_idle_batch.label"), lo=1, hi=100000),
        _s("DEDUP_DEFAULT_BATCH", lazy_gettext("settings.dedup_default_batch.label"), lo=1, hi=100000),
        _s("DEDUP_CLI_BATCH", lazy_gettext("settings.dedup_cli_batch.label"), lo=1, hi=100000),
        _s("DEDUP_IDLE_PAUSE", lazy_gettext("settings.dedup_idle_pause.label"), lo=1, hi=86400),
    ]),
    (lazy_gettext("settings.group.g8.title"), lazy_gettext("settings.group.g8.desc"), [
        _s("HISTORY_PER_PAGE", lazy_gettext("settings.history_per_page.label"), lo=1, hi=500),
        _s("PANEL_PER_PAGE", lazy_gettext("settings.panel_per_page.label"), lo=1, hi=500),
        _s("DASHBOARD_RECENT_LIMIT", lazy_gettext("settings.dashboard_recent_limit.label"), lo=1, hi=100),
        _s("USER_RECENT_ANALYSES_LIMIT", lazy_gettext("settings.user_recent_analyses_limit.label"), lo=1, hi=200),
        _s("REJECTED_FILES_SHOWN", lazy_gettext("settings.rejected_files_shown.label"), lo=1, hi=100),
        _s("BULK_DELETE_MAX_IDS", lazy_gettext("settings.bulk_delete_max_ids.label"), lo=1, hi=100000),
        _s("MODEL_NAME_MAX_LEN", lazy_gettext("settings.model_name_max_len.label"), lo=1, hi=500),
    ]),
    (lazy_gettext("settings.group.g9.title"), lazy_gettext("settings.group.g9.desc"), [
        _s("TIMEZONE", lazy_gettext("settings.timezone.label"), "str", hint=lazy_gettext("settings.timezone.hint"), validator="timezone"),
        _s("DATETIME_FORMAT", lazy_gettext("settings.datetime_format.label"), "str", validator="strftime"),
        _s("DATETIME_SHORT_FORMAT", lazy_gettext("settings.datetime_short_format.label"), "str", validator="strftime"),
        _s("DATE_FORMAT", lazy_gettext("settings.date_format.label"), "str", validator="strftime"),
        _s("TIME_FORMAT", lazy_gettext("settings.time_format.label"), "str", validator="strftime"),
        _s("DATETIME_CHAT_FORMAT", lazy_gettext("settings.datetime_chat_format.label"), "str", validator="strftime"),
    ]),
    (lazy_gettext("settings.group.g10.title"), lazy_gettext("settings.group.g10.desc"), [
        _s("PASSWORD_MIN_LENGTH", lazy_gettext("settings.password_min_length.label"), lo=1, hi=128),
        _s("PASSWORD_USERNAME_MIN_LEN", lazy_gettext("settings.password_username_min_len.label"), lo=1, hi=50),
        _s("PASSWORD_USERNAME_SIMILARITY", lazy_gettext("settings.password_username_similarity.label"), "float", 0, 1),
        _s("PERMANENT_SESSION_LIFETIME", lazy_gettext("settings.permanent_session_lifetime.label"), "days", 0.01, 3650,
           hint=lazy_gettext("settings.permanent_session_lifetime.hint")),
        _s("SESSION_COOKIE_HTTPONLY", lazy_gettext("settings.session_cookie_httponly.label"), "bool"),
        _s("SESSION_COOKIE_SAMESITE", lazy_gettext("settings.session_cookie_samesite.label"), "choice", choices=("Lax", "Strict", "None"),
           hint=lazy_gettext("settings.session_cookie_samesite.hint")),
        _s("SESSION_COOKIE_SECURE", lazy_gettext("settings.session_cookie_secure.label"), "bool",
           hint=lazy_gettext("settings.session_cookie_secure.hint")),
        _s("WTF_CSRF_TIME_LIMIT", lazy_gettext("settings.wtf_csrf_time_limit.label"), "opt_int", 0, 31536000,
           hint=lazy_gettext("settings.wtf_csrf_time_limit.hint")),
    ]),
    (lazy_gettext("settings.group.g11.title"), "", [
        _s("SQL_IN_CHUNK", lazy_gettext("settings.sql_in_chunk.label"), lo=10, hi=900,
           hint=lazy_gettext("settings.sql_in_chunk.hint")),
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
    return gettext("settings.err.kind_int") if spec.kind in ("int", "opt_int") else gettext("settings.err.kind_number")


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


class SettingError(ValueError):
    """Ошибка проверки с уже готовым текстом для пользователя (с названием настройки)."""


def _check_string(spec: RuntimeSetting, value: str) -> None:
    label = spec.label
    if spec.kind == "choice":
        if value not in spec.choices:
            raise SettingError(gettext("settings.err.choice", label=label, choices=", ".join(spec.choices)))
        return
    if not value:
        raise SettingError(gettext("settings.err.empty", label=label))
    if len(value) > conf("SETTING_STR_MAX_LEN"):
        raise SettingError(gettext("settings.err.too_long", label=label))
    if spec.validator == "url":
        if not value.lower().startswith(("http://", "https://")) or "://" not in value or not value.split("://", 1)[1]:
            raise SettingError(gettext("settings.err.url", label=label))
    elif spec.validator == "timezone":
        try:
            ZoneInfo(value)
        except Exception:  # noqa: BLE001 — ZoneInfoNotFoundError, ValueError, OSError
            raise SettingError(gettext("settings.err.timezone", label=label, value=value)) from None
    elif spec.validator == "strftime":
        try:
            ok = bool(datetime(2000, 1, 2, 3, 4).strftime(value).strip())
        except (ValueError, TypeError):
            ok = False
        if not ok:
            raise SettingError(gettext("settings.err.date_format", label=label))


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
            errors.append(message if isinstance(exc, SettingError) else gettext("settings.err.enter", label=spec.label, kind=_kind_hint(spec)))

    # Диапазоны — после разбора всех полей: границы могли поменяться в этой же форме.
    for key, number in numbers.items():
        spec = RUNTIME_SETTINGS_BY_KEY[key]
        lo, hi = effective_bounds(spec, candidates)
        shown_lo = None if lo is None else (lo / spec.scale if spec.kind == "int" and spec.min_key else lo)
        shown_hi = None if hi is None else (hi / spec.scale if spec.kind == "int" and spec.max_key else hi)
        if shown_lo is not None and number < shown_lo:
            errors.append(gettext("settings.err.min", label=spec.label, value=_fmt_number(shown_lo)))
        if shown_hi is not None and number > shown_hi:
            errors.append(gettext("settings.err.max", label=spec.label, value=_fmt_number(shown_hi)))

    for lo_key, hi_key in BOUND_PAIRS:
        if lo_key in candidates or hi_key in candidates:
            lo = candidates.get(lo_key, conf(lo_key))
            hi = candidates.get(hi_key, conf(hi_key))
            if lo > hi:
                errors.append(gettext(
                    "settings.err.bounds",
                    lo=RUNTIME_SETTINGS_BY_KEY[lo_key].label,
                    hi=RUNTIME_SETTINGS_BY_KEY[hi_key].label,
                ))

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
                warnings.append(gettext("settings.warn.out_of_bounds", label=spec.label, value=display_value(spec, current)))
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
