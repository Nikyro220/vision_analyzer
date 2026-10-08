"""
Тонкий клиент для vision_analyzer_server.py (см. /analyze, /chat, /health, /lang).

Сервер ожидает сырые байты изображения в теле POST /analyze с
Content-Type: image/<тип> и возвращает JSON вида:

    {
      "count": 1,
      "requested_backend": "vllm",
      "results": [
        {"file": "...", "backend": "vllm", "report": {...}}
      ]
    }

где report — словарь с полями risk_level, needs_human_review, description,
signals, rationale, recommendation, text_on_image, context, либо {"_raw": "..."},
если модель не вернула валидный JSON.
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import requests
from flask import has_app_context

from . import credentials
from .config import conf
from .extensions import db
from .models import ProviderModelsCache, utcnow

log = logging.getLogger("vision_app.services")


# Параметры генерации, которыми управляет POST /sampling.
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "seed", "num_ctx", "num_predict", "think")

# Список бэкендов здесь НЕ зашит: его отдаёт сервер анализа (GET /providers, см. get_providers()),
# поэтому новый провайдер, добавленный в inference/providers/, сразу виден и настраивается в панели.

class VisionApiError(Exception):
    """Любая ошибка при обращении к API анализа изображений."""


class EmbeddingNotReady(VisionApiError):
    """Модель эмбеддингов на сервере ещё скачивается/загружается (HTTP 503 с
    state=downloading|loading). Не поломка: запрос стоит повторить позже."""

    def __init__(self, message: str, retry_after: int | None = None):
        super().__init__(message)
        self.retry_after = retry_after


@dataclass
class AnalysisOutcome:
    backend: str = ""
    risk_level: str = "unknown"
    needs_human_review: bool = False
    description: str = ""
    raw_report: dict = field(default_factory=dict)
    is_raw_fallback: bool = False


def _base_url() -> str:
    from .settings_store import get_runtime_setting  # локальный импорт — settings_store импортирует services

    return get_runtime_setting("VISION_API_BASE_URL").rstrip("/")


def _timeout() -> int:
    from .settings_store import get_runtime_setting

    return get_runtime_setting("VISION_API_TIMEOUT")


def _auth_headers(backend: str = "") -> dict[str, str]:
    """Заголовки с API-ключами для сервера анализа (он ключи НЕ хранит — см. credentials.py).

    backend задан — только ключ этого провайдера (/analyze, /chat, /models с явным выбором).
    Пусто — ключи всех провайдеров (/health, /providers и запросы «бэкенд по умолчанию»: какой
    именно выберет сервер, заранее неизвестно). Без контекста приложения (или если БД
    недоступна) заголовков нет: запрос уйдёт без ключа и сервер вернёт понятную ошибку.
    """
    if not has_app_context():
        return {}
    try:
        if backend:
            key = credentials.get_key(backend)
            return {credentials.header_name(backend): key} if key else {}
        return {credentials.header_name(name): key for name, key in credentials.get_all_keys().items()}
    except Exception:  # noqa: BLE001 — не роняем запрос из-за хранилища ключей
        log.warning("Не удалось прочитать API-ключи из БД", exc_info=True)
        return {}


def _fallback_enabled() -> bool:
    return bool(conf("VISION_API_FALLBACK"))


def _request_auth_headers(backend: str) -> dict[str, str]:
    """Ключи для /analyze и /chat. Если разрешён фолбэк — ключи ВСЕХ провайдеров: сервер сам
    решает, на какого уйти при сбое, и без их ключей облачные в цепочку не попадут."""
    return _auth_headers("" if _fallback_enabled() else backend)


_health_lock = threading.Lock()
_health_state: dict = {"data": None, "at": 0.0}


def check_health(force: bool = False) -> dict:
    """Статус сервера анализа (GET /health) или VisionApiError.

    Ответ кэшируется на VISION_API_HEALTH_CACHE_TTL секунд, чтобы страница статуса и сводка
    не дёргали сервер (а он — бэкенды) на каждое открытие. force=True — свежий опрос
    (кнопка «Обновить статус», проверка после смены ключа). Ошибки не кэшируются."""
    ttl = conf("VISION_API_HEALTH_CACHE_TTL")
    now = time.monotonic()
    with _health_lock:
        st = _health_state
        if not force and ttl > 0 and st["data"] is not None and now - st["at"] < ttl:
            return dict(st["data"])
        data = _fetch_health()
        st.update(data=data, at=now)
        return dict(data)


def invalidate_health() -> None:
    with _health_lock:
        _health_state.update(data=None, at=0.0)


def _fetch_health() -> dict:
    try:
        resp = requests.get(
            f"{_base_url()}/health", headers=_auth_headers(), timeout=conf("VISION_API_HEALTH_TIMEOUT"),
        )
        data = resp.json()
        if not isinstance(data, dict):
            raise ValueError("ожидался JSON-объект")
        data["_http_status"] = resp.status_code
        return data
    except requests.exceptions.ConnectionError as exc:
        raise VisionApiError("Не удалось подключиться к серверу анализа изображений.") from exc
    except requests.exceptions.Timeout as exc:
        raise VisionApiError("Сервер анализа изображений не отвечает (таймаут).") from exc
    except Exception as exc:  # noqa: BLE001
        raise VisionApiError(f"Не удалось получить статус сервера: {exc}") from exc


def analyze_image(
    image_bytes: bytes,
    mime_type: str,
    lang: str | None = None,  # None -> DEFAULT_LANG из настроек
    backend: str = "",
    model: str = "",
    caption: str = "",
    categories: list[dict] | None = None,
) -> AnalysisOutcome:
    """Отправляет одно изображение на /analyze и возвращает разобранный результат.

    backend/model — необязательные; если пусты, сервер выбирает бэкенд по умолчанию
    и автоматически определяет модель. caption — необязательный контекст к конкретному
    изображению (передаётся серверу как есть, влияет только на промпт модели).

    categories — разовые категории оценивания на этот вызов (см.
    inference/categories.py: build_overlay). Единственный источник этого списка —
    таблица categories в БД vision_app (categories_store.build_categories_payload);
    сервер анализа своих категорий больше не хранит.

    Формат запроса: без категорий — как раньше, сырые байты в теле
    (Content-Type: image/*) плюс query-параметры (быстрее, без base64).
    С категориями — целиком JSON-телом (Content-Type: application/json,
    см. inference/analyze.py: _parse_json_body), потому что правила
    категорий (full/compact) в сумме легко превышают ограничение aiohttp
    на длину строки запроса (~8 КБ) — через query-параметры категории
    туда просто не влезают уже на 4-5 включённых категориях.
    """
    url = f"{_base_url()}/analyze"
    lang = lang or conf("DEFAULT_LANG")

    if categories:
        payload: dict = {
            "image": base64.b64encode(image_bytes).decode("ascii"),
            "lang": lang,
            "categories": categories,  # уже list[dict] — сервер примет как есть, без доп. json.dumps
        }
        if backend:
            payload["backend"] = backend
        if model:
            payload["model"] = model
        if caption:
            payload["caption"] = caption
        payload["fallback"] = _fallback_enabled()
        request_kwargs = {"json": payload, "headers": _request_auth_headers(backend)}
    else:
        headers = {"Content-Type": mime_type or "image/jpeg", **_request_auth_headers(backend)}
        params = {"fallback": int(_fallback_enabled())}
        if lang:
            params["lang"] = lang
        if backend:
            params["backend"] = backend
        if model:
            params["model"] = model
        if caption:
            params["caption"] = caption
        request_kwargs = {"data": image_bytes, "headers": headers, "params": params}

    try:
        resp = requests.post(url, timeout=_timeout(), **request_kwargs)
    except requests.exceptions.ConnectionError as exc:
        raise VisionApiError(
            "Не удалось подключиться к серверу анализа изображений. "
            "Проверьте, что vision_analyzer_server.py запущен."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise VisionApiError("Сервер анализа изображений не ответил вовремя (таймаут).") from exc

    if resp.status_code >= 400:
        try:
            payload = resp.json()
            message = payload.get("error", resp.text) if isinstance(payload, dict) else resp.text
        except ValueError:
            message = resp.text
        raise VisionApiError(f"Сервер вернул ошибку ({resp.status_code}): {message}")

    try:
        payload = resp.json()
    except ValueError as exc:
        raise VisionApiError("Сервер вернул некорректный JSON-ответ.") from exc

    results = payload.get("results") if isinstance(payload, dict) else None
    if not results:
        raise VisionApiError("Сервер не вернул ни одного результата анализа.")

    first = results[0] if isinstance(results[0], dict) else {}
    return _outcome_from_result(first)


def _outcome_from_result(item: dict) -> AnalysisOutcome:
    """Один элемент results[] ответа /analyze -> AnalysisOutcome."""
    report = item.get("report") or {}
    if not isinstance(report, dict):
        report = {}
    backend = str(item.get("backend", ""))[: conf("VISION_API_BACKEND_NAME_CHARS")]

    if "_raw" in report:
        return AnalysisOutcome(
            backend=backend,
            risk_level="unknown",
            needs_human_review=True,
            description=str(report.get("_raw", ""))[: conf("VISION_API_RAW_TEXT_CHARS")],
            raw_report=report,
            is_raw_fallback=True,
        )

    risk_level = report.get("risk_level", "unknown")
    if risk_level not in ("low", "medium", "high"):
        risk_level = "unknown"

    return AnalysisOutcome(
        backend=backend,
        risk_level=risk_level,
        needs_human_review=bool(report.get("needs_human_review", False)),
        description=str(report.get("description", "") or ""),
        raw_report=report,
        is_raw_fallback=False,
    )


@dataclass
class LinkItem:
    """Одна картинка из поста + результат её анализа."""

    name: str  # ссылка (для поста с несколькими картинками — со скрепой «#N»)
    image_bytes: bytes
    mime: str
    outcome: AnalysisOutcome
    caption: str = ""  # контекст поста (платформа, автор, текст...), который ушёл модели


@dataclass
class LinkOutcome:
    items: list[LinkItem] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)  # ссылки/картинки, которые не удалось разобрать


def analyze_link(
    link: str,
    lang: str | None = None,
    backend: str = "",
    model: str = "",
    categories: list[dict] | None = None,
) -> LinkOutcome:
    """Анализ по ссылке на пост: сервер сам скачивает картинки (yt-dlp / Open Graph) и контекст поста.

    Запрос идёт с return_images=1 — сервер дополнительно возвращает исходную картинку
    (source.image_b64), чтобы приложение сохранило её для превью и страницы результата.
    Пост с несколькими картинками даёт несколько LinkItem. Бросает VisionApiError, если не
    получилось ничего (ошибка ссылки, сервер недоступен и т. п.).
    """
    url = f"{_base_url()}/analyze"
    payload: dict = {"url": link, "lang": lang or conf("DEFAULT_LANG")}
    if categories:
        payload["categories"] = categories
    if backend:
        payload["backend"] = backend
    if model:
        payload["model"] = model

    try:
        resp = requests.post(
            url, json=payload, params={"return_images": "1"}, headers=_auth_headers(backend), timeout=_timeout(),
        )
    except requests.exceptions.ConnectionError as exc:
        raise VisionApiError(
            "Не удалось подключиться к серверу анализа изображений. "
            "Проверьте, что vision_analyzer_server.py запущен."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise VisionApiError("Сервер анализа изображений не ответил вовремя (таймаут).") from exc

    try:
        data = resp.json()
    except ValueError:
        data = None

    if resp.status_code >= 400:
        message = data.get("error", resp.text) if isinstance(data, dict) else resp.text
        raise VisionApiError(f"Сервер вернул ошибку ({resp.status_code}): {message}")
    if not isinstance(data, dict):
        raise VisionApiError("Сервер вернул некорректный JSON-ответ.")

    outcome = LinkOutcome(
        errors=[str(e.get("error", "")) for e in data.get("link_errors") or [] if isinstance(e, dict)],
    )
    for entry in data.get("results") or []:
        if not isinstance(entry, dict):
            continue
        source = entry.get("source") if isinstance(entry.get("source"), dict) else {}
        raw_image = source.get("image_b64")
        try:
            image_bytes = base64.b64decode(raw_image, validate=True) if raw_image else b""
        except (ValueError, TypeError):
            image_bytes = b""
        if not image_bytes:
            outcome.errors.append(
                "Сервер анализа не вернул картинку из ссылки (устаревшая версия сервера без return_images?)."
            )
            continue
        outcome.items.append(
            LinkItem(
                name=str(entry.get("file") or link),
                image_bytes=image_bytes,
                mime=str(source.get("image_mime") or "image/jpeg"),
                outcome=_outcome_from_result(entry),
                caption=str(source.get("caption") or ""),
            )
        )

    if not outcome.items:
        raise VisionApiError(outcome.errors[0] if outcome.errors else "Сервер не вернул ни одного результата анализа.")
    return outcome


# ----------------------------------------------------------------------------
# Эмбеддинги текста (POST/GET /embeddings на сервере анализа)
# ----------------------------------------------------------------------------
def get_embedding_status() -> dict:
    """GET /embeddings -> {"state": "ready"|"downloading"|"loading"|"error"|..., "model": ...}.

    Сервер отвечает 503 и в штатных состояниях (пока модель качается), поэтому тело
    разбирается независимо от HTTP-статуса. Бросает VisionApiError, если сервер
    недоступен или не знает такой ручки (старая версия)."""
    try:
        resp = requests.get(f"{_base_url()}/embeddings", timeout=conf("VISION_API_HEALTH_TIMEOUT"))
    except requests.exceptions.RequestException as exc:
        raise VisionApiError(f"Не удалось получить статус эмбеддингов: {exc}") from exc
    try:
        data = resp.json()
    except ValueError:
        data = None
    if not isinstance(data, dict):
        raise VisionApiError("Сервер анализа не отдаёт статус эмбеддингов (устаревшая версия?).")
    return data


def embed_texts(texts: list[str], timeout: int | None = None) -> tuple[list[list[float]], str]:
    """Тексты -> (векторы в порядке texts, имя модели, которой они посчитаны).

    texts не должны содержать пустых строк и быть длиннее 64 штук: сервер молча
    выкидывает пустые и усекает батч, из-за чего порядок «текст -> вектор» поплыл бы.
    Бросает EmbeddingNotReady, пока модель качается/грузится (с retry_after из
    заголовка Retry-After), и VisionApiError при любой другой ошибке."""
    if not texts:
        return [], ""
    if timeout is None:
        timeout = conf("VISION_API_EMBED_TIMEOUT")
    try:
        resp = requests.post(f"{_base_url()}/embeddings", json={"texts": texts}, timeout=timeout)
    except requests.exceptions.ConnectionError as exc:
        raise VisionApiError("Не удалось подключиться к серверу анализа изображений.") from exc
    except requests.exceptions.Timeout as exc:
        raise VisionApiError("Сервер анализа изображений не отвечает (таймаут).") from exc
    except requests.exceptions.RequestException as exc:
        raise VisionApiError(f"Ошибка запроса к серверу анализа: {exc}") from exc

    try:
        data = resp.json()
    except ValueError:
        data = None

    if resp.status_code == 503 and isinstance(data, dict) and data.get("state") in ("downloading", "loading"):
        try:
            retry_after = int(resp.headers.get("Retry-After", ""))
        except ValueError:
            retry_after = None
        raise EmbeddingNotReady(
            "Модель эмбеддингов ещё скачивается или загружается.", retry_after=retry_after
        )
    if resp.status_code >= 400:
        message = data.get("error") if isinstance(data, dict) and data.get("error") else resp.text
        raise VisionApiError(f"Сервер вернул ошибку ({resp.status_code}): {str(message)[: conf('VISION_API_ERROR_CHARS')]}")

    vectors = data.get("embeddings") if isinstance(data, dict) else None
    model = data.get("model") if isinstance(data, dict) else None
    if not isinstance(vectors, list) or len(vectors) != len(texts) or not model:
        raise VisionApiError("Сервер вернул некорректный ответ /embeddings.")
    return vectors, str(model)


# ----------------------------------------------------------------------------
# Провайдеры (GET /providers на сервере анализа)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class CredentialField:
    """Секрет, который провайдеру нужен в каждом запросе (API-ключ). Сервер анализа его не хранит:
    значение лежит в БД приложения в зашифрованном виде (credentials.py) и уходит серверу
    заголовком ``header`` в каждом запросе (_auth_headers)."""

    label: str
    hint: str = ""
    header: str = ""


@dataclass(frozen=True)
class ProviderInfo:
    name: str
    label: str = ""
    fallback: str | None = None
    configured: bool = True
    endpoint: str = ""
    sampling_keys: tuple[str, ...] = ()
    credential: CredentialField | None = None

    @property
    def title(self) -> str:
        return self.label or self.name


_PROVIDERS_TTL = 15.0       # секунд: как долго список считается свежим
_PROVIDERS_RETRY = 5.0      # секунд: пауза между попытками, пока сервер не отвечает
_providers_lock = threading.Lock()
_providers_state: dict = {"data": None, "fetched_at": 0.0, "failed_at": 0.0, "error": ""}


def _parse_provider(item: dict) -> ProviderInfo:
    cred = item.get("credential")
    credential = None
    if isinstance(cred, dict):
        credential = CredentialField(
            label=str(cred.get("label") or "API-ключ"),
            hint=str(cred.get("hint") or ""),
            header=str(cred.get("header") or ""),
        )
    return ProviderInfo(
        name=str(item["name"]),
        label=str(item.get("label") or ""),
        fallback=item.get("fallback") or None,
        configured=bool(item.get("configured", True)),
        endpoint=str(item.get("endpoint") or ""),
        sampling_keys=tuple(k for k in (item.get("sampling_keys") or []) if k in SAMPLING_KEYS),
        credential=credential,
    )


def get_providers(force: bool = False) -> list[ProviderInfo]:
    """Все провайдеры, которые знает сервер анализа (в порядке регистрации), — включая
    не настроенные (например, Gemini без ключа). Результат кэшируется на несколько секунд,
    чтобы не ходить на сервер на каждый запрос; пока сервер не отвечает, отдаётся прежний
    список (если был), иначе бросается VisionApiError."""
    now = time.monotonic()
    with _providers_lock:
        st = _providers_state
        if not force and st["data"] is not None and now - st["fetched_at"] < _PROVIDERS_TTL:
            return list(st["data"])
        if not force and st["failed_at"] and now - st["failed_at"] < _PROVIDERS_RETRY:
            if st["data"] is not None:
                return list(st["data"])
            raise VisionApiError(st["error"])

        try:
            payload = _call("get", "/providers", keys=True)
            items = payload.get("providers") if isinstance(payload, dict) else None
            if not isinstance(items, list):
                raise VisionApiError("Сервер анализа не отдаёт список провайдеров (устаревшая версия?).")
            data = [_parse_provider(i) for i in items if isinstance(i, dict) and i.get("name")]
        except VisionApiError as exc:
            st["failed_at"], st["error"] = now, str(exc)
            if st["data"] is not None:
                return list(st["data"])
            raise

        st.update(data=data, fetched_at=now, failed_at=0.0, error="")
        return list(data)


def invalidate_providers() -> None:
    """Сбрасывает кэш списка провайдеров (после смены их настроек)."""
    with _providers_lock:
        _providers_state.update(fetched_at=0.0, failed_at=0.0)


def get_provider(name: str) -> ProviderInfo | None:
    """Провайдер по имени или None, если сервер такого не знает. Бросает VisionApiError,
    если список получить не удалось."""
    return next((p for p in get_providers() if p.name == name), None)


def is_known_backend(name: str) -> bool | None:
    """Знает ли сервер такой бэкенд. None — определить не удалось (сервер не отвечает):
    в таком случае сохранённый выбор лучше не выбрасывать, а оставить как есть."""
    try:
        return get_provider(name) is not None
    except VisionApiError:
        return None


# ----------------------------------------------------------------------------
# Модели и параметры генерации
# ----------------------------------------------------------------------------
def _call(method: str, path: str, *, keys: bool | str = False, **kwargs):
    """GET/POST к серверу анализа с единообразной обработкой ошибок. Возвращает JSON.

    keys — прикладывать ли API-ключи (заголовки X-Api-Key-*): True — все провайдеры,
    строка — только ключ этого провайдера, False — без ключей (по умолчанию).
    """
    url = f"{_base_url()}{path}"
    if keys:
        kwargs["headers"] = {**_auth_headers("" if keys is True else keys), **kwargs.get("headers", {})}
    try:
        resp = getattr(requests, method)(url, timeout=kwargs.pop("timeout", conf("VISION_API_CALL_TIMEOUT")), **kwargs)
    except requests.exceptions.ConnectionError as exc:
        raise VisionApiError("Не удалось подключиться к серверу анализа изображений.") from exc
    except requests.exceptions.Timeout as exc:
        raise VisionApiError("Сервер анализа изображений не отвечает (таймаут).") from exc
    except requests.exceptions.RequestException as exc:
        raise VisionApiError(f"Ошибка запроса к серверу анализа: {exc}") from exc

    try:
        data = resp.json()
    except ValueError:
        data = None

    if resp.status_code >= 400:
        message = data.get("error") if isinstance(data, dict) and data.get("error") else resp.text
        raise VisionApiError(f"Сервер вернул ошибку ({resp.status_code}): {str(message)[: conf('VISION_API_ERROR_CHARS')]}")
    if data is None:
        raise VisionApiError("Сервер вернул некорректный JSON-ответ.")
    return data


def _dict_name(item: dict) -> str:
    for key in ("id", "name", "model"):
        value = item.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def extract_model_names(payload, backend: str = "") -> list[str]:
    """Достаёт имена моделей из ответа /models.

    Точный формат ответа зависит от сервера, поэтому разбор терпимый. Понимает, например:
      ["a", "b"]                                    {"models": ["a", "b"]}
      {"data": [{"id": "a"}]}  (OpenAI/vLLM)        {"models": [{"name": "a"}]}  (Ollama)
      {"vllm": {"models": [...]}, "ollama": {...}}  {"models": {"a": {...}}}
    """

    def walk(obj, depth=0) -> list[str]:
        if depth > 5:
            return []
        if isinstance(obj, str):
            return [obj]
        if isinstance(obj, list):
            names: list[str] = []
            for item in obj:
                if isinstance(item, str):
                    names.append(item)
                elif isinstance(item, dict):
                    name = _dict_name(item)
                    names.extend([name] if name else walk(item, depth + 1))
            return names
        if isinstance(obj, dict):
            if backend and backend in obj:
                return walk(obj[backend], depth + 1)
            for key in ("models", "data", "available", "items"):
                if key in obj:
                    value = obj[key]
                    found = walk(value, depth + 1)
                    if not found and key == "models" and isinstance(value, dict):
                        found = [k for k in value if isinstance(k, str)]
                    return found
        return []

    seen: set[str] = set()
    result: list[str] = []
    for name in walk(payload):
        name = name.strip()
        if name and name not in seen:
            seen.add(name)
            result.append(name)
    return result


@dataclass
class ChatOutcome:
    reply: str = ""
    backend: str = ""
    model: str = ""


def chat_with_model(
    message: str,
    history: list[dict] | None = None,
    images: list[str] | None = None,
    backend: str = "",
    model: str = "",
    lang: str | None = None,  # None -> DEFAULT_LANG из настроек
    system: str = "",
    system_mode: str = "",
) -> ChatOutcome:
    """Отправляет одно сообщение на POST /chat (см. inference/chat.py) и
    возвращает разобранный ответ.

    Сервер не хранит историю — при каждом вызове ей нужно передавать всю
    историю целиком (``history``: список {"role": "user"|"assistant", "content": "..."}).
    ``images`` — необязательный список data-url строк (data:image/...;base64,...)
    для мультимодального сообщения.

    ``system_mode`` — как сервер использует ``system``: "append" (по умолчанию на сервере) —
    добавляет к своему промпту-персоне, "replace" — НЕ добавляет свой, модель получает только
    ``system`` (так делает чат панели: роль задаётся в БД, см. chat_prompt.py).
    """
    url = f"{_base_url()}/chat"
    lang = lang or conf("DEFAULT_LANG")

    payload: dict = {"message": message, "lang": lang}
    if history:
        payload["history"] = history
    if images:
        payload["images"] = images
    if backend:
        payload["backend"] = backend
    if model:
        payload["model"] = model
    if system:
        payload["system"] = system
    if system_mode:
        payload["system_mode"] = system_mode
    payload["fallback"] = _fallback_enabled()

    try:
        resp = requests.post(url, json=payload, headers=_request_auth_headers(backend), timeout=_timeout())
    except requests.exceptions.ConnectionError as exc:
        raise VisionApiError(
            "Не удалось подключиться к серверу анализа изображений. "
            "Проверьте, что vision_analyzer_server.py запущен."
        ) from exc
    except requests.exceptions.Timeout as exc:
        raise VisionApiError("Сервер не ответил вовремя (таймаут).") from exc

    if resp.status_code >= 400:
        try:
            data = resp.json()
            message_err = data.get("error", resp.text) if isinstance(data, dict) else resp.text
        except ValueError:
            message_err = resp.text
        raise VisionApiError(f"Сервер вернул ошибку ({resp.status_code}): {message_err}")

    try:
        data = resp.json()
    except ValueError as exc:
        raise VisionApiError("Сервер вернул некорректный JSON-ответ.") from exc

    if not isinstance(data, dict) or "reply" not in data:
        raise VisionApiError("Сервер не вернул ответ модели.")

    return ChatOutcome(
        reply=str(data.get("reply", "")),
        backend=str(data.get("backend", ""))[: conf("VISION_API_BACKEND_NAME_CHARS")],
        model=str(data.get("model", ""))[: conf("VISION_API_MODEL_NAME_CHARS")],
    )


def _fetch_models(backend: str) -> list[str]:
    """Живой запрос GET /models?backend=... (с ключом провайдера, если он сохранён)."""
    # Облачный провайдер (есть сохранённый ключ) может отвечать медленно — у него свой, более
    # длинный таймаут, чтобы не раздувать общий VISION_API_CALL_TIMEOUT для /sampling и т. п.
    cloud = bool(_auth_headers(backend))
    timeout = conf("VISION_API_MODELS_TIMEOUT") if cloud else conf("VISION_API_CALL_TIMEOUT")
    payload = _call("get", "/models", keys=backend, params={"backend": backend}, timeout=timeout)
    # Ошибка бэкенда может прийти как {"error": "..."} или {"vllm": {"error": "..."}}.
    for holder in (payload, payload.get(backend) if isinstance(payload, dict) else None):
        if isinstance(holder, dict) and isinstance(holder.get("error"), str) and holder["error"]:
            raise VisionApiError(holder["error"])
    return extract_model_names(payload, backend)


def _aware(dt: datetime | None) -> datetime | None:
    if dt is not None and dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)  # SQLite отдаёт naive-время (в БД всё в UTC)
    return dt


def _models_cache_row(backend: str) -> ProviderModelsCache | None:
    """Строка кэша для ТЕКУЩЕГО ключа провайдера. Нет ключа — кэша нет: кэшируются только
    облачные провайдеры (у локальных Ollama/vLLM список быстрый и меняется при `ollama pull`)."""
    key = credentials.get_key(backend) if has_app_context() else ""
    if not key:
        return None
    row = db.session.get(ProviderModelsCache, backend)
    if row is None or row.fingerprint != credentials.fingerprint(key):
        return None  # кэш другого ключа
    return row


def models_cached_at(backend: str) -> datetime | None:
    """Когда список моделей облачного провайдера был получен (None — кэша нет)."""
    row = _models_cache_row(backend)
    return _aware(row.fetched_at) if row else None


def invalidate_models_cache(backend: str) -> None:
    row = db.session.get(ProviderModelsCache, backend)
    if row is not None:
        db.session.delete(row)
        db.session.commit()


def get_models(backend: str, *, force: bool = False, cached_only: bool = False) -> list[str]:
    """Список моделей бэкенда.

    Для облачного провайдера (с сохранённым ключом) список кэшируется в БД на
    MODELS_CACHE_TTL секунд: запрос к облачному провайдеру медленный, а модели меняются редко.
      force       — игнорировать кэш и обновить (кнопка «Обновить список»);
      cached_only — не ходить в сеть: вернуть кэш (даже устаревший) или [] (отрисовка страницы
                    статуса не должна ждать провайдера).
    Если свежий запрос не удался, а старый кэш есть — отдаётся он (кроме force: там ошибку нужно
    показать). У локальных провайдеров кэша нет — всегда живой запрос.
    """
    row = _models_cache_row(backend)
    if row is not None and not force:
        age = utcnow() - _aware(row.fetched_at)
        if cached_only or age < timedelta(seconds=conf("MODELS_CACHE_TTL")):
            return list(row.models or [])
    elif cached_only and has_app_context() and credentials.get_key(backend):
        return []  # ключ есть, кэша ещё нет

    try:
        models = _fetch_models(backend)
    except VisionApiError:
        if row is not None and not force:
            log.warning("Список моделей %r не обновился — отдаю устаревший кэш", backend, exc_info=True)
            return list(row.models or [])
        raise

    key = credentials.get_key(backend) if has_app_context() else ""
    if key and models:  # пустой ответ не кэшируем — это скорее сбой, чем «моделей нет»
        fp = credentials.fingerprint(key)
        existing = db.session.get(ProviderModelsCache, backend)
        if existing is None:
            db.session.add(ProviderModelsCache(provider=backend, fingerprint=fp, models=models))
        else:
            existing.fingerprint, existing.models, existing.fetched_at = fp, models, utcnow()
        db.session.commit()
    return models


def _pick_sampling(data) -> dict:
    if not isinstance(data, dict):
        return {}
    for wrapper in ("sampling", "current", "values", "params"):
        if isinstance(data.get(wrapper), dict):
            data = data[wrapper]
            break
    return {key: data[key] for key in SAMPLING_KEYS if key in data}


_sampling_cache: dict = {"data": None, "at": 0.0}


def get_sampling() -> dict:
    """Текущие параметры генерации сервера (GET /sampling). Общие для всех бэкендов.
    Кэшируется на VISION_API_HEALTH_CACHE_TTL секунд; set_sampling обновляет кэш сам."""
    ttl = conf("VISION_API_HEALTH_CACHE_TTL")
    now = time.monotonic()
    c = _sampling_cache
    if ttl > 0 and c["data"] is not None and now - c["at"] < ttl:
        return dict(c["data"])
    data = _pick_sampling(_call("get", "/sampling"))
    c.update(data=data, at=now)
    return dict(data)


def set_sampling(values: dict) -> dict:
    """Меняет параметры (POST /sampling, JSON). Передаются только переданные ключи.
    Действует сразу на все последующие запросы /analyze — для всего сервера."""
    payload = {k: v for k, v in values.items() if k in SAMPLING_KEYS}
    if not payload:
        raise VisionApiError("Нет параметров для сохранения.")
    data = _pick_sampling(_call("post", "/sampling", json=payload))
    _sampling_cache.update(data=data, at=time.monotonic())
    return data

def set_default_backend(name: str) -> dict:
    """Меняет бэкенд по умолчанию на сервере анализа (POST /config, поле backend).

    Это общее значение для всех клиентов сервера: оно используется там, где бэкенд явно не
    указан (в том числе панелью, пока админ не выбрал бэкенд вручную на карточке), и от него
    считается автоматический фолбэк. Хранится в памяти сервера — при его перезапуске
    возвращается значение из VISION_ANALYZER_BACKEND."""
    return _call("post", "/config", json={"backend": name})


# ----------------------------------------------------------------------------
# Категории оценивания (GET /categories, /categories/<имя> на сервере анализа)
# ----------------------------------------------------------------------------
def fetch_server_categories() -> list[dict]:
    """Категории, которые сервер анализа держит по умолчанию, — в каноническом порядке.

    GET /categories отдаёт порядок ({"order": [...]}), затем по одному запросу на категорию
    GET /categories/<имя> — {name, summary, full, compact, examples?}. Возвращаются сырые
    ответы сервера как есть (full/compact — с обёрткой <signal_category>); разбор под модель
    БД — в categories_store. Бросает VisionApiError, если сервер недоступен или ответил
    неожиданным форматом; пустой список — сервер не знает ни одной категории."""
    listing = _call("get", "/categories")
    order = listing.get("order") if isinstance(listing, dict) else listing
    if not isinstance(order, list) or not all(isinstance(n, str) for n in order):
        raise VisionApiError("Сервер вернул некорректный список категорий.")

    items: list[dict] = []
    for name in order:
        record = _call("get", f"/categories/{quote(name, safe='')}")
        if not isinstance(record, dict) or not isinstance(record.get("full"), str):
            raise VisionApiError(f"Сервер вернул некорректное описание категории «{name}».")
        items.append({**record, "name": name})
    return items
