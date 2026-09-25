"""
Тонкий клиент для vision_analyzer_server.py (см. /analyze, /health, /lang).

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

from dataclasses import dataclass, field

import requests
from flask import current_app


# Бэкенды, которые понимает сервер (параметр ?backend=).
BACKENDS = ("vllm", "ollama")

# Параметры генерации, которыми управляет POST /sampling.
SAMPLING_KEYS = ("temperature", "top_p", "top_k", "seed", "num_ctx", "num_predict", "think")

class VisionApiError(Exception):
    """Любая ошибка при обращении к API анализа изображений."""


@dataclass
class AnalysisOutcome:
    backend: str = ""
    risk_level: str = "unknown"
    needs_human_review: bool = False
    description: str = ""
    raw_report: dict = field(default_factory=dict)
    is_raw_fallback: bool = False


def _base_url() -> str:
    from .settings_store import get_runtime_setting  # локальный импорт — settings_store импортирует BACKENDS отсюда

    return (get_runtime_setting("VISION_API_BASE_URL") or "http://127.0.0.1:6769").rstrip("/")


def _timeout() -> int:
    from .settings_store import get_runtime_setting

    return get_runtime_setting("VISION_API_TIMEOUT") or 120


def check_health() -> dict:
    """Опрашивает /health. Возвращает словарь статуса или бросает VisionApiError."""
    try:
        resp = requests.get(f"{_base_url()}/health", timeout=10)
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
    lang: str = "ru",
    backend: str = "",
    model: str = "",
    caption: str = "",
) -> AnalysisOutcome:
    """Отправляет одно изображение на /analyze и возвращает разобранный результат.

    backend/model — необязательные; если пусты, сервер выбирает бэкенд по умолчанию
    и автоматически определяет модель. caption — необязательный контекст к конкретному
    изображению (передаётся серверу как есть, влияет только на промпт модели).
    """
    url = f"{_base_url()}/analyze"
    headers = {"Content-Type": mime_type or "image/jpeg"}
    params = {"lang": lang} if lang else {}
    if backend:
        params["backend"] = backend
    if model:
        params["model"] = model
    if caption:
        params["caption"] = caption

    try:
        resp = requests.post(
            url, data=image_bytes, headers=headers, params=params, timeout=_timeout()
        )
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
    report = first.get("report") or {}
    if not isinstance(report, dict):
        report = {}
    backend = str(first.get("backend", ""))[:32]

    if "_raw" in report:
        return AnalysisOutcome(
            backend=backend,
            risk_level="unknown",
            needs_human_review=True,
            description=str(report.get("_raw", ""))[:2000],
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


# ----------------------------------------------------------------------------
# Модели и параметры генерации
# ----------------------------------------------------------------------------
def _call(method: str, path: str, **kwargs):
    """GET/POST к серверу анализа с единообразной обработкой ошибок. Возвращает JSON."""
    url = f"{_base_url()}{path}"
    try:
        resp = getattr(requests, method)(url, timeout=kwargs.pop("timeout", 15), **kwargs)
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
        raise VisionApiError(f"Сервер вернул ошибку ({resp.status_code}): {str(message)[:300]}")
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


def get_models(backend: str) -> list[str]:
    """Список моделей, которые сейчас сообщает бэкенд (GET /models?backend=...)."""
    payload = _call("get", "/models", params={"backend": backend})
    # Ошибка бэкенда может прийти как {"error": "..."} или {"vllm": {"error": "..."}}.
    for holder in (payload, payload.get(backend) if isinstance(payload, dict) else None):
        if isinstance(holder, dict) and isinstance(holder.get("error"), str) and holder["error"]:
            raise VisionApiError(holder["error"])
    return extract_model_names(payload, backend)


def _pick_sampling(data) -> dict:
    if not isinstance(data, dict):
        return {}
    for wrapper in ("sampling", "current", "values", "params"):
        if isinstance(data.get(wrapper), dict):
            data = data[wrapper]
            break
    return {key: data[key] for key in SAMPLING_KEYS if key in data}


def get_sampling() -> dict:
    """Текущие параметры генерации сервера (GET /sampling). Общие для всех бэкендов."""
    return _pick_sampling(_call("get", "/sampling"))


def set_sampling(values: dict) -> dict:
    """Меняет параметры (POST /sampling, JSON). Передаются только переданные ключи.
    Действует сразу на все последующие запросы /analyze — для всего сервера."""
    payload = {k: v for k, v in values.items() if k in SAMPLING_KEYS}
    if not payload:
        raise VisionApiError("Нет параметров для сохранения.")
    return _pick_sampling(_call("post", "/sampling", json=payload))