"""
Тонкий клиент для vision_analyzer_server.py (см. /analyze, /health, /lang).

Сервер ожидает сырые байты изображения в теле POST /analyze с
Content-Type: image/<тип>, и возвращает JSON вида:

    {
      "count": 1,
      "requested_backend": "vllm",
      "results": [
        {"file": "...", "backend": "vllm", "report": {...}}
      ]
    }

где report — словарь с полями risk_level, needs_human_review,
description, signals, rationale, recommendation, либо {"_raw": "..."}
если модель не вернула валидный JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import requests
from django.conf import settings


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
    return getattr(settings, "VISION_API_BASE_URL", "http://127.0.0.1:6769").rstrip("/")


def _timeout() -> int:
    return getattr(settings, "VISION_API_TIMEOUT", 120)


def check_health() -> dict:
    """Опрашивает /health. Возвращает словарь статуса или бросает VisionApiError."""
    try:
        resp = requests.get(f"{_base_url()}/health", timeout=10)
        data = resp.json()
        data["_http_status"] = resp.status_code
        return data
    except requests.exceptions.ConnectionError as exc:
        raise VisionApiError("Не удалось подключиться к серверу анализа изображений.") from exc
    except requests.exceptions.Timeout as exc:
        raise VisionApiError("Сервер анализа изображений не отвечает (таймаут).") from exc
    except Exception as exc:  # noqa: BLE001
        raise VisionApiError(f"Не удалось получить статус сервера: {exc}") from exc


def analyze_image(image_bytes: bytes, mime_type: str, lang: str = "ru") -> AnalysisOutcome:
    """Отправляет одно изображение на /analyze и возвращает разобранный результат."""
    url = f"{_base_url()}/analyze"
    headers = {"Content-Type": mime_type or "image/jpeg"}
    params = {"lang": lang} if lang else {}

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
            message = payload.get("error", resp.text)
        except ValueError:
            message = resp.text
        raise VisionApiError(f"Сервер вернул ошибку ({resp.status_code}): {message}")

    try:
        payload = resp.json()
    except ValueError as exc:
        raise VisionApiError("Сервер вернул некорректный JSON-ответ.") from exc

    results = payload.get("results") or []
    if not results:
        raise VisionApiError("Сервер не вернул ни одного результата анализа.")

    first = results[0]
    report = first.get("report") or {}
    backend = first.get("backend", "")

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
        description=report.get("description", ""),
        raw_report=report,
        is_raw_fallback=False,
    )
