"""
embeddings.py — всё, что относится к POST /embeddings: превращает текст в
вектор для векторного поиска на стороне vision_app (см. config.py:
"Эмбеддинги текста" — там же обоснование, почему это отдельная от
BACKEND/OLLAMA_HOST/VLLM_URL модель, зафиксированная конфигом, а не
переключаемая на лету).

В отличие от backends.py/chat_backends.py, здесь НЕТ похода по сети к
внешнему бэкенду (Ollama/vLLM) — модель эмбеддингов (fastembed, ONNX) живёт
прямо в этом процессе, на CPU. Поэтому:
  - нет aiohttp.ClientSession, нет config.REQUEST_TIMEOUT/DISCOVERY_TIMEOUT;
  - зато сам вызов модели — блокирующий и CPU-bound, поэтому его нельзя
    звать напрямую из async-хендлера (заблокирует event loop и все
    остальные запросы, включая /health). Он всегда уходит в
    loop.run_in_executor (см. embed_texts_async).
  - модель — не потокобезопасный per-call объект, а один загруженный на
    процесс инстанс; конкурентные вызовы из executor-пула сериализуются
    через _model_lock, чтобы не полагаться на потокобезопасность
    конкретной сборки onnxruntime.

Модель грузится ЛЕНИВО, при первом реальном запросе — не при импорте
модуля и не при старте сервера, чтобы python server.py не подвисал на
загрузку весов, если они вдруг не прогреты заранее (прогрев на этапе
сборки — см. warm_embeddings.py в этой же папке). Если веса недоступны
(например, нет сети до HuggingFace Hub), первый запрос один раз честно
пытается их скачать, ловит ошибку и КЭШИРУЕТ её (_model_load_error) —
следующие запросы сразу получают 503, не повторяя провальную попытку
на каждый вызов.
GET /embeddings (handle_embeddings_info) поэтому НЕ грузит модель — он
показывает то, что можно узнать офлайн, без скачивания и без инференса.
"""

from __future__ import annotations

import asyncio
import logging
import threading

from aiohttp import web

import config
from analyze import _json
from config import locales

log = logging.getLogger("vision_analyzer.embeddings")

try:
    from fastembed import TextEmbedding
except ImportError:
    TextEmbedding = None  # см. config.py: предупреждение о недостающем пакете


class EmbeddingUnavailableError(Exception):
    """Модель эмбеддингов недоступна: пакет не установлен, отключена конфигом,
    либо не удалось загрузить веса (например, недоступен HuggingFace Hub)."""


# ---------------------------------------------------------------------------
# Офлайн-метаданные модели (без скачивания весов, см. embeddings.py docstring)
# ---------------------------------------------------------------------------

_model_info_cache: dict | None = None


def _lookup_model_metadata(model_name: str) -> dict | None:
    """Статическое описание модели из реестра fastembed (dim, license, размер
    на диске) — не требует сети, это данные, зашитые в саму библиотеку."""
    if TextEmbedding is None:
        return None
    for entry in TextEmbedding.list_supported_models():
        if entry.get("model") == model_name:
            return entry
    return None


def known_dim() -> int | None:
    """Размерность вектора EMBEDDING_MODEL, если модель есть в реестре
    fastembed. None — модель неизвестна реестру (например, опечатка в
    конфиге) ИЛИ fastembed не установлен; в обоих случаях реальную
    размерность узнать не выйдет до первого успешного вызова модели."""
    global _model_info_cache
    if _model_info_cache is None:
        _model_info_cache = _lookup_model_metadata(config.EMBEDDING_MODEL) or {}
    return _model_info_cache.get("dim")


# ---------------------------------------------------------------------------
# Ленивая загрузка модели + синхронный инференс
# ---------------------------------------------------------------------------

_model = None
_model_lock = threading.Lock()
_model_load_error: str | None = None  # запоминаем последнюю ошибку загрузки, чтобы не долбить HF Hub на каждый запрос


def _get_model():
    """Возвращает загруженный TextEmbedding, загружая его при первом вызове.

    Вызывать только из фонового потока (executor) — конструктор скачивает/
    читает с диска веса модели и блокирует поток на секунды-минуты при
    первом запуске.
    """
    global _model, _model_load_error

    if TextEmbedding is None:
        raise EmbeddingUnavailableError("пакет fastembed не установлен на сервере")

    with _model_lock:
        if _model is not None:
            return _model
        if _model_load_error is not None:
            # Не повторяем заведомо провальную загрузку (например, HF Hub
            # недоступен) на каждый запрос — это лишние секунды ожидания и
            # шум в логах. Сбрасывается только перезапуском процесса.
            raise EmbeddingUnavailableError(_model_load_error)

        log.info("embeddings: загружаю модель %s (первый запрос, может занять время)...", config.EMBEDDING_MODEL)
        try:
            _model = TextEmbedding(model_name=config.EMBEDDING_MODEL)
        except Exception as exc:  # noqa: BLE001 — любая ошибка загрузки (сеть, диск, битые веса)
            _model_load_error = str(exc)
            log.exception("embeddings: не удалось загрузить модель %s", config.EMBEDDING_MODEL)
            raise EmbeddingUnavailableError(_model_load_error) from exc

        log.info("embeddings: модель %s загружена", config.EMBEDDING_MODEL)
        return _model


def _embed_sync(texts: list[str]) -> list[list[float]]:
    """Блокирующий вызов модели. С _model_lock — сериализует конкурентные
    вызовы из разных потоков executor-пула (см. embeddings.py docstring)."""
    model = _get_model()
    with _model_lock:
        # model.embed возвращает генератор numpy.ndarray, в порядке входа
        return [vec.tolist() for vec in model.embed(texts)]


async def embed_texts_async(texts: list[str]) -> tuple[list[list[float]], str, int]:
    """Async-обёртка для хендлера: считает CPU-bound инференс в executor'е,
    не блокируя event loop сервера (остальные запросы — /health, /analyze,
    /chat — продолжают обрабатываться, пока считается эмбеддинг).

    Возвращает (векторы в порядке texts, имя модели, размерность).
    Бросает EmbeddingUnavailableError, если модель недоступна.
    """
    loop = asyncio.get_running_loop()
    vectors = await loop.run_in_executor(None, _embed_sync, texts)
    dim = len(vectors[0]) if vectors else (known_dim() or 0)
    return vectors, config.EMBEDDING_MODEL, dim


def status() -> dict:
    """Статус для /health и GET /embeddings — не грузит модель и не трогает
    сеть, только то, что известно офлайн + загружена ли модель прямо сейчас."""
    return {
        "enabled": config.EMBEDDING_ENABLED,
        "fastembed_installed": TextEmbedding is not None,
        "model": config.EMBEDDING_MODEL,
        "dim": known_dim(),
        "loaded": _model is not None,
        "load_error": _model_load_error,
    }


def is_healthy() -> bool:
    """True, если сервис эмбеддингов в рабочем состоянии: включён конфигом,
    пакет установлен, и последняя попытка загрузки (если была) не провалилась.
    Не гарантирует, что модель реально загрузится — только что нет уже
    известной причины ей не загрузиться."""
    s = status()
    return bool(s["enabled"] and s["fastembed_installed"] and s["load_error"] is None)


# ---------------------------------------------------------------------------
# HTTP-хендлеры (POST/GET /embeddings)
# ---------------------------------------------------------------------------

_MAX_BATCH = 64  # разумный потолок на один запрос — защита от случайного гигантского батча


def _query_lang(request: web.Request) -> str | None:
    lang = request.query.get("lang")
    return lang.strip().lower() if lang else None


async def handle_embeddings_info(request: web.Request) -> web.Response:
    """GET /embeddings — офлайн-статус сервиса: включён ли, установлен ли
    fastembed, какая модель/размерность сконфигурирована, загружена ли уже
    в память. Не запускает загрузку модели и не делает инференс."""
    info = status()
    return _json(info, status=200 if is_healthy() else 503)


async def handle_embeddings(request: web.Request) -> web.Response:
    """POST /embeddings — текст(ы) -> вектор(ы).

    Тело — только JSON (в отличие от /chat и /analyze, тут нет ни картинок,
    ни multipart-файлов, отправлять есть смысл только текстом):
      {"text": "..."}              — один текст, ответ содержит "embedding"
      {"texts": ["...", "..."]}    — несколько сразу (для бэкфилла), ответ —
                                      только "embeddings" (без "embedding")
    lang можно передать query-параметром (?lang=en) или в теле — влияет
    только на язык текста ошибки, не на сам эмбеддинг.
    """
    lang = _query_lang(request)

    if request.content_type != "application/json":
        return _json(
            {"error": config._t("error.embeddings_unsupported_content_type", lang=lang)}, status=400
        )

    try:
        body = await request.json() or {}
    except Exception:
        return _json({"error": config._t("error.invalid_json_body", lang=lang)}, status=400)

    lang = lang or ((body.get("lang") or "").strip().lower() or None)
    if lang and locales is not None and lang not in locales.LOCALES:
        return _json(
            {"error": config._t("error.lang_not_loaded", requested_lang=lang, lang=lang),
             "available": list(locales.LOCALES.keys())},
            status=400,
        )

    single = "texts" not in body
    raw_texts = body.get("texts") if not single else ([body.get("text")] if body.get("text") is not None else [])

    if not isinstance(raw_texts, list):
        return _json({"error": config._t("error.empty_text", lang=lang)}, status=400)

    texts = [str(t).strip() if t is not None else "" for t in raw_texts]
    texts = [t for t in texts if t]  # пустые строки в батче просто выкидываем, а не роняем весь запрос

    if not texts:
        return _json({"error": config._t("error.empty_text", lang=lang)}, status=400)
    if len(texts) > _MAX_BATCH:
        texts = texts[:_MAX_BATCH]  # мягкое усечение — лучше частичный ответ, чем 400 на бэкфилле
    too_long = [i for i, t in enumerate(texts) if len(t) > config.EMBEDDING_MAX_CHARS]
    if too_long:
        return _json(
            {"error": config._t("error.text_too_long", max_chars=config.EMBEDDING_MAX_CHARS, lang=lang)},
            status=400,
        )

    if not config.EMBEDDING_ENABLED:
        return _json({"error": config._t("error.embeddings_disabled", lang=lang)}, status=503)

    try:
        vectors, model_name, dim = await embed_texts_async(texts)
    except EmbeddingUnavailableError as exc:
        return _json(
            {"error": config._t("error.embeddings_unavailable", detail=str(exc), lang=lang)}, status=503
        )
    except Exception:
        log.exception("embeddings: сбой инференса")
        return _json({"error": config._t("error.embeddings_failed", lang=lang)}, status=500)

    payload = {"model": model_name, "dim": dim, "count": len(vectors), "embeddings": vectors}
    if single:
        payload["embedding"] = vectors[0]
    return _json(payload)
