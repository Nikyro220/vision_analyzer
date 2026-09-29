"""
embeddings.py — всё, что относится к POST /embeddings: превращает текст в
вектор для векторного поиска на стороне vision_app (см. config.py:
"Эмбеддинги текста" — там же обоснование, почему это отдельная от
BACKEND/OLLAMA_HOST/VLLM_URL модель, зафиксированная конфигом, а не
переключаемая на лету).

В отличие от backends.py/chat_backends.py, здесь НЕТ похода по сети к
внешнему бэкенду (Ollama/vLLM) — модель эмбеддингов (fastembed, ONNX) живёт
прямо в этом процессе, на CPU.

Скачивание и загрузка модели
----------------------------
Веса лежат в ЯВНОЙ отдельной папке config.EMBEDDING_CACHE_DIR
(по умолчанию inference/models/embeddings, переопределяется
VISION_ANALYZER_EMBEDDING_DIR) — fastembed получает её как cache_dir.

При первом запуске сервера (server.py: on_startup ->
start_background_load) в отдельном daemon-потоке стартует скачивание и
загрузка модели. Сервер при этом отвечает сразу:
  - /health, /analyze, /chat работают как обычно;
  - GET /embeddings показывает state (downloading / loading / ready /
    error) и примерный прогресс скачивания;
  - POST /embeddings пока модель не готова возвращает 503 + Retry-After.
На следующих запусках веса уже в папке: поток только читает их с диска
(state=loading) и модель становится ready за секунды.

Свой поток, а не loop.run_in_executor(None, ...): дефолтный пул executor'а
делят инференс эмбеддингов и другие блокирующие вызовы сервера, а
скачивание на сотни МБ может занять минуты — оно не должно занимать в нём
слот. Сам инференс (CPU-bound) по-прежнему уходит в executor
(embed_texts_async), чтобы не блокировать event loop; конкурентные вызовы
сериализуются через _infer_lock, чтобы не полагаться на потокобезопасность
конкретной сборки onnxruntime.

Инференс сам модель НЕ грузит: если она не готова — сразу
EmbeddingNotReadyError (503), а не зависание запроса на скачивании.
Ошибка скачивания (например, нет сети до HuggingFace Hub) сохраняется в
state=error и не повторяется сама; повторить можно через
POST /embeddings/download или перезапуском сервера. Недокачанные файлы
huggingface_hub оставляет как *.incomplete и докачивает при повторе.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from pathlib import Path

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
    либо не удалось скачать/загрузить веса (например, недоступен HuggingFace Hub)."""


class EmbeddingNotReadyError(EmbeddingUnavailableError):
    """Ничего не сломано, модель просто ещё скачивается/загружается в
    фоновом потоке — клиенту стоит повторить запрос позже (HTTP 503 +
    Retry-After), а не считать сервис вышедшим из строя."""

    def __init__(self, state: str):
        super().__init__(f"model is {state}")
        self.state = state


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
# Скачивание/загрузка модели в фоновом потоке + синхронный инференс
# ---------------------------------------------------------------------------

# idle -> downloading (весов нет на диске) | loading (веса уже есть) -> ready | error
_state = "idle"
_model = None
_load_error: str | None = None
_loader_thread: threading.Thread | None = None
_state_lock = threading.Lock()   # защищает _state/_model/_load_error/_loader_thread
_infer_lock = threading.Lock()   # сериализует model.embed() между потоками executor'а

_RETRY_AFTER_SEC = 15


def _model_files_present() -> bool:
    """Есть ли веса модели в config.EMBEDDING_CACHE_DIR (по раскладке кэша
    huggingface_hub: models--<org>--<repo>/snapshots/<hash>/<model_file>).
    huggingface_hub кладёт файл в snapshots/ только после полного скачивания
    (до этого он лежит как blobs/*.incomplete), поэтому это надёжный признак."""
    cache_dir = Path(config.EMBEDDING_CACHE_DIR)
    if not cache_dir.is_dir():
        return False
    meta = _lookup_model_metadata(config.EMBEDDING_MODEL) or {}
    hf_repo = (meta.get("sources") or {}).get("hf")
    model_file = meta.get("model_file")
    if hf_repo and model_file:
        snapshots = cache_dir / ("models--" + hf_repo.replace("/", "--")) / "snapshots"
        return snapshots.is_dir() and any((snap / model_file).exists() for snap in snapshots.iterdir())
    return any(cache_dir.rglob("*.onnx"))  # модель без HF-источника — грубая проверка


def _dir_size_bytes() -> int:
    """Сколько уже лежит в папке модели, включая недокачанные *.incomplete —
    по ней считается примерный прогресс скачивания."""
    cache_dir = Path(config.EMBEDDING_CACHE_DIR)
    if not cache_dir.is_dir():
        return 0
    total = 0
    for f in cache_dir.rglob("*"):
        try:
            if f.is_file() and not f.is_symlink():  # snapshots/ — симлинки на blobs/, не считаем дважды
                total += f.stat().st_size
        except OSError:
            pass  # файл исчез между листингом и stat (докачка/переименование)
    return total


def _construct_model():
    """Блокирующее скачивание (если весов нет в папке) + загрузка в память.
    fastembed сам решает, что нужно докачать; cache_dir — наша явная папка."""
    if TextEmbedding is None:
        raise EmbeddingUnavailableError("пакет fastembed не установлен на сервере")
    Path(config.EMBEDDING_CACHE_DIR).mkdir(parents=True, exist_ok=True)
    return TextEmbedding(
        model_name=config.EMBEDDING_MODEL,
        cache_dir=str(config.EMBEDDING_CACHE_DIR),
    )


def _loader() -> None:
    """Тело фонового потока: скачать (если надо) и загрузить модель."""
    global _state, _model, _load_error
    started = time.monotonic()
    try:
        model = _construct_model()
    except Exception as exc:  # noqa: BLE001 — любая ошибка (сеть, диск, битые веса)
        with _state_lock:
            _state = "error"
            _load_error = str(exc) or exc.__class__.__name__
        log.exception("embeddings: не удалось скачать/загрузить модель %s", config.EMBEDDING_MODEL)
        return
    with _state_lock:
        _model = model
        _load_error = None
        _state = "ready"
    log.info("embeddings: модель %s готова за %.1f с (папка: %s)",
             config.EMBEDDING_MODEL, time.monotonic() - started, config.EMBEDDING_CACHE_DIR)


def start_background_load() -> bool:
    """Запускает скачивание/загрузку модели в отдельном daemon-потоке и сразу
    возвращается — не блокирует ни вызывающего, ни event loop. Идемпотентна:
    если модель уже качается/грузится/готова — ничего не делает (False).
    Из state=error запускает новую попытку (так работает POST
    /embeddings/download). True — поток запущен."""
    global _state, _load_error, _loader_thread
    if not config.EMBEDDING_ENABLED or TextEmbedding is None:
        return False
    with _state_lock:
        if _state in ("downloading", "loading", "ready"):
            return False
        already_on_disk = _model_files_present()
        _state = "loading" if already_on_disk else "downloading"
        _load_error = None
        _loader_thread = threading.Thread(target=_loader, name="embeddings-loader", daemon=True)
        _loader_thread.start()
    if already_on_disk:
        log.info("embeddings: веса %s найдены в %s — загружаю в память (фоновый поток)",
                 config.EMBEDDING_MODEL, config.EMBEDDING_CACHE_DIR)
    else:
        log.info("embeddings: весов %s в %s нет — скачиваю в фоновом потоке, сервер отвечает как обычно",
                 config.EMBEDDING_MODEL, config.EMBEDDING_CACHE_DIR)
    return True


def load_blocking():
    """Синхронный вариант для CLI (warm_embeddings.py): скачать/загрузить
    модель в вызывающем потоке и вернуть её. Сервер этим не пользуется."""
    return _construct_model()


def _embed_sync(texts: list[str]) -> list[list[float]]:
    """Блокирующий вызов уже загруженной модели (только из executor'а).
    Сама модель здесь не грузится — см. embed_texts_async."""
    model = _model
    if model is None:
        raise EmbeddingNotReadyError(_state)
    with _infer_lock:
        # model.embed возвращает генератор numpy.ndarray, в порядке входа
        return [vec.tolist() for vec in model.embed(texts)]


async def embed_texts_async(texts: list[str]) -> tuple[list[list[float]], str, int]:
    """Async-обёртка для хендлера: считает CPU-bound инференс в executor'е,
    не блокируя event loop сервера (остальные запросы — /health, /analyze,
    /chat — продолжают обрабатываться, пока считается эмбеддинг).

    Возвращает (векторы в порядке texts, имя модели, размерность).
    Бросает EmbeddingNotReadyError, если модель ещё качается/грузится, и
    EmbeddingUnavailableError, если она недоступна (ошибка скачивания,
    fastembed не установлен).
    """
    if not is_ready():
        if _state == "error":
            raise EmbeddingUnavailableError(_load_error or "unknown error")
        if TextEmbedding is None:
            raise EmbeddingUnavailableError("пакет fastembed не установлен на сервере")
        start_background_load()  # страховка: сервер поднят не через server.py/on_startup
        raise EmbeddingNotReadyError(_state)
    loop = asyncio.get_running_loop()
    vectors = await loop.run_in_executor(None, _embed_sync, texts)
    dim = len(vectors[0]) if vectors else (known_dim() or 0)
    return vectors, config.EMBEDDING_MODEL, dim


def is_ready() -> bool:
    """Модель загружена в память и готова считать эмбеддинги."""
    return _model is not None


def status() -> dict:
    """Статус для /health и GET /embeddings — не запускает загрузку и не
    ходит в сеть: только то, что известно офлайн + состояние фонового потока.

    state: disabled | unavailable (нет fastembed) | idle | downloading |
           loading | ready | error
    download_progress — ПРИБЛИЗИТЕЛЬНО (байты в папке / size_in_GB из
    реестра fastembed), пока качается; не дойдёт до 1.0 раньше ready.
    """
    if not config.EMBEDDING_ENABLED:
        state = "disabled"
    elif TextEmbedding is None:
        state = "unavailable"
    else:
        state = _state

    info = {
        "enabled": config.EMBEDDING_ENABLED,
        "fastembed_installed": TextEmbedding is not None,
        "model": config.EMBEDDING_MODEL,
        "dim": known_dim(),
        "cache_dir": str(config.EMBEDDING_CACHE_DIR),
        "state": state,
        "loaded": _model is not None,
        "load_error": _load_error,
    }
    if state == "downloading":
        size_gb = (_lookup_model_metadata(config.EMBEDDING_MODEL) or {}).get("size_in_GB")
        downloaded = _dir_size_bytes()
        info["downloaded_mb"] = round(downloaded / 1e6, 1)
        if size_gb:
            info["download_progress"] = round(min(downloaded / (size_gb * 1e9), 0.99), 2)
    return info


def is_healthy() -> bool:
    """True, если сервис эмбеддингов в порядке: включён конфигом, пакет
    установлен и скачивание/загрузка не упали. Идущее скачивание — это НЕ
    поломка (см. is_ready — для "можно ли считать прямо сейчас")."""
    s = status()
    return bool(s["enabled"] and s["fastembed_installed"] and s["load_error"] is None)


# ---------------------------------------------------------------------------
# HTTP-хендлеры (POST/GET /embeddings)
# ---------------------------------------------------------------------------

_MAX_BATCH = 64  # разумный потолок на один запрос — защита от случайного гигантского батча


def _query_lang(request: web.Request) -> str | None:
    lang = request.query.get("lang")
    return lang.strip().lower() if lang else None


def _with_retry_after(resp: web.Response) -> web.Response:
    resp.headers["Retry-After"] = str(_RETRY_AFTER_SEC)
    return resp


async def handle_embeddings_info(request: web.Request) -> web.Response:
    """GET /embeddings — статус сервиса: включён ли, установлен ли fastembed,
    какая модель/размерность/папка сконфигурированы, state фонового
    скачивания и его примерный прогресс. Ничего не запускает и не
    считает. 200 — только когда модель готова (ready); 503 — во всех
    остальных случаях (с тем же телом; при скачивании/загрузке ещё и
    Retry-After — это не поломка, а "подожди")."""
    info = status()
    if info["state"] == "ready":
        return _json(info, status=200)
    resp = _json(info, status=503)
    if info["state"] in ("downloading", "loading"):
        _with_retry_after(resp)
    return resp


async def handle_embeddings_download(request: web.Request) -> web.Response:
    """POST /embeddings/download — (пере)запустить фоновое скачивание/загрузку.
    Нужна после state=error (например, не было сети до HuggingFace Hub).
    Ничего не ждёт: 202 — попытка запущена, 200 — уже качается/готова
    (ничего не делаем), 503 — сервис выключен/fastembed не установлен."""
    lang = _query_lang(request)
    if not config.EMBEDDING_ENABLED:
        return _json({"error": config._t("error.embeddings_disabled", lang=lang)}, status=503)
    if TextEmbedding is None:
        return _json(
            {"error": config._t("error.embeddings_unavailable",
                                detail="пакет fastembed не установлен на сервере", lang=lang)},
            status=503,
        )
    started = start_background_load()
    return _json(status(), status=202 if started else 200)


async def handle_embeddings(request: web.Request) -> web.Response:
    """POST /embeddings — текст(ы) -> вектор(ы).

    Тело — только JSON (в отличие от /chat и /analyze, тут нет ни картинок,
    ни multipart-файлов, отправлять есть смысл только текстом):
      {"text": "..."}              — один текст, ответ содержит "embedding"
      {"texts": ["...", "..."]}    — несколько сразу (для бэкфилла), ответ —
                                      только "embeddings" (без "embedding")
    lang можно передать query-параметром (?lang=en) или в теле — влияет
    только на язык текста ошибки, не на сам эмбеддинг.

    Пока модель скачивается/загружается в фоне (первый запуск) — 503 с
    Retry-After и state/прогрессом в теле; клиент повторяет запрос позже.
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
    except EmbeddingNotReadyError as exc:
        # Модель качается/грузится в фоне — не поломка, клиент должен повторить.
        body = {"error": config._t("error.embeddings_downloading", state=exc.state, lang=lang)}
        body.update({k: v for k, v in status().items() if k in ("state", "download_progress", "downloaded_mb")})
        return _with_retry_after(_json(body, status=503))
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
