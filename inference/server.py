"""
server.py — HTTP-слой vision_analyzer_server: все хендлеры (/, /health,
/analyze, /lang, /config, /sampling, /models), сборка приложения aiohttp
и entrypoint. Собственно общением с моделью (Ollama/vLLM) занимается
backends.py — этот файл только парсит запросы, вызывает backends и
форматирует ответы.

Поддерживает два бэкенда:
  - vllm   — OpenAI-совместимый API (/v1/chat/completions), напр. gvllm2.service
  - ollama — /api/chat с картинкой в base64 и принудительным JSON-выводом

Запуск:
    python server.py

Использование (curl):
	# проанализировать изображение, отправляет json ответ
	curl -X POST -H "Content-Type: image/" --data-binary "@/путь/к/файл.png" http://xxx.xxx.xxx.xxx:6769/analyze

	# состояние программы
	curl http://xxx.xxx.xxx.xxx:6769/health

	# индекс, использование программы и ее описание
    curl http://xxx.xxx.xxx.xxx:6769/
"""

import asyncio
import base64
import io
import json
import logging

import aiohttp
from aiohttp import web
from PIL import Image

import backends
import config
from config import image_upscaler, locales




def _json(data, status: int = 200) -> web.Response:
    """web.json_response с ensure_ascii=False и indent=2 по умолчанию.

    ensure_ascii=False — иначе aiohttp сериализует не-ASCII (кириллицу,
    эмодзи) в виде \\uXXXX escape-последовательностей, и curl/PowerShell
    показывают их как есть вместо нормального текста.
    indent=2 — чтобы все ответы (и /health, и /analyze, и ошибки)
    одинаково читались с отступами, а не слипались в одну строку.
    """
    return web.json_response(
        data,
        status=status,
        dumps=lambda obj: json.dumps(obj, ensure_ascii=False, indent=2),
    )


# ---------------------------------------------------------------------------
# HTTP-слой
# ---------------------------------------------------------------------------

def _detect_image_mime_sync(data: bytes) -> str | None:
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
        mime = Image.MIME.get(img.format)
        logging.info(
            "Pillow определил изображение: format=%s mime=%s size=%d байт",
            img.format, mime, len(data),
        )
        return mime
    except Exception as e:
        logging.warning(
            "Pillow не смог распознать содержимое как изображение (%d байт): %s",
            len(data), e,
        )
        return None # оно итак возвращает None, нафиг return None?



async def _detect_image_mime(data: bytes) -> str | None:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _detect_image_mime_sync, data)


async def handle_index(request: web.Request) -> web.Response:
    return web.Response(
        text=config._t("index.body", backend=config.BACKEND),
        content_type="text/plain",
    )


async def handle_health(request: web.Request) -> web.Response:
    vllm_status, ollama_status = await asyncio.gather(
        backends._ping_backend("vllm"),
        backends._ping_backend("ollama"),
    )

    overall_ok = vllm_status["ok"] or ollama_status["ok"]
    default_ok = {"vllm": vllm_status, "ollama": ollama_status}[config.BACKEND]["ok"]

    return _json(
        {
            "ok": overall_ok,
            "default_backend": config.BACKEND,
            "default_backend_ok": default_ok,
            "backends": {
                "vllm": vllm_status,
                "ollama": ollama_status,
            },
        },
        status=200 if overall_ok else 503,
    )


async def handle_lang(request: web.Request) -> web.Response:
    if locales is None:
        return _json(
            {"error": "locales.py не найден на сервере."}, status=503
        )

    # Принимаем и query-параметр (?lang=en), и JSON/form-тело ({"lang": "en"}).
    lang = request.query.get("lang")
    if lang is None:
        if request.content_type == "application/json":
            body = await request.json()
            lang = (body or {}).get("lang")
        else:
            data = await request.post()
            lang = data.get("lang")

    if not lang:
        return _json(
            {"error": config._t("error.lang_missing_param")}, status=400
        )
    lang = lang.strip().lower()

    available = list(locales.LOCALES.keys())
    if lang not in available:
        return _json(
            {"error": config._t("error.lang_not_loaded", requested_lang=lang), "available": available},
            status=400,
        )

    locales.set_default_lang(lang)
    return _json({"ok": True, "default_lang": lang, "available": available})


async def handle_config(request: web.Request) -> web.Response:
    """GET — вернуть текущий backend/ollama_host/vllm_url.

    POST — изменить один или несколько из них "на лету", без перезапуска.
    Принимает поля тремя способами — query, JSON-тело или form-поле,
    как и /lang: backend, ollama_host, vllm_url (все необязательные,
    но хотя бы одно должно быть передано).
    """
    if request.method == "GET":
        return _json({"backend": config.BACKEND, "ollama_host": config.OLLAMA_HOST, "vllm_url": config.VLLM_URL})

    new_backend = request.query.get("backend")
    new_ollama_host = request.query.get("ollama_host")
    new_vllm_url = request.query.get("vllm_url")

    # Как и в /lang: сначала query-параметры, и только если среди них нет
    # ни одного — пробуем распарсить тело (JSON или form-data).
    if new_backend is None and new_ollama_host is None and new_vllm_url is None:
        if request.content_type == "application/json":
            body = await request.json() or {}
        else:
            body = await request.post()
        new_backend = body.get("backend")
        new_ollama_host = body.get("ollama_host")
        new_vllm_url = body.get("vllm_url")

    if not any([new_backend, new_ollama_host, new_vllm_url]):
        return _json({"error": config._t("error.config_missing_fields")}, status=400)

    if new_backend:
        new_backend = new_backend.strip().lower()
        if new_backend not in ("vllm", "ollama"):
            return _json({"error": config._t("error.unknown_backend", backend=new_backend)}, status=400)
        config.BACKEND = new_backend

    if new_ollama_host:
        config.OLLAMA_HOST = new_ollama_host.strip()

    if new_vllm_url:
        config.VLLM_URL = new_vllm_url.strip()

    # Кэш автоопределённых моделей (и обнаруженного контекста vLLM) мог
    # указывать на прежний хост/URL — сбрасываем, чтобы следующий запрос
    # заново определил всё там, куда сейчас реально указывают
    # OLLAMA_HOST/VLLM_URL.
    backends._model_cache.clear()
    backends._vllm_context_cache.clear()

    logging.info(
        "Конфигурация обновлена извне: backend=%s ollama_host=%s vllm_url=%s", # f строки кому придумали
        config.BACKEND, config.OLLAMA_HOST, config.VLLM_URL,
    )
    return _json({"ok": True, "backend": config.BACKEND, "ollama_host": config.OLLAMA_HOST, "vllm_url": config.VLLM_URL})


_SAMPLING_PARSERS = {
    "temperature": lambda v: float(v),
    "top_p": lambda v: float(v),
    "top_k": lambda v: int(v),
    "seed": lambda v: int(v),
    "num_ctx": lambda v: _parse_resettable_int(v),
    "num_predict": lambda v: _parse_resettable_int(v),
    "think": lambda v: config.parse_think(v),
}

def _parse_resettable_int(v):
    if v is None or (isinstance(v, str) and v.strip().lower() in ("", "auto", "none", "default")):
        return None
    return int(v)

_SAMPLING_KEYS = ("temperature", "top_p", "top_k", "seed", "num_ctx", "num_predict", "think")

async def handle_sampling(request: web.Request) -> web.Response:
    if request.method == "GET":
        vllm_context_window = await backends._get_vllm_context_window()
        return _json({**config.SAMPLING_DEFAULTS, "vllm_context_window": vllm_context_window})

    raw_values = {}
    for key in _SAMPLING_KEYS:
        v = request.query.get(key)
        if v is not None:
            raw_values[key] = v

    if not raw_values:
        if request.content_type == "application/json":
            body = await request.json() or {}
        else:
            body = await request.post()
        for key in _SAMPLING_KEYS:
            if key in body:
                raw_values[key] = body[key]

    if not raw_values:
        return _json({"error": config._t("error.sampling_missing_fields")}, status=400)

    try:
        for key, parser in _SAMPLING_PARSERS.items():
            if key in raw_values:
                config.SAMPLING_DEFAULTS[key] = parser(raw_values[key])
    except (TypeError, ValueError):
        return _json(
            {"error": config._t("error.sampling_invalid_value", value=str(raw_values))},
            status=400,
        )
    logging.info("Параметры сэмплинга обновлены извне: %s", config.SAMPLING_DEFAULTS)
    return _json({"ok": True, **config.SAMPLING_DEFAULTS})


async def handle_models(request: web.Request) -> web.Response:
    """Сканирует бэкенд(ы) и возвращает список всех доступных там моделей.

    ?backend=vllm|ollama — только один бэкенд; без параметра — оба сразу.
    Не путать с /health: там только уже автоопределённая (первая) модель,
    здесь — полный список, чтобы было видно, из чего вообще выбирать.
    """
    backend_param = request.query.get("backend")
    backend_list = [backend_param.strip().lower()] if backend_param else ["vllm", "ollama"]

    result = {}
    for b in backend_list:
        if b not in ("vllm", "ollama"):
            return _json({"error": config._t("error.unknown_backend", backend=b)}, status=400)
        endpoint = config.VLLM_URL if b == "vllm" else config.OLLAMA_HOST
        try:
            models = await backends._list_models(b)
            result[b] = {"ok": True, "endpoint": endpoint, "models": models}
        except aiohttp.ClientConnectorError:
            result[b] = {"ok": False, "endpoint": endpoint, "error": config._t("error.backend_conn_refused")}
        except asyncio.TimeoutError:
            result[b] = {"ok": False, "endpoint": endpoint, "error": config._t("error.backend_timeout")}
        except Exception as e:
            result[b] = {"ok": False, "endpoint": endpoint, "error": str(e)}

    return _json(result)


async def handle_analyze(request: web.Request) -> web.Response:
    content_type = request.content_type

    override_backend = request.query.get("backend")
    override_model = request.query.get("model")
    override_lang = request.query.get("lang")
    override_history = request.query.get("history")
    override_caption = request.query.get("caption")

    _MULTIPART_TEXT_FIELDS = {"backend": True, "model": True, "lang": True, "history": True, "caption": False}

    # --- Новый путь: сырое изображение прямо в теле запроса ---
    if content_type.startswith("image/"):
        data = await request.read()
        if not data:
            return _json({"error": config._t("error.empty_body", lang=override_lang)}, status=400)

        real_mime = await _detect_image_mime(data)
        if real_mime is None:
            return _json({"error": config._t("error.not_image", lang=override_lang)}, status=400)

        if image_upscaler is not None:
            data, new_mime = await image_upscaler.upscale_if_needed(data, source_name="body")
            if new_mime:
                real_mime = new_mime

        image_b64 = base64.b64encode(data).decode("utf-8")
        tasks = [(image_b64, real_mime)]
        names = ["body"]
        captions = [override_caption]

    # --- Путь для UI/ботов: всё одним JSON-телом, включая историю ---
    elif content_type == "application/json":
        body = await request.json() or {}
        override_backend = override_backend or body.get("backend")
        override_model = override_model or body.get("model")
        override_lang = override_lang or body.get("lang")
        override_history = override_history or body.get("history")
        override_caption = override_caption or body.get("caption")

        raw_images = body.get("images")
        if not raw_images:
            single = body.get("image")
            raw_images = [single] if single else []

        if not raw_images:
            return _json({"error": config._t("error.empty_body", lang=override_lang)}, status=400)

        tasks = []
        names = []
        captions = []
        for idx, item in enumerate(raw_images):
            # Элемент — либо просто строка с картинкой (старый формат,
            # caption общий на весь батч из override_caption), либо объект
            # {"image": "...", "caption": "..."} — свой caption на картинку.
            if isinstance(item, dict):
                img = item.get("image")
                item_caption = item.get("caption") or override_caption
            else:
                img = item
                item_caption = override_caption

            try:
                data = base64.b64decode(backends._strip_data_url(img))
            except Exception:
                return _json({"error": config._t("error.not_image", lang=override_lang)}, status=400)

            real_mime = await _detect_image_mime(data)
            if real_mime is None:
                return _json({"error": config._t("error.not_image", lang=override_lang)}, status=400)

            source_name = f"json_{idx + 1}"
            if image_upscaler is not None:
                data, new_mime = await image_upscaler.upscale_if_needed(data, source_name=source_name)
                if new_mime:
                    real_mime = new_mime

            tasks.append((base64.b64encode(data).decode("utf-8"), real_mime))
            names.append(source_name)
            captions.append(item_caption)


    # --- Старый путь: multipart/form-data ---
    elif content_type.startswith("multipart/"):
        reader = await request.multipart()
        tasks, names = [], []
        overrides: dict[str, Any] = {}

        async for part in reader:
            if part.name in _MULTIPART_TEXT_FIELDS:
                raw = (await part.read(decode=True)).decode("utf-8")
                overrides[part.name] = raw.strip() if _MULTIPART_TEXT_FIELDS[part.name] else raw
                continue
            if part.name not in ("images", "image"):
                continue

            data = await part.read(decode=True)
            real_mime = await _detect_image_mime(data)
            if real_mime is None:
                logging.warning("Пропускаю не-изображение: %s", part.filename)
                continue
            ...

            source_name = part.filename or f"image_{len(names) + 1}"

            if image_upscaler is not None:
                data, new_mime = await image_upscaler.upscale_if_needed(data, source_name=source_name)
                if new_mime:
                    real_mime = new_mime

            image_b64 = base64.b64encode(data).decode("utf-8")
            names.append(source_name)
            tasks.append((image_b64, real_mime))

        if not tasks:
            return _json(
                {"error": config._t("error.no_images_multipart", lang=override_lang)},
                status=400,
            )
        captions = [override_caption] * len(tasks)

    else:
        return _json(
            {"error": config._t("error.unsupported_content_type", lang=override_lang)},
            status=400,
        )

    # Разовый (per-request) язык ответа/промпта — не трогает глобальный
    # locales.DEFAULT_LANG, поэтому параллельные запросы с разными lang
    # не мешают друг другу.
    resolved_lang = None
    if override_lang:
        resolved_lang = override_lang.strip().lower()
        if locales is not None and resolved_lang not in locales.LOCALES:
            return _json(
                {
                    "error": config._t("error.lang_not_loaded", requested_lang=resolved_lang),
                    "available": list(locales.LOCALES.keys()),
                },
                status=400,
            )

    backend = (override_backend or config.BACKEND).strip().lower()
    if backend not in ("vllm", "ollama"):
        return _json(
            {"error": config._t("error.unknown_backend", backend=backend, lang=resolved_lang)},
            status=400,
        )

    try:
        resolved_history = backends._parse_history_json(override_history)
    except ValueError:
        return _json({"error": config._t("error.invalid_history", lang=resolved_lang)}, status=400)

    backend_was_explicit = bool(override_backend)

    logging.info(
        "Получено изображений в запросе: %d (%s) | backend=%s model=%s lang=%s history=%d",
        len(tasks), ", ".join(names), backend, override_model or "auto",
        resolved_lang or "default", len(resolved_history),
    )

    try:
        results_raw = await asyncio.gather(*[
            backends._analyze_image(
                img_b64, img_mime,
                backend=backend, model=override_model,
                allow_fallback=not backend_was_explicit,
                lang=resolved_lang, history=resolved_history,
                caption=cap,
            )
            for (img_b64, img_mime), cap in zip(tasks, captions)
        ])
    except aiohttp.ClientConnectorError:
        endpoint = config.VLLM_URL if backend == "vllm" else config.OLLAMA_HOST
        logging.error("Не удалось подключиться к бэкенду %s (%s)", backend, endpoint)
        return _json(
            {"error": config._t("error.backend_unavailable", backend=backend, endpoint=endpoint, lang=resolved_lang)},
            status=502,
        )
    except asyncio.TimeoutError:
        logging.warning(
            "Таймаут при обращении к backend=%s (не уложились в %.0fс) — "
            "проверь num_ctx (/sampling) и размер history, бэкенд может быть перегружен",
            backend, config.REQUEST_TIMEOUT.total,
        )
        return _json({"error": config._t("error.backend_timeout", lang=resolved_lang)}, status=504)
    except (ValueError, RuntimeError) as e:
        return _json({"error": str(e)}, status=400)
    except Exception as e:
        logging.exception("Ошибка при анализе изображений: %s", e)
        return _json({"error": config._t("error.analyze_failed", lang=resolved_lang)}, status=500)

    results = []
    for name, (report, actual_backend) in zip(names, results_raw):
        if "_raw" in report:
            logging.info(
                "Анализ %r завершён (backend=%s), ответ модели не по JSON-схеме",
                name, actual_backend,
            )
        else:
            logging.info("Анализ %r завершён (backend=%s)", name, actual_backend)
        results.append({"file": name, "backend": actual_backend, "report": report})

    return _json(
        {"count": len(results), "requested_backend": backend, "results": results},
    )


def build_app() -> web.Application:
    app = web.Application(client_max_size=64 * 1024 * 1024)  # до 64 МБ на запрос
    app.router.add_get("/", handle_index)
    app.router.add_get("/health", handle_health)
    app.router.add_post("/analyze", handle_analyze)
    app.router.add_post("/lang", handle_lang)
    app.router.add_get("/config", handle_config)
    app.router.add_post("/config", handle_config)
    app.router.add_get("/sampling", handle_sampling)
    app.router.add_post("/sampling", handle_sampling)
    app.router.add_get("/models", handle_models)
    return app


if __name__ == "__main__":
    logging.info(
        "Запускаю vision_analyzer_server на http://%s:%d (backend по умолчанию: %s)",
        config.SERVER_HOST, config.SERVER_PORT, config.BACKEND,
    )
    web.run_app(build_app(), host=config.SERVER_HOST, port=config.SERVER_PORT)