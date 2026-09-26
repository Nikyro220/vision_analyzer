"""
server.py — HTTP-слой vision_analyzer_server: простые хендлеры (/,
/health, /lang, /config, /sampling, /models), сборка приложения aiohttp
и entrypoint. Сам /analyze — самый сложный путь — вынесен в analyze.py:
подготовка изображений и разбор трёх форматов тела запроса там.
Собственно общением с моделью (Ollama/vLLM) занимается backends.py.
Хендлеры /categories (просмотр и правка категорий на лету) — в
categories_api.py, бизнес-логика — в categories.py.

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
import logging

import aiohttp
from aiohttp import web

import backends
import categories_api
import config
from analyze import _json, handle_analyze
from config import locales


async def handle_index(request: web.Request) -> web.Response:
    return web.Response(
        text=config._page("index.body", backend=config.BACKEND),
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
        "Конфигурация обновлена извне: backend=%s ollama_host=%s vllm_url=%s",
        config.BACKEND, config.OLLAMA_HOST, config.VLLM_URL,
    )
    return _json({"ok": True, "backend": config.BACKEND, "ollama_host": config.OLLAMA_HOST, "vllm_url": config.VLLM_URL})


_SAMPLING_KEYS = ("temperature", "top_p", "top_k", "seed", "num_ctx", "num_predict", "think")


def _parse_resettable_int(v):
    """int(...), но 'auto'/'none'/'default'/'' (и None из JSON) → None,
    т.е. "вернуться к дефолту модели". Общий парсер для num_ctx и
    num_predict — у обоих одинаковая семантика сброса."""
    if v is None or (isinstance(v, str) and v.strip().lower() in ("", "auto", "none", "default")):
        return None
    return int(v)


# Каждому полю /sampling — своя функция разбора строки/значения из запроса
# в то, что кладётся в config.SAMPLING_DEFAULTS. Все бросают ValueError
# при некорректном значении — handle_sampling ловит это одним try/except
# на весь цикл, отдельная обработка ошибок на поле не нужна.
_SAMPLING_PARSERS = {
    "temperature": float,
    "top_p": float,
    "top_k": int,
    "seed": int,
    "num_ctx": _parse_resettable_int,
    "num_predict": _parse_resettable_int,
    "think": config.parse_think,
}


async def handle_sampling(request: web.Request) -> web.Response:
    """GET — вернуть текущие temperature/top_p/top_k/seed/num_ctx/
    num_predict/think, а также реально обнаруженный (если получилось)
    контекст vLLM.

    POST — изменить любое подмножество этих полей на лету, без
    перезапуска. Поля принимаются через query, JSON-тело или form-поле,
    как и /config.

    num_ctx — конфигурируемый дефолт, актуален ТОЛЬКО для backend=ollama
    (это per-request параметр options.num_ctx). У vLLM размер контекста
    фиксирован при запуске сервера (--max-model-len) и через POST
    /sampling не меняется — значение num_ctx для vllm просто игнорируется
    при реальном анализе. По умолчанию num_ctx = null — т.е. НЕ
    переопределяется, Ollama использует дефолт модели из её Modelfile
    (так было и до появления /sampling). Поднимать его стоит осознанно:
    большой num_ctx заметно увеличивает объём KV-cache и время prefill —
    вплоть до таймаута запроса (REQUEST_TIMEOUT), если не хватает VRAM.
    Задать явно — POST num_ctx=<число>; вернуть обратно на дефолт модели —
    POST num_ctx=auto (также подойдут "none"/"default"/"").

    num_predict — максимум токенов в одном ответе модели (Ollama:
    options.num_predict; vLLM: max_tokens). Та же семантика сброса —
    num_predict=auto. По умолчанию null (без лимита). Слишком маленькое
    значение обрежет JSON-отчёт на середине (в ответе будет "_raw"
    вместо разобранного отчёта).

    think — режим размышлений модели: true/false для большинства
    моделей, либо low/medium/high для GPT-OSS (см. config.parse_think).

    Вместо num_ctx в GET-ответе отдельным read-only-полем
    'vllm_context_window' отдаётся то, что реально узнали у самой vLLM
    (через GET /v1/models, поле max_model_len) — либо null, если бэкенд
    недоступен или конкретная сборка это поле не отдаёт.
    """
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
        for key, parse in _SAMPLING_PARSERS.items():
            if key in raw_values:
                config.SAMPLING_DEFAULTS[key] = parse(raw_values[key])
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
    # Статические /categories/summaries и /categories/order — раньше
    # динамического /categories/{name}, иначе он перехватит их как имя
    # категории (aiohttp резолвит роуты в порядке регистрации).
    app.router.add_get("/categories", categories_api.handle_categories_list)
    app.router.add_get("/categories/summaries", categories_api.handle_categories_summaries)
    app.router.add_post("/categories/order", categories_api.handle_categories_order)
    app.router.add_get("/categories/{name}", categories_api.handle_category_detail)
    app.router.add_post("/categories/{name}", categories_api.handle_category_upsert)
    return app


if __name__ == "__main__":
    logging.info(
        "Запускаю vision_analyzer_server на http://%s:%d (backend по умолчанию: %s)",
        config.SERVER_HOST, config.SERVER_PORT, config.BACKEND,
    )
    web.run_app(build_app(), host=config.SERVER_HOST, port=config.SERVER_PORT)