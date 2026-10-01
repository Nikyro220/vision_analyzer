"""
server.py — HTTP-слой vision_analyzer_server: простые хендлеры (/,
/health, /lang, /config, /sampling, /models), сборка приложения aiohttp
и entrypoint. Сам /analyze — самый сложный путь — вынесен в analyze.py:
подготовка изображений и разбор трёх форматов тела запроса там.
Собственно общением с моделью (Ollama/vLLM/Gemini) занимаются провайдеры
в providers/ (по модулю на бэкенд), конвейер /analyze — backends.py.
/providers и /providers/<name>/settings — самоописание провайдеров и их
настройка «на лету» (ключ API и т.п.), чтобы клиенты не зашивали список бэкендов.
Хендлеры /categories (только чтение дефолтов) — в categories_api.py,
бизнес-логика — в categories.py. Разовые категории на один вызов
передаются прямо в POST /analyze (см. analyze.py, поле "categories").
POST /chat — свободный диалог с моделью (текст + картинки, с историей),
без риск-JSON-схемы /analyze — вынесен в chat.py/chat_backends.py.

Поддерживает три бэкенда (см. providers/):
  - vllm   — OpenAI-совместимый API (/v1/chat/completions), напр. gvllm2.service
  - ollama — /api/chat с картинкой в base64 и принудительным JSON-выводом
  - gemini — Google Gemini (generateContent), по умолчанию gemini-2.5-flash

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

import categories_api
import chat
import config
import embeddings
import providers
from analyze import _json, handle_analyze
from config import locales


async def handle_index(request: web.Request) -> web.Response:
    return web.Response(
        text=config._page("index.body", backend=config.BACKEND),
        content_type="text/plain",
    )


def _visible_providers() -> list[providers.Provider]:
    """Провайдеры, которые показываем в /health и /models без явного ?backend=.

    Не настроенный провайдер (сейчас — Gemini без ключа API) скрыт, пока он
    не выбран бэкендом по умолчанию: иначе у тех, кто Gemini не использует,
    в статусе вечно висел бы «недоступный» бэкенд.
    """
    return [p for p in providers.all_providers() if p.is_configured() or p.name == config.BACKEND]


async def handle_health(request: web.Request) -> web.Response:
    shown = _visible_providers()
    statuses = await asyncio.gather(*(p.ping() for p in shown))
    backends_status = {p.name: st for p, st in zip(shown, statuses)}

    overall_ok = any(st["ok"] for st in backends_status.values())
    default_ok = bool((backends_status.get(config.BACKEND) or {}).get("ok"))

    return _json(
        {
            "ok": overall_ok,
            "default_backend": config.BACKEND,
            "default_backend_ok": default_ok,
            "backends": backends_status,
            # Эмбеддинги — вспомогательная возможность (см. embeddings.py), а не
            # основная функция сервера, поэтому её статус НЕ влияет на "ok"
            # выше: если модель эмбеддингов недоступна, риск-анализ и чат
            # продолжают работать как обычно.
            "embeddings": embeddings.status(),
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


_CONFIG_FIELDS = ("backend", "ollama_host", "vllm_url", "gemini_model")


def _config_snapshot() -> dict:
    # Ключ API Gemini наружу НЕ отдаём — только факт, что он задан.
    return {
        "backend": config.BACKEND,
        "available_backends": list(providers.names()),
        "ollama_host": config.OLLAMA_HOST,
        "vllm_url": config.VLLM_URL,
        "gemini_model": config.GEMINI_MODEL,
        "gemini_configured": bool(config.GEMINI_API_KEY),
    }


async def handle_config(request: web.Request) -> web.Response:
    """GET — вернуть текущие backend/ollama_host/vllm_url/gemini_model
    (+ gemini_configured — задан ли ключ API Gemini; сам ключ не отдаётся).

    POST — изменить одно или несколько полей "на лету", без перезапуска.
    Принимает backend, ollama_host, vllm_url, gemini_model (query, JSON-тело
    или form-поле, как и /lang) и gemini_api_key — последний ТОЛЬКО в теле
    запроса, не в query: URL попадает в логи, секрет там быть не должен.
    Хотя бы одно поле должно быть передано.
    """
    if request.method == "GET":
        return _json(_config_snapshot())

    if "gemini_api_key" in request.query:
        return _json({"error": config._t("error.gemini_key_in_query")}, status=400)

    try:
        if request.content_type == "application/json":
            body = await request.json() or {}
        else:
            body = await request.post()
    except Exception:
        body = {}
    if not hasattr(body, "get"):
        body = {}

    # Как и в /lang: query-параметры в приоритете над телом.
    values = {
        key: request.query.get(key) if request.query.get(key) is not None else body.get(key)
        for key in _CONFIG_FIELDS
    }
    new_api_key = body.get("gemini_api_key")

    if not any(values.values()) and not new_api_key:
        return _json({"error": config._t("error.config_missing_fields")}, status=400)

    new_backend = values["backend"]
    if new_backend:
        new_backend = new_backend.strip().lower()
        if not providers.is_known(new_backend):
            return _json({"error": config._t("error.unknown_backend", backend=new_backend)}, status=400)
        config.BACKEND = new_backend

    if values["ollama_host"]:
        config.OLLAMA_HOST = values["ollama_host"].strip()

    if values["vllm_url"]:
        config.VLLM_URL = values["vllm_url"].strip()

    if values["gemini_model"]:
        config.GEMINI_MODEL = values["gemini_model"].strip()

    if new_api_key:
        config.GEMINI_API_KEY = str(new_api_key).strip()

    # Кэш автоопределённых моделей (и обнаруженного контекста vLLM) мог
    # указывать на прежний хост/URL — сбрасываем, чтобы следующий запрос
    # заново определил всё там, куда сейчас реально указывают настройки.
    providers.reset_all_caches()

    logging.info(
        "Конфигурация обновлена извне: backend=%s ollama_host=%s vllm_url=%s gemini_model=%s gemini_api_key=%s",
        config.BACKEND, config.OLLAMA_HOST, config.VLLM_URL, config.GEMINI_MODEL,
        "<задан>" if config.GEMINI_API_KEY else "<не задан>",
    )
    return _json({"ok": True, **_config_snapshot()})


async def handle_providers(request: web.Request) -> web.Response:
    """GET /providers — описание ВСЕХ зарегистрированных провайдеров (в отличие от
    /health, где не настроенные скрыты): название, фолбэк, настроен ли, какие
    параметры /sampling он использует и какие поля можно менять через
    POST /providers/<name>/settings. По этому ответу клиент (веб-панель) сам строит
    свой UI — новый провайдер появляется там без правок на стороне клиента.
    Значения секретных полей не отдаются, только флаг "set"."""
    return _json({
        "default": config.BACKEND,
        "providers": [p.describe() for p in providers.all_providers()],
    })


async def handle_provider_settings(request: web.Request) -> web.Response:
    """POST /providers/<name>/settings — поменять поля провайдера «на лету».

    Тело — JSON {"<поле>": "<значение>", ..., "clear": ["<поле>", ...]}; допустимые
    поля — из settings_fields провайдера (см. GET /providers). Пустое значение
    означает «не менять», сброс — только явно через "clear". Только тело запроса, не
    query: URL попадает в логи, а среди полей бывают секреты (API-ключ).
    В ответе — результат проверки подключения (ping) уже с новыми настройками."""
    name = request.match_info["name"].strip().lower()
    if not providers.is_known(name):
        return _json({"error": config._t("error.unknown_backend", backend=name)}, status=400)
    provider = providers.get(name)

    if request.query:
        return _json({"error": config._t("error.settings_in_query")}, status=400)

    try:
        body = await request.json()
    except Exception:
        body = None
    if not isinstance(body, dict):
        return _json({"error": config._t("error.invalid_json_body")}, status=400)

    clear = body.pop("clear", None) or []
    if not isinstance(clear, list):
        return _json({"error": config._t("error.invalid_json_body")}, status=400)

    try:
        changed = provider.update_settings(body, clear)
    except ValueError as e:
        return _json({"error": str(e)}, status=400)
    if not changed:
        return _json({"error": config._t("error.settings_nothing_to_change")}, status=400)

    providers.reset_all_caches()
    logging.info("Настройки провайдера %s обновлены извне: %s", name, ", ".join(changed))  # значения не логируем
    return _json({"ok": True, "changed": changed, "status": await provider.ping(), "provider": provider.describe()})


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
        vllm_context_window = await providers.get("vllm").context_window() if providers.is_known("vllm") else None
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

    ?backend=<имя> — только один бэкенд (vllm | ollama | gemini); без
    параметра — все настроенные (см. _visible_providers).
    Не путать с /health: там только уже автоопределённая (первая) модель,
    здесь — полный список, чтобы было видно, из чего вообще выбирать.
    """
    backend_param = request.query.get("backend")
    if backend_param:
        name = backend_param.strip().lower()
        if not providers.is_known(name):
            return _json({"error": config._t("error.unknown_backend", backend=name)}, status=400)
        selected = [providers.get(name)]
    else:
        selected = _visible_providers()

    result = {}
    for p in selected:
        try:
            models = await p.list_models()
            result[p.name] = {"ok": True, "endpoint": p.endpoint, "models": models}
        except aiohttp.ClientConnectorError:
            result[p.name] = {"ok": False, "endpoint": p.endpoint, "error": config._t("error.backend_conn_refused")}
        except asyncio.TimeoutError:
            result[p.name] = {"ok": False, "endpoint": p.endpoint, "error": config._t("error.backend_timeout")}
        except Exception as e:
            result[p.name] = {"ok": False, "endpoint": p.endpoint, "error": str(e)}

    return _json(result)


async def _start_embeddings_download(app: web.Application) -> None:
    """on_startup: запускает скачивание/загрузку модели эмбеддингов в
    отдельном потоке и СРАЗУ возвращается — сервер начинает принимать запросы,
    не дожидаясь весов (см. embeddings.py: start_background_load)."""
    embeddings.start_background_load()


def build_app() -> web.Application:
    app = web.Application(client_max_size=64 * 1024 * 1024)  # до 64 МБ на запрос
    app.on_startup.append(_start_embeddings_download)
    app.router.add_get("/", handle_index)
    app.router.add_get("/health", handle_health)
    app.router.add_post("/analyze", handle_analyze)
    app.router.add_post("/chat", chat.handle_chat)
    app.router.add_post("/lang", handle_lang)
    app.router.add_get("/config", handle_config)
    app.router.add_post("/config", handle_config)
    app.router.add_get("/sampling", handle_sampling)
    app.router.add_post("/sampling", handle_sampling)
    app.router.add_get("/models", handle_models)
    app.router.add_get("/providers", handle_providers)
    app.router.add_post("/providers/{name}/settings", handle_provider_settings)
    app.router.add_get("/embeddings", embeddings.handle_embeddings_info)
    app.router.add_post("/embeddings", embeddings.handle_embeddings)
    app.router.add_post("/embeddings/download", embeddings.handle_embeddings_download)
    # Статический /categories/summaries — раньше динамического
    # /categories/{name}, иначе он перехватит его как имя категории
    # (aiohttp резолвит роуты в порядке регистрации). Только чтение —
    # правка/добавление дефолтов через API отключены (см.
    # categories_api.py); разовые категории идут через POST /analyze.
    app.router.add_get("/categories", categories_api.handle_categories_list)
    app.router.add_get("/categories/summaries", categories_api.handle_categories_summaries)
    app.router.add_get("/categories/{name}", categories_api.handle_category_detail)
    return app


if __name__ == "__main__":
    logging.info(
        "Запускаю vision_analyzer_server на http://%s:%d (backend по умолчанию: %s)",
        config.SERVER_HOST, config.SERVER_PORT, config.BACKEND,
    )
    web.run_app(build_app(), host=config.SERVER_HOST, port=config.SERVER_PORT)