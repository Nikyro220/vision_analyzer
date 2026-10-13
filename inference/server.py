"""
server.py — HTTP-слой vision_analyzer_server: простые хендлеры (/,
/health, /lang, /config, /providers, /models), сборка приложения aiohttp
и entrypoint. Сам /analyze — самый сложный путь — вынесен в tools/vision_analyzer/analyze.py:
подготовка изображений и разбор трёх форматов тела запроса там.
Собственно общением с моделью (Ollama/vLLM/Gemini/Anthropic) занимаются провайдеры
в providers/ (по модулю на бэкенд), конвейер /analyze — tools/vision_analyzer/backends.py.
/providers — самоописание провайдеров (в т.ч. какой ключ API им нужен), чтобы клиенты не
зашивали список бэкендов. Ключи API сервер НЕ хранит: клиент присылает ключ в каждом запросе
заголовком X-Api-Key-<Провайдер> (см. credentials_middleware ниже).
Хендлеры /categories (только чтение дефолтов) — в tools/vision_analyzer/categories_api.py,
бизнес-логика — в tools/vision_analyzer/categories.py. Разовые категории на один вызов
передаются прямо в POST /analyze (см. analyze.py, поле "categories").
POST /chat — свободный диалог с моделью (текст + картинки, с историей),
без риск-JSON-схемы /analyze — вынесен в chat.py/chat_backends.py.

Поддерживает четыре бэкенда (см. providers/):
  - vllm   — OpenAI-совместимый API (/v1/chat/completions), напр. gvllm2.service
  - ollama — /api/chat с картинкой в base64 и принудительным JSON-выводом
  - gemini — Google Gemini (generateContent), по умолчанию gemini-2.5-flash
  - anthropic — Anthropic Claude (Messages API), по умолчанию claude-sonnet-5-5

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
import time

import aiohttp
from aiohttp import web

import chat
import config
import providers
from config import locales
from http_utils import _json
from tools.embeddings import embeddings
from tools.vision_analyzer import categories_api
from tools.vision_analyzer.analyze import handle_analyze


@web.middleware
async def credentials_middleware(request: web.Request, handler):
    """Раскладывает заголовки X-Api-Key-<Провайдер> в providers.request_credentials — только на
    время этого запроса. Ключи нигде не сохраняются и не логируются; ContextVar гарантирует,
    что параллельные запросы не видят ключи друг друга."""
    prefix = providers.CREDENTIAL_HEADER_PREFIX.lower()
    creds = {
        name[len(prefix):].lower(): value.strip()
        for name, value in request.headers.items()
        if name.lower().startswith(prefix) and value.strip()
    }
    token = providers.request_credentials.set(creds)
    try:
        return await handler(request)
    finally:
        providers.request_credentials.reset(token)


async def _close_providers(app: web.Application) -> None:
    """on_cleanup: закрывает HTTP-сессии провайдеров (иначе «Unclosed client session» при остановке)."""
    for p in providers.all_providers():
        await p.aclose()


async def handle_index(request: web.Request) -> web.Response:
    return web.Response(
        text=config._page("index.body", backend=config.BACKEND),
        content_type="text/plain",
    )


def _visible_providers() -> list[providers.Provider]:
    """Провайдеры, которые показываем в /health и /models без явного ?backend=.

    Не настроенный провайдер (облачный без ключа API: Gemini, Anthropic) скрыт, пока он
    не выбран бэкендом по умолчанию: иначе у тех, кто Gemini не использует,
    в статусе вечно висел бы «недоступный» бэкенд.
    """
    return [p for p in providers.all_providers() if p.is_configured() or p.name == config.BACKEND]


# {(имя, endpoint): (время_устаревания, результат ping)} — только для локальных провайдеров.
# endpoint в ключе: POST /config меняет адрес, и старый результат автоматически не подходит.
_health_cache: dict[tuple[str, str], tuple[float, dict]] = {}


async def _ping_bounded(p: providers.Provider) -> dict:
    """ping() с жёстким потолком по времени и (для локальных бэкендов) коротким кэшем.

    Без этого /health ждал самый медленный бэкенд: недоступный хост без отказа в
    соединении (пакеты молча дропаются) держал запрос до DISCOVERY_TIMEOUT.
    Облачные провайдеры не кэшируем: их результат зависит от ключа API текущего запроса.
    """
    cacheable = p.credential is None
    key = (p.name, p.endpoint)
    now = time.monotonic()
    if cacheable:
        hit = _health_cache.get(key)
        if hit and hit[0] > now:
            return hit[1]
    try:
        st = await asyncio.wait_for(p.ping(), timeout=config.HEALTH_PROBE_TIMEOUT)
    except asyncio.TimeoutError:
        st = {"ok": False, "endpoint": p.endpoint, "error": config._t("error.backend_timeout")}
    if cacheable:
        ttl = config.HEALTH_CACHE_OK_TTL if st.get("ok") else config.HEALTH_CACHE_FAIL_TTL
        _health_cache[key] = (time.monotonic() + ttl, st)
    return st


async def handle_health(request: web.Request) -> web.Response:
    shown = _visible_providers()
    statuses = await asyncio.gather(*(_ping_bounded(p) for p in shown))
    backends_status = {p.name: st for p, st in zip(shown, statuses)}

    overall_ok = any(st["ok"] for st in backends_status.values())
    default_ok = bool((backends_status.get(config.BACKEND) or {}).get("ok"))

    return _json(
        {
            "ok": overall_ok,
            "default_backend": config.BACKEND,
            "default_backend_ok": default_ok,
            "backends": backends_status,
            # Эмбеддинги — вспомогательная возможность (см. tools/embeddings/embeddings.py), а не
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


_CONFIG_FIELDS = ("backend", "ollama_host", "vllm_url", "gemini_model", "anthropic_model")

# Ключи API здесь не принимаются: поле -> текст ошибки (сервер ключи не хранит).
_REJECTED_KEY_FIELDS = {
    "gemini_api_key": "error.gemini_key_not_stored",
    "anthropic_api_key": "error.anthropic_key_not_stored",
}


def _config_snapshot() -> dict:
    # Ключей API сервер не хранит (их присылает клиент в каждом запросе), поэтому и в снимке их нет.
    return {
        "backend": config.BACKEND,
        "available_backends": list(providers.names()),
        "ollama_host": config.OLLAMA_HOST,
        "vllm_url": config.VLLM_URL,
        "gemini_model": config.GEMINI_MODEL,
        "anthropic_model": config.ANTHROPIC_MODEL,
    }


async def handle_config(request: web.Request) -> web.Response:
    """GET — вернуть текущие backend/ollama_host/vllm_url/gemini_model/anthropic_model.

    POST — изменить одно или несколько полей "на лету", без перезапуска.
    Принимает backend, ollama_host, vllm_url, gemini_model, anthropic_model (query, JSON-тело
    или form-поле, как и /lang). Хотя бы одно поле должно быть передано.

    Ключи API здесь НЕ принимаются и нигде не хранятся: клиент присылает ключ провайдера
    в каждом запросе заголовком X-Api-Key-<Имя> (см. providers/base.py).
    """
    if request.method == "GET":
        return _json(_config_snapshot())

    for field, message_key in _REJECTED_KEY_FIELDS.items():
        if field in request.query or field in request.headers:
            return _json({"error": config._t(message_key)}, status=400)

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
    for field, message_key in _REJECTED_KEY_FIELDS.items():
        if body.get(field):
            return _json({"error": config._t(message_key)}, status=400)

    if not any(values.values()):
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

    if values["anthropic_model"]:
        config.ANTHROPIC_MODEL = values["anthropic_model"].strip()

    # Кэш автоопределённых моделей (и обнаруженного контекста vLLM) мог
    # указывать на прежний хост/URL — сбрасываем, чтобы следующий запрос
    # заново определил всё там, куда сейчас реально указывают настройки.
    providers.reset_all_caches()
    _health_cache.clear()

    logging.info(
        "Конфигурация обновлена извне: backend=%s ollama_host=%s vllm_url=%s gemini_model=%s anthropic_model=%s",
        config.BACKEND, config.OLLAMA_HOST, config.VLLM_URL, config.GEMINI_MODEL, config.ANTHROPIC_MODEL,
    )
    return _json({"ok": True, **_config_snapshot()})


async def handle_providers(request: web.Request) -> web.Response:
    """GET /providers — описание ВСЕХ зарегистрированных провайдеров (в отличие от
    /health, где не настроенные скрыты): название, фолбэк, настроен ли, какие
    параметры генерации (settings) у него есть и какой секрет (API-ключ) ему нужен. По этому ответу клиент (веб-панель) сам строит
    свой UI — новый провайдер появляется там без правок на стороне клиента.
    Поле "credential" описывает секрет (API-ключ), который клиент должен присылать в
    каждом запросе заголовком из credential.header; "configured" — пришёл ли ключ в ЭТОМ
    запросе (сервер ключи не хранит)."""
    return _json({
        "default": config.BACKEND,
        "providers": [p.describe() for p in providers.all_providers()],
    })


async def _read_params(request: web.Request) -> dict:
    """Параметры запроса одним словарём: JSON-тело или form-поля, поверх них — query
    (query в приоритете, как в /config и /lang)."""
    params: dict = {}
    if request.content_type == "application/json":
        try:
            body = await request.json()
        except Exception:
            body = None
        if isinstance(body, dict):
            params.update(body)
    elif request.can_read_body:
        params.update(await request.post())
    params.update(request.query)
    return params


def _setting_error(e: providers.SettingError) -> web.Response:
    if e.kind == "unknown":
        message = config._t(
            "error.setting_unknown", backend=e.provider, param=e.key, available=", ".join(e.available) or "-",
        )
    else:
        message = config._t("error.setting_invalid", backend=e.provider, param=e.key, reason=e.reason)
    return _json({"error": message, "provider": e.provider, "key": e.key}, status=400)


def _settings_payload(provider: providers.Provider) -> dict:
    return {"provider": provider.name, "values": provider.settings, "settings": provider.describe_settings()}


async def handle_provider_settings(request: web.Request) -> web.Response:
    """Настройки ОДНОГО провайдера — /providers/{name}/settings. У каждого провайдера свой набор
    параметров, свои значения по умолчанию и свои допустимые значения (providers/settings.py);
    изменение настроек одного провайдера не затрагивает другие.

    GET    — текущие значения и схема параметров (тип, диапазон/варианты, значение по умолчанию).
    POST   — изменить любое подмножество параметров (query, JSON-тело или form). Всё или ничего:
             неизвестный параметр или недопустимое значение → 400, и не меняется ничего.
             Для параметров, где это разрешено, значение auto сбрасывает «не задано».
    DELETE — вернуть значения по умолчанию: все или только перечисленные в ?keys=a,b.
    """
    name = request.match_info["name"].strip().lower()
    if not providers.is_known(name):
        return _json({"error": config._t("error.unknown_backend", backend=name)}, status=404)
    provider = providers.get(name)

    try:
        if request.method == "POST":
            params = await _read_params(request)
            if not params:
                return _json({
                    "error": config._t(
                        "error.sampling_missing_fields", backend=name, available=", ".join(provider.settings) or "-",
                    ),
                }, status=400)
            provider.update_settings(params)
            logging.info("Настройки провайдера %s обновлены: %s", name, provider.settings)
            return _json({"ok": True, **_settings_payload(provider)})

        if request.method == "DELETE":
            keys = [k.strip() for k in request.query.get("keys", "").split(",") if k.strip()] or None
            provider.reset_settings(keys)
            logging.info("Настройки провайдера %s сброшены (%s): %s", name, keys or "все", provider.settings)
            return _json({"ok": True, **_settings_payload(provider)})
    except providers.SettingError as e:
        return _setting_error(e)

    return _json(_settings_payload(provider))


async def handle_sampling(request: web.Request) -> web.Response:
    """УСТАРЕЛО (оставлено для старых клиентов): GET/POST /sampling — то же, что
    /providers/{name}/settings, но для одного провайдера, выбранного параметром ?backend=
    (по умолчанию — бэкенд сервера по умолчанию). Раньше параметры были общими для всех
    бэкендов — теперь у каждого провайдера свои, и этот хендлер работает ровно с одним.

    Параметры, которых у выбранного провайдера нет (например, num_ctx у vLLM), не считаются
    ошибкой: они возвращаются в поле "ignored". Недопустимое значение — 400, как и раньше.
    """
    name = (request.query.get("backend") or config.BACKEND).strip().lower()
    if not providers.is_known(name):
        return _json({"error": config._t("error.unknown_backend", backend=name)}, status=400)
    provider = providers.get(name)
    headers = {"Deprecation": "true"}

    if request.method == "GET":
        vllm_context_window = await providers.get("vllm").context_window() if providers.is_known("vllm") else None
        return _json(
            {**provider.settings, "backend": name, "vllm_context_window": vllm_context_window}, headers=headers,
        )

    params = {k: v for k, v in (await _read_params(request)).items() if k != "backend"}
    known = {k: v for k, v in params.items() if k in provider.settings}
    ignored = sorted(set(params) - set(known))
    if not known:
        return _json({
            "error": config._t(
                "error.sampling_missing_fields", backend=name, available=", ".join(provider.settings) or "-",
            ),
            "ignored": ignored,
        }, status=400, headers=headers)
    try:
        provider.update_settings(known)
    except providers.SettingError as e:
        return _setting_error(e)

    logging.info("Настройки провайдера %s обновлены через /sampling: %s", name, provider.settings)
    return _json({"ok": True, "backend": name, **provider.settings, "ignored": ignored}, headers=headers)


async def handle_models(request: web.Request) -> web.Response:
    """Сканирует бэкенд(ы) и возвращает список всех доступных там моделей.

    ?backend=<имя> — только один бэкенд (vllm | ollama | gemini | anthropic); без
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

    async def scan(p: providers.Provider) -> dict:
        try:
            models = await p.list_models()
            return {"ok": True, "endpoint": p.endpoint, "models": models}
        except aiohttp.ClientConnectorError:
            return {"ok": False, "endpoint": p.endpoint, "error": config._t("error.backend_conn_refused")}
        except asyncio.TimeoutError:
            return {"ok": False, "endpoint": p.endpoint, "error": config._t("error.backend_timeout")}
        except Exception as e:
            return {"ok": False, "endpoint": p.endpoint, "error": str(e)}

    # Параллельно: раньше провайдеры опрашивались по очереди, и один недоступный бэкенд
    # (таймаут подключения) задерживал ответ про остальные.
    scanned = await asyncio.gather(*(scan(p) for p in selected))
    result = {p.name: item for p, item in zip(selected, scanned)}

    return _json(result)


async def _start_embeddings_download(app: web.Application) -> None:
    """on_startup: запускает скачивание/загрузку модели эмбеддингов в
    отдельном потоке и СРАЗУ возвращается — сервер начинает принимать запросы,
    не дожидаясь весов (см. tools/embeddings/embeddings.py: start_background_load)."""
    embeddings.start_background_load()


def build_app() -> web.Application:
    app = web.Application(client_max_size=64 * 1024 * 1024, middlewares=[credentials_middleware])  # до 64 МБ на запрос
    app.on_startup.append(_start_embeddings_download)
    app.on_cleanup.append(_close_providers)
    app.router.add_get("/", handle_index)
    app.router.add_get("/health", handle_health)
    app.router.add_post("/analyze", handle_analyze)
    app.router.add_post("/chat", chat.handle_chat)
    app.router.add_post("/lang", handle_lang)
    app.router.add_get("/config", handle_config)
    app.router.add_post("/config", handle_config)
    app.router.add_get("/sampling", handle_sampling)  # устарело: см. handle_sampling
    app.router.add_post("/sampling", handle_sampling)
    app.router.add_get("/models", handle_models)
    app.router.add_get("/providers", handle_providers)
    app.router.add_route("*", "/providers/{name}/settings", handle_provider_settings)
    app.router.add_get("/embeddings", embeddings.handle_embeddings_info)
    app.router.add_post("/embeddings", embeddings.handle_embeddings)
    app.router.add_post("/embeddings/download", embeddings.handle_embeddings_download)
    # Статический /categories/summaries — раньше динамического
    # /categories/{name}, иначе он перехватит его как имя категории
    # (aiohttp резолвит роуты в порядке регистрации). Только чтение —
    # правка/добавление дефолтов через API отключены (см.
    # tools/vision_analyzer/categories_api.py); разовые категории идут через POST /analyze.
    app.router.add_get("/categories", categories_api.handle_categories_list)
    app.router.add_get("/categories/summaries", categories_api.handle_categories_summaries)
    app.router.add_get("/categories/{name}", categories_api.handle_category_detail)
    return app


def _port_in_use(host: str, port: int) -> bool:
    """True, если порт уже занят (например, забытым фоновым экземпляром)."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # На Windows SO_REUSEADDR разрешает "угон" порта, поэтому не ставим его.
        try:
            sock.bind((host, port))
        except OSError:
            return True
    return False


def _find_pid_on_port(port: int) -> str | None:
    """Best-effort поиск PID слушателя порта через netstat (Windows/Linux)."""
    import subprocess
    try:
        out = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        return None
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and "LISTEN" in line.upper() and parts[1].endswith(f":{port}"):
            return parts[-1]
    return None


if __name__ == "__main__":
    if _port_in_use(config.SERVER_HOST, config.SERVER_PORT):
        pid = _find_pid_on_port(config.SERVER_PORT)
        logging.error(
            "Порт %d уже занят%s — скорее всего, сервер уже запущен в фоне. "
            "Закройте старый процесс и запустите снова.",
            config.SERVER_PORT, f" (PID {pid})" if pid else "",
        )
        raise SystemExit(1)
    logging.info(
        "Запускаю vision_analyzer_server на http://%s:%d (backend по умолчанию: %s)",
        config.SERVER_HOST, config.SERVER_PORT, config.BACKEND,
    )
    web.run_app(build_app(), host=config.SERVER_HOST, port=config.SERVER_PORT)