"""
vision_analyzer_server.py — локальный HTTP-сервер риск-триажа изображений
на базе локальной vision-модели.

Поддерживает два бэкенда:
  - vllm   — OpenAI-совместимый API (/v1/chat/completions), напр. gvllm2.service
  - ollama — /api/chat с картинкой в base64 и принудительным JSON-выводом

Запуск:
    python vision_analyzer_server.py

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
import json
import logging
from PIL import Image
import io
import aiohttp
from aiohttp import web

try:
    import vision_analyzer_prompt
except ImportError:
    vision_analyzer_prompt = None
    logging.warning(
        "vision_analyzer_server: AI/prompts/vision_analyzer_prompt.py не найден, "
    )

try:
    import locales
except ImportError:
    locales = None
    logging.warning(
        "vision_analyzer_server: locales.py не найден, POST /lang будет недоступен"
    )

try:
    import image_upscaler
except ImportError:
    image_upscaler = None
    logging.warning(
        "vision_analyzer_server: image_upscaler.py не найден, "
        "апскейлинг маленьких изображений отключён"
    )


def _current_lang() -> str:
    """Текущий язык сервера (для промпта модели и текстов ответов)."""
    return locales.DEFAULT_LANG if locales is not None else "ru"


def _t(key: str, **kwargs) -> str:
    """Короткий алиас для locales.get_formatted с фолбэком, если locales.py нет."""
    if locales is None:
        return f"???{key}???"
    return locales.get_formatted(key, **kwargs)


def _get_system_prompt(lang: str | None = None) -> str:
    if vision_analyzer_prompt is None:
        return ""
    return vision_analyzer_prompt.get_system_prompt(lang or _current_lang())


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

BACKEND = "vllm"  # "vllm" или "ollama" — бэкенд по умолчанию, можно переопределить в запросе

OLLAMA_HOST = "http://127.0.0.1:11434"
VLLM_URL = "http://host.docker.internal:8000/v1"

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 6769

REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=180)
DISCOVERY_TIMEOUT = aiohttp.ClientTimeout(total=10)

RISK_EMOJI = {
    "low": "🟢",
    "medium": "🟡",
    "high": "🔴",
}

def _risk_label(risk_level: str, lang: str | None = None) -> str:
    label = _t(f"risk.{risk_level}", lang=lang)
    return risk_level if label.startswith("???") else label


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


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# ---------------------------------------------------------------------------
# Автоопределение модели
# ---------------------------------------------------------------------------

_model_cache: dict[str, str] = {}


async def _discover_model(backend: str) -> str:
    """Возвращает первую доступную модель у бэкенда, кэширует результат.

    Явно заданная в запросе модель (параметр model=...) этот кэш не трогает
    и не использует — discovery нужен только когда модель не указана.
    """
    if backend in _model_cache:
        return _model_cache[backend]

    async with aiohttp.ClientSession(timeout=DISCOVERY_TIMEOUT) as session:
        if backend == "vllm":
            async with session.get(f"{VLLM_URL}/models") as resp:
                resp.raise_for_status()
                data = await resp.json()
            models = [m["id"] for m in data.get("data", [])]
        elif backend == "ollama":
            async with session.get(f"{OLLAMA_HOST}/api/tags") as resp:
                resp.raise_for_status()
                data = await resp.json()
            models = [m["name"] for m in data.get("models", [])]
        else:
            raise ValueError(_t("error.unknown_backend", backend=backend))

    if not models:
        raise RuntimeError(_t("error.no_models_returned", backend=backend))

    _model_cache[backend] = models[0]
    logging.info("Автоопределена модель для backend=%s: %s", backend, models[0])
    return models[0]


async def _ping_backend(backend: str) -> dict:
    """Проверяет доступность бэкенда и возвращает статус для /health.

    Никогда не бросает исключение наружу — любая ошибка превращается
    в {"ok": False, "error": ...}, чтобы падение одного бэкенда не мешало
    проверить остальные.
    """
    endpoint = VLLM_URL if backend == "vllm" else OLLAMA_HOST
    try:
        model = await _discover_model(backend)
        return {"ok": True, "endpoint": endpoint, "model": model}
    except aiohttp.ClientConnectorError:
        return {"ok": False, "endpoint": endpoint, "error": _t("error.backend_conn_refused")}
    except asyncio.TimeoutError:
        return {"ok": False, "endpoint": endpoint, "error": _t("error.backend_timeout")}
    except Exception as e:
        return {"ok": False, "endpoint": endpoint, "error": str(e)}


# ---------------------------------------------------------------------------
# Анализ изображения
# ---------------------------------------------------------------------------

_USER_PROMPT = "Проанализируй это изображение и верни JSON по заданной схеме."


async def _analyze_ollama(image_b64: str, model: str, lang: str | None = None) -> str:
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": _get_system_prompt(lang)},
            {"role": "user", "content": _USER_PROMPT, "images": [image_b64]},
        ],
        "options": {"temperature": 0.1},
    }

    async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as session:
        async with session.post(f"{OLLAMA_HOST}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()

    return data.get("message", {}).get("content", "").strip()


async def _analyze_vllm(image_b64: str, image_mime: str, model: str, lang: str | None = None) -> str:
    base_payload = {
        "model": model,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": _get_system_prompt(lang)},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _USER_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{image_mime};base64,{image_b64}"},
                    },
                ],
            },
        ],
    }

    async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as session:
        # Пытаемся получить строгий JSON через response_format (guided decoding).
        payload = {**base_payload, "response_format": {"type": "json_object"}}
        async with session.post(f"{VLLM_URL}/chat/completions", json=payload) as resp:
            if resp.status == 400:
                # Некоторые сборки vLLM без guided-decoding backend отвергают
                # response_format — повторяем запрос без него.
                logging.warning("vLLM отклонил response_format, повторяю запрос без него")
                async with session.post(f"{VLLM_URL}/chat/completions", json=base_payload) as resp2:
                    resp2.raise_for_status()
                    data = await resp2.json()
            else:
                resp.raise_for_status()
                data = await resp.json()

    return data["choices"][0]["message"]["content"].strip()


async def _analyze_image(
    image_b64: str,
    image_mime: str = "image/jpeg",
    backend: str = BACKEND,
    model: str | None = None,
    allow_fallback: bool = True,
    lang: str | None = None,
) -> tuple[dict, str]:  
    try:
        resolved_model = model or await _discover_model(backend)

        if backend == "vllm":
            content = await _analyze_vllm(image_b64, image_mime, resolved_model, lang=lang)
        elif backend == "ollama":
            content = await _analyze_ollama(image_b64, resolved_model, lang=lang)
        else:
            raise ValueError(_t("error.unknown_backend", backend=backend, lang=lang))
    except aiohttp.ClientConnectorError:
        if not allow_fallback:
            raise
        fallback_backend = "ollama" if backend == "vllm" else "vllm"
        logging.warning(
            "Бэкенд %r недоступен по подключению, пробую фолбэк на %r",
            backend, fallback_backend,
        )
        return await _analyze_image(
            image_b64, image_mime,
            backend=fallback_backend, model=None, allow_fallback=False, lang=lang,
        )

    try:
        return json.loads(content), backend
    except (json.JSONDecodeError, TypeError):
        logging.warning("vision_analyzer: модель (%s/%s) вернула невалидный JSON", backend, resolved_model)
        return {"_raw": content or _t("model.empty_response", lang=lang)}, backend


def _format_report(report: dict, source_name: str = "", lang: str | None = None) -> str:
    header = (
        _t("report.header_named", source_name=source_name, lang=lang)
        if source_name else _t("report.header_default", lang=lang)
    )

    if "_raw" in report:
        return _t("report.raw_fallback", header=header, raw=report["_raw"], lang=lang)

    risk_level = report.get("risk_level", "low")
    emoji = RISK_EMOJI.get(risk_level, "⚪️")
    risk_label = _risk_label(risk_level, lang=lang)
    needs_review = report.get("needs_human_review", False)

    lines = [
        header,
        _t("report.risk_level_line", emoji=emoji, risk_label=risk_label, lang=lang),
        _t("report.needs_review_yes", lang=lang) if needs_review else _t("report.needs_review_no", lang=lang),
        "",
        _t("report.description_label", lang=lang),
        report.get("description", "—"),
    ]

    text_on_image = report.get("text_on_image", "")
    if text_on_image:
        lines += ["", _t("report.text_on_image_label", lang=lang), text_on_image]

    context = report.get("context", "")
    if context:
        lines += ["", _t("report.context_label", lang=lang), context]

    signals = report.get("signals", [])
    if signals:
        lines += ["", _t("report.signals_label", lang=lang)]
        for s in signals:
            lines.append(f"{s.get('id', '?')}: {s.get('category', '—')} — {s.get('detail', '')}")

    rationale = report.get("rationale", "")
    if rationale:
        lines += ["", _t("report.rationale_label", lang=lang), rationale]

    recommendation = report.get("recommendation", "")
    if recommendation:
        lines += ["", _t("report.recommendation_label", lang=lang), recommendation]

    return "\n".join(lines)



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
        return None


async def _detect_image_mime(data: bytes) -> str | None:
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _detect_image_mime_sync, data)

async def handle_index(request: web.Request) -> web.Response:
    return web.Response(
        text=_t("index.body", backend=BACKEND),
        content_type="text/plain",
    )


async def handle_health(request: web.Request) -> web.Response:
    vllm_status, ollama_status = await asyncio.gather(
        _ping_backend("vllm"),
        _ping_backend("ollama"),
    )

    overall_ok = vllm_status["ok"] or ollama_status["ok"]
    default_ok = {"vllm": vllm_status, "ollama": ollama_status}[BACKEND]["ok"]

    return _json(
        {
            "ok": overall_ok,
            "default_backend": BACKEND,
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
            {"error": _t("error.lang_missing_param")}, status=400
        )
    lang = lang.strip().lower()

    available = list(locales.LOCALES.keys())
    if lang not in available:
        return _json(
            {"error": _t("error.lang_not_loaded", requested_lang=lang), "available": available},
            status=400,
        )

    locales.set_default_lang(lang)
    return _json({"ok": True, "default_lang": lang, "available": available})


async def handle_analyze(request: web.Request) -> web.Response:
    content_type = request.content_type

    override_backend = request.query.get("backend")
    override_model = request.query.get("model")
    override_lang = request.query.get("lang")

    # --- Новый путь: сырое изображение прямо в теле запроса ---
    if content_type.startswith("image/"):
        data = await request.read()
        if not data:
            return _json({"error": _t("error.empty_body", lang=override_lang)}, status=400)

        real_mime = await _detect_image_mime(data)
        if real_mime is None:
            return _json({"error": _t("error.not_image", lang=override_lang)}, status=400)

        if image_upscaler is not None:
            data, new_mime = await image_upscaler.upscale_if_needed(data, source_name="body")
            if new_mime:
                real_mime = new_mime

        image_b64 = base64.b64encode(data).decode("utf-8")
        tasks = [(image_b64, real_mime)]
        names = ["body"]

    # --- Старый путь: multipart/form-data ---
    elif content_type.startswith("multipart/"):
        reader = await request.multipart()
        tasks = []
        names = []

        async for part in reader:
            if part.name == "backend":
                override_backend = (await part.read(decode=True)).decode("utf-8").strip()
                continue
            if part.name == "model":
                override_model = (await part.read(decode=True)).decode("utf-8").strip()
                continue
            if part.name == "lang":
                override_lang = (await part.read(decode=True)).decode("utf-8").strip()
                continue
            if part.name not in ("images", "image"):
                continue

            data = await part.read(decode=True)
            real_mime = await _detect_image_mime(data)
            if real_mime is None:
                logging.warning("Пропускаю не-изображение: %s", part.filename)
                continue

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
                {"error": _t("error.no_images_multipart", lang=override_lang)},
                status=400,
            )

    else:
        return _json(
            {"error": _t("error.unsupported_content_type", lang=override_lang)},
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
                    "error": _t("error.lang_not_loaded", requested_lang=resolved_lang),
                    "available": list(locales.LOCALES.keys()),
                },
                status=400,
            )

    backend = (override_backend or BACKEND).strip().lower()
    if backend not in ("vllm", "ollama"):
        return _json(
            {"error": _t("error.unknown_backend", backend=backend, lang=resolved_lang)},
            status=400,
        )

    backend_was_explicit = bool(override_backend)

    logging.info(
        "Получено изображений в запросе: %d (%s) | backend=%s model=%s lang=%s",
        len(tasks), ", ".join(names), backend, override_model or "auto", resolved_lang or "default",
    )

    try:    
        results_raw = await asyncio.gather(*[
            _analyze_image(
                img_b64, img_mime,
                backend=backend, model=override_model,
                allow_fallback=not backend_was_explicit,
                lang=resolved_lang,
            )
            for img_b64, img_mime in tasks
        ])
    except aiohttp.ClientConnectorError:
        endpoint = VLLM_URL if backend == "vllm" else OLLAMA_HOST
        logging.error("Не удалось подключиться к бэкенду %s (%s)", backend, endpoint)
        return _json(
            {"error": _t("error.backend_unavailable", backend=backend, endpoint=endpoint, lang=resolved_lang)},
            status=502,
        )
    except (ValueError, RuntimeError) as e:
        return _json({"error": str(e)}, status=400)
    except Exception as e:
        logging.exception("Ошибка при анализе изображений: %s", e)
        return _json({"error": _t("error.analyze_failed", lang=resolved_lang)}, status=500)

    results = []
    for name, (report, actual_backend) in zip(names, results_raw):
        formatted = _format_report(report, source_name=name, lang=resolved_lang)
        print(formatted)
        print()
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
    return app


if __name__ == "__main__":
    logging.info(
        "Запускаю vision_analyzer_server на http://%s:%d (backend по умолчанию: %s)",
        SERVER_HOST, SERVER_PORT, BACKEND,
    )
    web.run_app(build_app(), host=SERVER_HOST, port=SERVER_PORT)