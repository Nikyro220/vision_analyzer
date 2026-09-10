"""
vision_analyzer_server.py — локальный HTTP-сервер риск-триажа изображений
на базе локальной vision-модели.

Поддерживает два бэкенда:
  - vllm   — OpenAI-совместимый API (/v1/chat/completions), напр. gvllm2.service
  - ollama — /api/chat с картинкой в base64 и принудительным JSON-выводом

Модель определяется автоматически (первая из /v1/models или /api/tags и
кэшируется), либо задаётся явно в запросе полем "model". Бэкенд по
умолчанию задаётся константой BACKEND, но тоже может быть переопределён
в запросе полем "backend".

Запуск:
    python vision_analyzer_server.py

Использование (curl):
    # одно изображение, бэкенд и модель по умолчанию
    curl -F "images=@photo.jpg" http://localhost:6769/analyze

    # батч
    curl -F "images=@1.jpg" -F "images=@2.jpg" http://localhost:6769/analyze

    # явный бэкенд/модель на один запрос
    curl -F "backend=ollama" -F "model=qwen-analytical:latest" \
         -F "images=@photo.jpg" http://localhost:6769/analyze

    # то же самое через query-параметры
    curl -F "images=@photo.jpg" \
         "http://localhost:6769/analyze?backend=vllm&model=google/gemma-4-31B-it"
"""

import asyncio
import base64
import json
import logging

import aiohttp
from aiohttp import web

try:
    from vision_analyzer_prompt import SYSTEM_PROMPT
except ImportError:
    logging.warning(
        "vision_analyzer_server: AI/prompts/vision_analyzer_prompt.py не найден, "
    )


# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

BACKEND = "vllm"  # "vllm" или "ollama" — бэкенд по умолчанию, можно переопределить в запросе

OLLAMA_HOST = "http://127.0.0.1:11434"
VLLM_URL = "http://localhost:8000/v1"

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 6769

REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=180)
DISCOVERY_TIMEOUT = aiohttp.ClientTimeout(total=10)

RISK_EMOJI = {
    "low": "🟢",
    "medium": "🟡",
    "high": "🔴",
}

RISK_LABEL_RU = {
    "low": "низкий",
    "medium": "средний",
    "high": "высокий",
}

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
            raise ValueError(f"Неизвестный backend: {backend!r} (ожидается 'vllm' или 'ollama')")

    if not models:
        raise RuntimeError(f"Бэкенд {backend!r} не вернул ни одной модели.")

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
        return {"ok": False, "endpoint": endpoint, "error": "недоступен (connection refused)"}
    except asyncio.TimeoutError:
        return {"ok": False, "endpoint": endpoint, "error": "таймаут"}
    except Exception as e:
        return {"ok": False, "endpoint": endpoint, "error": str(e)}


# ---------------------------------------------------------------------------
# Анализ изображения
# ---------------------------------------------------------------------------

_USER_PROMPT = "Проанализируй это изображение и верни JSON по заданной схеме."


async def _analyze_ollama(image_b64: str, model: str) -> str:
    payload = {
        "model": model,
        "stream": False,
        "format": "json",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _USER_PROMPT, "images": [image_b64]},
        ],
        "options": {"temperature": 0.1},
    }

    async with aiohttp.ClientSession(timeout=REQUEST_TIMEOUT) as session:
        async with session.post(f"{OLLAMA_HOST}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            data = await resp.json()

    return data.get("message", {}).get("content", "").strip()


async def _analyze_vllm(image_b64: str, image_mime: str, model: str) -> str:
    base_payload = {
        "model": model,
        "temperature": 0.1,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
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
) -> dict:
    """Отправляет изображение в vision-модель выбранного бэкенда и возвращает
    разобранный отчёт.

    Если backend не был явно запрошен вызывающим кодом (allow_fallback=True)
    и оказался недоступен по подключению (connection refused), один раз
    пробуем единственный альтернативный бэкенд с автоопределением его модели.
    Если backend был явно указан в запросе — фолбэка не будет, ошибка
    подключения пробрасывается как есть.

    При невалидном JSON от модели возвращает словарь с ключом "_raw",
    содержащим исходный текст ответа — чтобы вызывающий код мог показать
    хоть что-то вместо ошибки.
    """
    try:
        resolved_model = model or await _discover_model(backend)

        if backend == "vllm":
            content = await _analyze_vllm(image_b64, image_mime, resolved_model)
        elif backend == "ollama":
            content = await _analyze_ollama(image_b64, resolved_model)
        else:
            raise ValueError(f"Неизвестный backend: {backend!r} (ожидается 'vllm' или 'ollama')")
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
            backend=fallback_backend, model=None, allow_fallback=False,
        )

    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError):
        logging.warning("vision_analyzer: модель (%s/%s) вернула невалидный JSON", backend, resolved_model)
        return {"_raw": content or "⚠️ Модель вернула пустой ответ."}


def _format_report(report: dict, source_name: str = "") -> str:
    header = f"=== {source_name} ===" if source_name else "=== отчёт ==="

    if "_raw" in report:
        return f"{header}\n⚠️ Модель ответила не по формату, вот сырой ответ:\n\n{report['_raw']}"

    risk_level = report.get("risk_level", "low")
    emoji = RISK_EMOJI.get(risk_level, "⚪️")
    risk_label = RISK_LABEL_RU.get(risk_level, risk_level)
    needs_review = report.get("needs_human_review", False)

    lines = [
        header,
        f"{emoji} Уровень риска: {risk_label}",
        "🧑 Нужна проверка человеком" if needs_review else "✅ Проверка человеком не требуется",
        "",
        "📝 Описание:",
        report.get("description", "—"),
    ]

    text_on_image = report.get("text_on_image", "")
    if text_on_image:
        lines += ["", "🔤 Текст на изображении:", text_on_image]

    signals = report.get("signals", [])
    if signals:
        lines += ["", "⚠️ Сигналы:"]
        for s in signals:
            lines.append(f"{s.get('id', '?')}: {s.get('category', '—')} — {s.get('detail', '')}")

    rationale = report.get("rationale", "")
    if rationale:
        lines += ["", "💡 Обоснование:", rationale]

    recommendation = report.get("recommendation", "")
    if recommendation:
        lines += ["", "✅ Рекомендация:", recommendation]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# HTTP-слой
# ---------------------------------------------------------------------------

async def handle_index(request: web.Request) -> web.Response:
    return web.Response(
        text=(
            "vision_analyzer_server работает.\n\n"
            f"Бэкенд по умолчанию: {BACKEND}\n\n"
            "GET /health — статус vLLM и Ollama (доступность + автоопределённая модель)\n\n"
            "POST /analyze — multipart/form-data:\n"
            "  images   — одно или несколько изображений (обязательно)\n"
            "  backend  — 'vllm' | 'ollama' (необязательно, переопределяет BACKEND)\n"
            "  model    — имя/id модели (необязательно, иначе автоопределение)\n\n"
            "Бэкенд/модель также можно передать query-параметрами: "
            "?backend=vllm&model=google/gemma-4-31B-it\n\n"
            "Пример:\n"
            "curl -F \"images=@photo.jpg\" http://localhost:6769/analyze\n"
        ),
        content_type="text/plain",
    )


async def handle_health(request: web.Request) -> web.Response:
    vllm_status, ollama_status = await asyncio.gather(
        _ping_backend("vllm"),
        _ping_backend("ollama"),
    )

    overall_ok = vllm_status["ok"] or ollama_status["ok"]
    default_ok = {"vllm": vllm_status, "ollama": ollama_status}[BACKEND]["ok"]

    return web.json_response(
        {
            "ok": overall_ok,
            "default_backend": BACKEND,
            "default_backend_ok": default_ok,
            "backends": {
                "vllm": vllm_status,
                "ollama": ollama_status,
            },
        },
        dumps=lambda obj: json.dumps(obj, ensure_ascii=False, indent=2),
        status=200 if overall_ok else 503,
    )


async def handle_analyze(request: web.Request) -> web.Response:
    if not request.content_type.startswith("multipart/"):
        return web.json_response(
            {"error": "Ожидается multipart/form-data с полем 'images'."},
            status=400,
        )

    # Дефолты из query-параметров, могут быть переопределены полями формы ниже.
    override_backend = request.query.get("backend")
    override_model = request.query.get("model")

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
        if part.name not in ("images", "image"):
            continue
        if not (part.headers.get("Content-Type", "").startswith("image/")):
            logging.warning("Пропускаю не-изображение: %s", part.filename)
            continue

        data = await part.read(decode=True)
        image_b64 = base64.b64encode(data).decode("utf-8")
        image_mime = part.headers.get("Content-Type", "image/jpeg")
        source_name = part.filename or f"image_{len(names) + 1}"
        names.append(source_name)
        tasks.append((image_b64, image_mime))

    if not tasks:
        return web.json_response(
            {"error": "Не найдено ни одного изображения в поле 'images'."},
            status=400,
        )

    backend = (override_backend or BACKEND).strip().lower()
    if backend not in ("vllm", "ollama"):
        return web.json_response(
            {"error": f"Неизвестный backend: {backend!r} (ожидается 'vllm' или 'ollama')"},
            status=400,
        )

    backend_was_explicit = bool(override_backend)

    logging.info(
        "Получено изображений в запросе: %d (%s) | backend=%s model=%s",
        len(tasks), ", ".join(names), backend, override_model or "auto",
    )

    try:
        reports = await asyncio.gather(*[
            _analyze_image(
                img_b64, img_mime,
                backend=backend, model=override_model,
                allow_fallback=not backend_was_explicit,
            )
            for img_b64, img_mime in tasks
        ])
    except aiohttp.ClientConnectorError:
        endpoint = VLLM_URL if backend == "vllm" else OLLAMA_HOST
        logging.error("Не удалось подключиться к бэкенду %s (%s)", backend, endpoint)
        return web.json_response(
            {"error": f"Бэкенд {backend!r} недоступен по адресу {endpoint}. Проверьте, что сервис запущен."},
            status=502,
        )
    except (ValueError, RuntimeError) as e:
        return web.json_response({"error": str(e)}, status=400)
    except Exception as e:
        logging.exception("Ошибка при анализе изображений: %s", e)
        return web.json_response({"error": "Не удалось проанализировать изображение(я)."}, status=500)

    results = []
    for name, report in zip(names, reports):
        formatted = _format_report(report, source_name=name)
        print(formatted)
        print()
        results.append({"file": name, "report": report})

    return web.json_response(
        {"count": len(results), "backend": backend, "results": results},
        dumps=lambda obj: json.dumps(obj, ensure_ascii=False, indent=2),
    )


def build_app() -> web.Application:
    app = web.Application(client_max_size=64 * 1024 * 1024)  # до 64 МБ на запрос
    app.router.add_get("/", handle_index)
    app.router.add_get("/health", handle_health)
    app.router.add_post("/analyze", handle_analyze)
    return app


if __name__ == "__main__":
    logging.info(
        "Запускаю vision_analyzer_server на http://%s:%d (backend по умолчанию: %s)",
        SERVER_HOST, SERVER_PORT, BACKEND,
    )
    web.run_app(build_app(), host=SERVER_HOST, port=SERVER_PORT)