"""
backends.py — всё, что говорит с моделью напрямую:
  - автоопределение и сканирование моделей (Ollama/vLLM)
  - обнаружение реального контекстного окна vLLM
  - health-пинг бэкенда
  - сборка истории диалога в формат конкретного бэкенда + защитная
    обрезка истории под контекст vLLM
  - низкоуровневая отправка ОДНОГО chat-запроса с картинкой (Ollama
    /api/chat, vLLM /v1/chat/completions) — _analyze_ollama/_analyze_vllm
    ничего не знают про схему ответа, просто шлют system+user+картинку
    и возвращают сырой текст; какой именно system/user-промпт подставить
    решает вызывающая сторона (см. prompt.py)
  - двухпроходный анализ одного изображения (_analyze_image):
      pass 1 (_select_categories) — дешёвая предклассификация, до
        _CLASSIFY_MAX_ATTEMPTS попыток; если модель так и не вернула
        валидный список категорий, сигнализирует об этом (None), и
        pass 2 уходит в compact-fallback (все категории разом, в
        сокращённом виде — см. prompt.get_system_prompt(compact=True))
      pass 2 (полный анализ) — то же, что раньше делал одиночный вызов,
        но с промптом, отфильтрованным по результату pass 1
  - общий wrapper с фолбэком между бэкендами (vllm <-> ollama)

Ничего из этого не хранит состояние диалога — история приходит целиком
от вызывающей стороны (см. server.py: handle_analyze) на каждый запрос.
История участвует только во втором проходе (полный анализ); первый
проход (классификация) всегда стателесс — ему не нужен контекст прошлых
сообщений, только текущая картинка и caption.
"""

import asyncio
import json
import logging
import re
import aiohttp
import uuid
import categories
import config


# ---------------------------------------------------------------------------
# Автоопределение модели
# ---------------------------------------------------------------------------

_model_cache: dict[str, str] = {}


async def _list_models(backend: str) -> list[str]:
    """Возвращает список всех моделей, которые сейчас отдаёт бэкенд.

    Не трогает _model_cache — это просто "сырое" сканирование, использует
    его и _discover_model (для авто-выбора первой модели), и /models
    (чтобы показать пользователю всё, что доступно).
    """
    async with aiohttp.ClientSession(timeout=config.DISCOVERY_TIMEOUT) as session:
        if backend == "vllm":
            async with session.get(f"{config.VLLM_URL}/models") as resp:
                resp.raise_for_status()
                data = await resp.json()
            return [m["id"] for m in data.get("data", [])]
        elif backend == "ollama":
            async with session.get(f"{config.OLLAMA_HOST}/api/tags") as resp:
                resp.raise_for_status()
                data = await resp.json()
            return [m["name"] for m in data.get("models", [])]
        else:
            raise ValueError(config._t("error.unknown_backend", backend=backend))


async def _discover_model(backend: str) -> str:
    """Возвращает первую доступную модель у бэкенда, кэширует результат.

    Явно заданная в запросе модель (параметр model=...) этот кэш не трогает
    и не использует — discovery нужен только когда модель не указана.
    """
    if backend in _model_cache:
        return _model_cache[backend]

    models = await _list_models(backend)

    if not models:
        raise RuntimeError(config._t("error.no_models_returned", backend=backend))

    _model_cache[backend] = models[0]
    logging.info("Автоопределена модель для backend=%s: %s", backend, models[0])
    return models[0]


_vllm_context_cache: dict[str, int] = {}


async def _get_vllm_context_window(model: str | None = None) -> int | None:
    """Спрашивает у vLLM реальный размер контекстного окна модели.

    Некоторые сборки vLLM отдают 'max_model_len' прямо в GET /v1/models
    (в отличие от Ollama, где это per-request параметр, у vLLM это то,
    с чем сервер был запущен — --max-model-len). Кэшируется по имени
    модели; кэш сбрасывается вместе с _model_cache в /config, если
    поменялся vllm_url.

    Возвращает None, если бэкенд недоступен, модель не нашлась в ответе,
    или конкретная сборка vLLM просто не отдаёт это поле — в таком случае
    вызывающий код должен считать контекст неизвестным и не обрезать
    историю "вслепую".
    """
    try:
        resolved_model = model or await _discover_model("vllm")
        if resolved_model in _vllm_context_cache:
            return _vllm_context_cache[resolved_model]

        async with aiohttp.ClientSession(timeout=config.DISCOVERY_TIMEOUT) as session:
            async with session.get(f"{config.VLLM_URL}/models") as resp:
                resp.raise_for_status()
                data = await resp.json()

        for m in data.get("data", []):
            if m.get("id") == resolved_model:
                max_len = m.get("max_model_len")
                if isinstance(max_len, int):
                    _vllm_context_cache[resolved_model] = max_len
                    return max_len
        return None
    except Exception as e:
        logging.debug("vLLM: не удалось узнать max_model_len (%s)", e)
        return None


async def _ping_backend(backend: str) -> dict:
    """Проверяет доступность бэкенда и возвращает статус для /health.

    Никогда не бросает исключение наружу — любая ошибка превращается
    в {"ok": False, "error": ...}, чтобы падение одного бэкенда не мешало
    проверить остальные.
    """
    endpoint = config.VLLM_URL if backend == "vllm" else config.OLLAMA_HOST
    try:
        model = await _discover_model(backend)
        return {"ok": True, "endpoint": endpoint, "model": model}
    except aiohttp.ClientConnectorError:
        return {"ok": False, "endpoint": endpoint, "error": config._t("error.backend_conn_refused")}
    except asyncio.TimeoutError:
        return {"ok": False, "endpoint": endpoint, "error": config._t("error.backend_timeout")}
    except Exception as e:
        return {"ok": False, "endpoint": endpoint, "error": str(e)}


# ---------------------------------------------------------------------------
# История диалога
# ---------------------------------------------------------------------------

def _strip_data_url(img: str) -> str:
    """Убирает 'data:...;base64,' префикс, если есть — Ollama ждёт чистый base64."""
    if img.startswith("data:") and ";base64," in img:
        return img.split(";base64,", 1)[1]
    return img


def _ensure_data_url(img: str, default_mime: str = "image/png") -> str:
    """Добавляет 'data:...;base64,' префикс, если его нет — нужен для vLLM image_url."""
    if img.startswith("data:"):
        return img
    return f"data:{default_mime};base64,{img}"


def _history_to_ollama_messages(history: list) -> list[dict]:
    """История прошлых сообщений диалога → формат сообщений Ollama.

    Сервер сам НИГДЕ не хранит историю — она целиком приходит в каждом
    запросе от клиента (бота/UI) и здесь просто конвертируется в нужный
    для конкретного бэкенда формат сообщений.
    """
    messages = []
    for turn in history:
        entry = {"role": turn["role"], "content": turn.get("content", "")}
        images = turn.get("images")
        if images:
            entry["images"] = [_strip_data_url(img) for img in images]
        messages.append(entry)
    return messages


def _history_to_vllm_messages(history: list) -> list[dict]:
    """История прошлых сообщений диалога → формат сообщений vLLM (OpenAI-style)."""
    messages = []
    for turn in history:
        blocks = []
        text = turn.get("content")
        if text:
            blocks.append({"type": "text", "text": text})
        for img in turn.get("images") or []:
            blocks.append({"type": "image_url", "image_url": {"url": _ensure_data_url(img)}})
        messages.append({"role": turn["role"], "content": blocks or ""})
    return messages


def _parse_history_json(raw) -> list:
    """Парсит и валидирует 'history' — список прошлых сообщений диалога.

    Сервер сам историю нигде не хранит — она целиком приходит от клиента
    (бота/UI) в каждом запросе. Формат бэкенд-агностичный:

        [
          {"role": "user", "content": "...", "images": ["data:image/png;base64,..."]},
          {"role": "assistant", "content": "..."}
        ]

    'content' и 'images' необязательны, но должны быть строкой/списком,
    если присутствуют. 'images' — data URL (или просто base64 — тоже
    примется, см. _strip_data_url/_ensure_data_url).

    raw может быть уже списком (если пришло в JSON-теле запроса) либо
    JSON-строкой (если пришло через query-параметр или multipart-поле).
    Пустое/отсутствующее значение — просто "истории нет".
    """
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = raw.strip()
        if not raw:
            return []
        raw = json.loads(raw)  # может бросить json.JSONDecodeError (это ValueError)

    if not isinstance(raw, list):
        raise ValueError("history must be a list")

    for turn in raw:
        if not isinstance(turn, dict) or turn.get("role") not in ("user", "assistant"):
            raise ValueError("each history item needs role: 'user' or 'assistant'")
        if "content" in turn and not isinstance(turn["content"], str):
            raise ValueError("history 'content' must be a string")
        if "images" in turn and not isinstance(turn["images"], list):
            raise ValueError("history 'images' must be a list")

    return raw


def _parse_categories_json(raw) -> list | None:
    """Парсит 'categories' — разовые категории для этого вызова /analyze
    (см. categories.build_overlay). Сам список ничего не валидирует по
    содержимому (имена/summary/full/compact) — этим занимается
    build_overlay, поднимая categories.CategoryError.

    raw может быть:
      - None / [] — разовых категорий нет, сервер работает с дефолтами;
      - списком уже готовых объектов-категорий — пришёл из JSON-тела
        запроса ({"categories": [{...}, {...}]});
      - списком JSON-строк — по одной на каждый multipart-файл или
        повторяющийся query-параметр 'categories' (см.
        analyze._parse_multipart_body/_query_overrides). Каждая строка
        разбирается отдельно: если внутри один объект — категория
        добавляется как есть; если внутри массив — разворачивается
        (клиент может прислать как отдельный файл на категорию, так и
        один файл сразу со всеми).

    Смешивать строки и уже готовые объекты в одном списке тоже можно —
    их не различают заранее, каждый элемент разбирается по своему типу.
    """
    if not raw:
        return None
    if not isinstance(raw, list):
        raise ValueError("categories must be a list")

    result: list = []
    for item in raw:
        if isinstance(item, str):
            item = item.strip()
            if not item:
                continue
            try:
                parsed = json.loads(item)
            except json.JSONDecodeError as e:
                raise ValueError(f"invalid categories JSON: {e}") from e
        else:
            parsed = item

        if isinstance(parsed, list):
            result.extend(parsed)
        elif isinstance(parsed, dict):
            result.append(parsed)
        else:
            raise ValueError("each categories item must be an object or a list of objects")

    return result or None


# Грубая оценка размера токенов для истории, отправляемой в vLLM — точного
# токенайзера конкретной модели у нас тут нет, поэтому это защитный запас,
# а не честный расчёт. Используется только для решения "обрезать ли
# историю", когда реальный max_model_len удалось узнать через
# _get_vllm_context_window; сама vLLM всё равно провалидирует запрос и
# кинет ошибку, если промпт всё же не влез.
_VLLM_EST_CHARS_PER_TOKEN = 4
_VLLM_EST_TOKENS_PER_IMAGE = 1500
_VLLM_CONTEXT_SAFETY_MARGIN = 0.9  # оставляем запас под системный промпт/ответ модели


def _estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // _VLLM_EST_CHARS_PER_TOKEN)


def _truncate_history_for_vllm(
    history: list, system_prompt: str, current_text: str, max_model_len: int,
) -> list:
    """Отбрасывает старые сообщения истории, если оценочно не влезаем
    в контекст vLLM. Идёт с конца истории (свежие сообщения важнее),
    оставляет максимум, что влезает в safety-margin от max_model_len.
    """
    budget = int(max_model_len * _VLLM_CONTEXT_SAFETY_MARGIN)
    fixed_tokens = (
        _estimate_tokens(system_prompt)
        + _estimate_tokens(current_text)
        + _VLLM_EST_TOKENS_PER_IMAGE  # текущее изображение
    )

    def turn_tokens(turn: dict) -> int:
        return _estimate_tokens(turn.get("content")) + _VLLM_EST_TOKENS_PER_IMAGE * len(turn.get("images") or [])

    kept = []
    total = fixed_tokens
    for turn in reversed(history):
        t = turn_tokens(turn)
        if total + t > budget:
            break
        kept.insert(0, turn)
        total += t

    dropped = len(history) - len(kept)
    if dropped:
        logging.warning(
            "vLLM: history (%d сообщений) оценочно не влезает в контекст "
            "max_model_len=%d — отброшено %d старых сообщений, оставлено %d "
            "(оценочно ~%d/%d токенов, safety_margin=%.0f%%)",
            len(history), max_model_len, dropped, len(kept),
            total, max_model_len, _VLLM_CONTEXT_SAFETY_MARGIN * 100,
        )

    return kept


# ---------------------------------------------------------------------------
# Низкоуровневая отправка одного chat-запроса с картинкой.
#
# Ничего не знают о том, что за system_prompt/user_prompt им передали —
# это может быть промпт первого (классифицирующего) прохода или второго
# (полного анализа): решает вызывающая сторона (_select_categories /
# _analyze_image), собирая текст через prompt.py. Это единственный слой,
# который реально говорит с бэкендом, поэтому оба прохода идут через
# него, а не дублируют HTTP/streaming-логику.
# ---------------------------------------------------------------------------

async def _analyze_ollama(
    image_b64: str, model: str, system_prompt: str, user_prompt: str,
    history: list | None = None,
) -> str:
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(_history_to_ollama_messages(history or []))
    messages.append({"role": "user", "content": user_prompt, "images": [image_b64]})

    options = {
        "temperature": config.SAMPLING_DEFAULTS["temperature"],
        "top_p": config.SAMPLING_DEFAULTS["top_p"],
        "top_k": config.SAMPLING_DEFAULTS["top_k"],
        "seed": config.SAMPLING_DEFAULTS["seed"],
    }
    if config.SAMPLING_DEFAULTS["num_ctx"] is not None:
        options["num_ctx"] = config.SAMPLING_DEFAULTS["num_ctx"]
    if config.SAMPLING_DEFAULTS["num_predict"] is not None:
        options["num_predict"] = config.SAMPLING_DEFAULTS["num_predict"]

    # stream=True — чтобы видеть 'thinking' модели в реальном времени в
    # консоли сервера, а не ждать молча всю генерацию (может занимать
    # много минут при включённом think). ВНИМАНИЕ: если /analyze гонит
    # несколько картинок параллельно (asyncio.gather), вывод нескольких
    # запросов будет перемежаться в одной консоли — тег [xxxxxx] перед
    # каждым куском (первые 6 символов image_b64) нужен, чтобы отличить,
    # какой поток что печатает.
    payload = {
        "model": model,
        "stream": True,
        "format": "json",
        "messages": messages,
        "options": options,
        "think": config.SAMPLING_DEFAULTS["think"],
    }

    tag = uuid.uuid4().hex[:6]
    content_parts: list[str] = []
    thinking_open = False
    final: dict = {}

    async with aiohttp.ClientSession(timeout=config.REQUEST_TIMEOUT) as session:
        async with session.post(f"{config.OLLAMA_HOST}/api/chat", json=payload) as resp:
            resp.raise_for_status()
            async for raw_line in resp.content:
                line = raw_line.strip()
                if not line:
                    continue
                chunk = json.loads(line)

                msg = chunk.get("message", {})
                thinking = msg.get("thinking")
                if thinking:
                    if not thinking_open:
                        print(f"\n[{tag}] --- think ---", flush=True)
                        thinking_open = True
                    print(thinking, end="", flush=True)

                piece = msg.get("content")
                if piece:
                    content_parts.append(piece)

                if chunk.get("done"):
                    final = chunk

    if thinking_open:
        print(f"\n[{tag}] --- /think ---", flush=True)

    content = "".join(content_parts).strip()
    logging.info(
        "Ollama: prompt_tokens=%s gen_tokens=%s done_reason=%s content_chars=%d load=%.1fs total=%.1fs",
        final.get("prompt_eval_count"), final.get("eval_count"), final.get("done_reason"),
        len(content), final.get("load_duration", 0) / 1e9, final.get("total_duration", 0) / 1e9,
    )
    return content


async def _analyze_vllm(
    image_b64: str, image_mime: str, model: str, system_prompt: str, user_prompt: str,
    history: list | None = None,
) -> str:
    history = history or []

    # Если у vLLM удалось узнать реальный max_model_len — используем его,
    # чтобы не отправлять заведомо не влезающий промпт. Если не удалось
    # (сборка не отдаёт max_model_len, бэкенд недоступен и т.п.) — не
    # трогаем историю вслепую, просто отправляем как есть; vLLM сама
    # вернёт ошибку, если промпт не влезет.
    # Оценка бюджета строится на РЕАЛЬНОМ system_prompt этого вызова
    # (он может быть промптом первого прохода, или второго с отфильтрованными
    # /compact-категориями) — не на некотором обобщённом "полном" промпте,
    # это важно для точности оценки после разбиения на два прохода.
    max_model_len = await _get_vllm_context_window(model)
    if max_model_len:
        history = _truncate_history_for_vllm(
            history, system_prompt, user_prompt, max_model_len,
        )

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(_history_to_vllm_messages(history))
    messages.append({
        "role": "user",
        "content": [
            {"type": "text", "text": user_prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:{image_mime};base64,{image_b64}"},
            },
        ],
    })

    base_payload = {
        "model": model,
        "temperature": config.SAMPLING_DEFAULTS["temperature"],
        "top_p": config.SAMPLING_DEFAULTS["top_p"],
        "seed": config.SAMPLING_DEFAULTS["seed"],
        "messages": messages,
    }

    async with aiohttp.ClientSession(timeout=config.REQUEST_TIMEOUT) as session:
        # Пытаемся получить строгий JSON через response_format (guided decoding).
        payload = {**base_payload, "response_format": {"type": "json_object"}}
        async with session.post(f"{config.VLLM_URL}/chat/completions", json=payload) as resp:
            if resp.status == 400:
                # Некоторые сборки vLLM без guided-decoding backend отвергают
                # response_format — повторяем запрос без него.
                logging.warning("vLLM отклонил response_format, повторяю запрос без него")
                async with session.post(f"{config.VLLM_URL}/chat/completions", json=base_payload) as resp2:
                    resp2.raise_for_status()
                    data = await resp2.json()
            else:
                resp.raise_for_status()
                data = await resp.json()

    return data["choices"][0]["message"]["content"].strip()


_RU_TO_EN = {"низкий": "low", "средний": "medium", "высокий": "high"}
_ORDER = {"low": 0, "medium": 1, "high": 2}


def _finalize_report(report, lang: str | None = None):
    if not isinstance(report, dict) or "_raw" in report:
        return report

    raw_level = str(report.get("risk_level", "")).strip().lower()
    level = _RU_TO_EN.get(raw_level, raw_level)
    if level not in _ORDER:
        level = "medium"

    for s in report.get("signals") or []:
        if not isinstance(s, dict) or s.get("category") != "weapons_and_dangerous_objects":
            continue
        codes = set(re.findall(r"W[1-6]", s.get("detail", "")))
        if not codes & {"W1", "W4", "W5", "W6"}:
            continue

        floor = "high" if codes & {"W2", "W3"} else "medium"
        if _ORDER[floor] > _ORDER[level]:
            level = floor

        rec = config._t("rec.verify_weapon", lang=lang)
        if not rec.startswith("???"):
            report["recommendation"] = rec

        note = config._t("rationale.authenticity_unconfirmed", lang=lang)
        rationale = report.get("rationale", "")
        if not note.startswith("???") and note.split()[0].lower() not in rationale.lower():
            report["rationale"] = f"{rationale.rstrip()} {note}".strip()

    report["risk_level"] = level
    if report.get("signals"):
        report["needs_human_review"] = True
    return report


# ---------------------------------------------------------------------------
# Проход 1: предклассификация (какие категории вообще имеет смысл
# проверять полными правилами во втором проходе).
# ---------------------------------------------------------------------------

_CLASSIFY_MAX_ATTEMPTS = 3


def _parse_candidate_categories(
    content: str, overlay: "categories.CategoryOverlay | None" = None,
) -> list[str] | None:
    """Разбирает ответ первого (классифицирующего) вызова.

    Возвращает список валидных имён категорий (может быть пустым списком
    — это легитимный ответ "ничего из списка не подходит"), либо None,
    если ответ пустой или не разбирается как объект с массивом
    candidate_categories — тогда вызывающая сторона (_select_categories)
    должна повторить попытку или в итоге уйти в compact-fallback.

    Имена категорий, которых нет в реестре (см. categories.py), а также
    в overlay этого запроса, если он есть, — галлюцинация модели —
    тихо отбрасываются с предупреждением в лог, это не делает весь
    ответ невалидным.
    """
    if not content or not content.strip():
        return None
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None

    raw = data.get("candidate_categories")
    if not isinstance(raw, list):
        return None

    valid = [name for name in raw if isinstance(name, str) and categories.is_valid(name, overlay)]
    unknown = [name for name in raw if name not in valid]
    if unknown:
        logging.warning("classify: модель предложила неизвестные категории, отброшены: %s", unknown)
    return valid


async def _select_categories(
    image_b64: str,
    image_mime: str,
    backend: str,
    model: str,
    caption: str | None,
    overlay: "categories.CategoryOverlay | None" = None,
) -> list[str] | None:
    """Первый вызов: описывает изображение и предлагает шорт-лист категорий
    для второго, полного анализа. До _CLASSIFY_MAX_ATTEMPTS попыток, если
    модель вернула пустой или не разбирающийся по схеме ответ (запрос
    повторяется целиком, включая саму картинку — первая попытка могла
    просто "сорваться").

    Возвращает:
      - список имён категорий (может быть пустым — легитимное "ничего
        не подходит") при успешном разборе, с первой попытки или позже;
      - None, если все попытки исчерпаны без валидного ответа — тогда
        второй проход уходит в compact-fallback: все категории разом,
        в сокращённом виде (см. prompt.get_system_prompt(compact=True)).

    Первый проход всегда стателесс — без истории диалога, ему нужна
    только текущая картинка и caption, не прошлые сообщения. lang сюда
    не приходит намеренно: язык вывода классифицирующего прохода никуда
    дальше не идёт (см. prompt.get_classify_system_prompt), поэтому его
    незачем даже спрашивать у вызывающей стороны.

    overlay — разовые категории этого запроса (см. categories.build_overlay),
    если клиент передал свои в /analyze; включаются в шорт-лист кандидатов
    наравне с дефолтными.
    """
    system_prompt = config.prompt.get_classify_system_prompt(overlay)
    user_prompt = config.prompt.get_classify_user_prompt(caption)

    for attempt in range(1, _CLASSIFY_MAX_ATTEMPTS + 1):
        if backend == "vllm":
            content = await _analyze_vllm(image_b64, image_mime, model, system_prompt, user_prompt)
        else:
            content = await _analyze_ollama(image_b64, model, system_prompt, user_prompt)

        candidates = _parse_candidate_categories(content, overlay)
        if candidates is not None:
            if attempt > 1:
                logging.info(
                    "classify: валидный ответ получен с попытки %d/%d", attempt, _CLASSIFY_MAX_ATTEMPTS,
                )
            logging.info("classify: категории-кандидаты (backend=%s model=%s): %s", backend, model, candidates or "(нет)")
            return candidates

        logging.warning(
            "classify: попытка %d/%d — пустой или некорректный (не по схеме) ответ модели (backend=%s model=%s)",
            attempt, _CLASSIFY_MAX_ATTEMPTS, backend, model,
        )

    logging.warning(
        "classify: все %d попытки исчерпаны без валидного ответа — второй проход уходит в "
        "compact-fallback (все категории разом, в сокращённом виде, без classify-фильтрации)",
        _CLASSIFY_MAX_ATTEMPTS,
    )
    return None


# ---------------------------------------------------------------------------
# Проход 2 + общий вход: полный анализ отфильтрованными правилами.
# ---------------------------------------------------------------------------

async def _analyze_image(
    image_b64: str,
    image_mime: str = "image/jpeg",
    backend: str = config.BACKEND,
    model: str | None = None,
    allow_fallback: bool = True,
    lang: str | None = None,
    history: list | None = None,
    caption: str | None = None,
    overlay: "categories.CategoryOverlay | None" = None,
) -> tuple[dict, str]:
    try:
        resolved_model = model or await _discover_model(backend)
        resolved_lang = lang or config._current_lang()

        selected = await _select_categories(
            image_b64, image_mime, backend, resolved_model, caption, overlay,
        )
        used_fallback = selected is None  # None = проход 1 исчерпал попытки

        system_prompt = config.prompt.get_system_prompt(
            resolved_lang, categories=selected, compact=used_fallback, overlay=overlay,
        )
        user_prompt = config.prompt.get_user_prompt(resolved_lang, caption)

        if backend == "vllm":
            content = await _analyze_vllm(
                image_b64, image_mime, resolved_model, system_prompt, user_prompt, history=history,
            )
        elif backend == "ollama":
            content = await _analyze_ollama(
                image_b64, resolved_model, system_prompt, user_prompt, history=history,
            )
        else:
            raise ValueError(config._t("error.unknown_backend", backend=backend, lang=lang))
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
            backend=fallback_backend, model=None, allow_fallback=False, lang=lang, history=history,
            caption=caption, overlay=overlay,
        )

    try:
        report = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        logging.warning("vision_analyzer: модель (%s/%s) вернула невалидный JSON", backend, resolved_model)
        return {"_raw": content or config._t("model.empty_response", lang=lang)}, backend

    return _finalize_report(report, lang), backend