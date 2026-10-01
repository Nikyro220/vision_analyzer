"""
backends.py — конвейер /analyze поверх провайдеров (см. providers/):
риск-анализ по строгой JSON-схеме, без истории — см. chat_backends.py,
если нужен свободный диалог с историей.

Сам разговор с моделью (HTTP к Ollama/vLLM/Gemini, автоопределение модели,
health-пинг, контекстное окно) живёт в providers/ — по одному модулю на
бэкенд, общий интерфейс providers.base.Provider. Здесь только то, что от
бэкенда не зависит:
  - двухпроходный анализ одного изображения (_analyze_image):
      pass 1 (_select_categories) — дешёвая предклассификация, до
        _CLASSIFY_MAX_ATTEMPTS попыток; если модель так и не вернула
        валидный список категорий, сигнализирует об этом (None), и
        pass 2 уходит в compact-fallback (все категории разом, в
        сокращённом виде — см. prompt.get_system_prompt(compact=True))
      pass 2 (полный анализ) — то же, что раньше делал одиночный вызов,
        но с промптом, отфильтрованным по результату pass 1
    какой именно system/user-промпт подставить, решает этот модуль
    (см. prompt.py), провайдер лишь шлёт system+user+картинку и
    возвращает сырой текст
  - фолбэк на другой бэкенд при недоступности (Provider.fallback)
  - разбор разовых категорий запроса (_parse_categories_json)
  - постобработка отчёта (_finalize_report)

Ничего из этого не хранит состояние диалога — каждый вызов /analyze
разовый, без контекста прошлых сообщений (история диалога — только у
/chat, см. chat_backends.py).
"""

import json
import logging
import re

import aiohttp

import categories
import config
import providers


# ---------------------------------------------------------------------------
# Разовые категории запроса
# ---------------------------------------------------------------------------

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
    provider = providers.get(backend)

    for attempt in range(1, _CLASSIFY_MAX_ATTEMPTS + 1):
        content = await provider.analyze(image_b64, image_mime, model, system_prompt, user_prompt)

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
    caption: str | None = None,
    overlay: "categories.CategoryOverlay | None" = None,
) -> tuple[dict, str]:
    try:
        provider = providers.get(backend)  # ValueError для неизвестного бэкенда
        resolved_model = model or await provider.discover_model()
        resolved_lang = lang or config._current_lang()

        selected = await _select_categories(
            image_b64, image_mime, backend, resolved_model, caption, overlay,
        )
        used_fallback = selected is None  # None = проход 1 исчерпал попытки

        system_prompt = config.prompt.get_system_prompt(
            resolved_lang, categories=selected, compact=used_fallback, overlay=overlay,
        )
        user_prompt = config.prompt.get_user_prompt(resolved_lang, caption)

        content = await provider.analyze(image_b64, image_mime, resolved_model, system_prompt, user_prompt)
    except aiohttp.ClientConnectorError:
        fallback_backend = providers.get(backend).fallback
        if not allow_fallback or not fallback_backend:
            raise
        logging.warning(
            "Бэкенд %r недоступен по подключению, пробую фолбэк на %r",
            backend, fallback_backend,
        )
        return await _analyze_image(
            image_b64, image_mime,
            backend=fallback_backend, model=None, allow_fallback=False, lang=lang,
            caption=caption, overlay=overlay,
        )

    try:
        report = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        logging.warning("vision_analyzer: модель (%s/%s) вернула невалидный JSON", backend, resolved_model)
        return {"_raw": content or config._t("model.empty_response", lang=lang)}, backend

    return _finalize_report(report, lang), backend
