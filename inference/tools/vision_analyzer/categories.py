"""
categories.py — реестр категорий сигналов для двухпроходного пайплайна
vision_analyzer (см. prompt.py: get_classify_system_prompt / get_system_prompt).

Категории лежат не в этом модуле, а на диске, в inference/categories/:
  categories/index.json                — {"order": [...], "summaries": {имя: summary}}
  categories/<имя_категории>.json       — {"full": ..., "compact": ..., "examples": {...}?}

Почему так разрезано (а не один файл на категорию целиком, и не один общий
файл на всё): summary — лёгкое поле, читается на КАЖДОМ первом
(классифицирующем) вызове через summaries_block(), и это единственное,
что вообще нужно первому вызову. full/compact/examples — тяжёлые поля,
нужны только во втором вызове, и то не для всех категорий разом, а
только для тех, что выбрал первый вызов (плюс compact — для всех разом,
но только в fallback). Держать order+summaries в одном index.json, а не
размазанными по 5 файлам категорий вместе с order.json как отдельным
шестым файлом, даёт две вещи:
  - reload() читает один маленький файл, чтобы получить полный список
    кандидатов для первого вызова (имя + summary + порядок) — не нужно
    открывать все 5 "тяжёлых" файлов ради одной строки из каждого;
  - будущий admin-эндпоинт, который правит порядок или формулировку
    summary (частая, лёгкая правка), трогает один маленький файл и не
    рискует что-то сломать в тяжёлых full/compact/examples остальных
    категорий; а правка правил конкретной категории (редкая, объёмная
    правка) остаётся изолированной в её собственном файле.
Такого эндпоинта пока нет (см. TODO ниже), но раскладка уже под него
рассчитана: он пишет/удаляет файлы в inference/categories/ и зовёт
reload().

Имя файла категории (без .json) == имя категории — само оно внутри
файла не хранится, чтобы при переименовании (mv файла + правка записи
в index.json) не пришлось отдельно поправлять содержимое.

Части одной категории:
  summary  — в index.json. Короткий абзац на английском: что должно
             быть видно на изображении, чтобы категория стала
             кандидатом. Используется ТОЛЬКО в первом
             (классифицирующем) вызове — recall важнее precision,
             поэтому формулировки намеренно нестрогие.
  full     — в categories/<имя>.json. Полный блок правил
             <signal_category>...</signal_category>. Используется во
             втором (полном) вызове для каждой категории, которую
             выбрал первый вызов.
  compact  — в categories/<имя>.json. Сокращённая версия тех же
             правил. Используется во втором вызове ТОЛЬКО как
             fallback — для всех категорий сразу, когда
             классифицирующему вызову нельзя было доверять (см.
             backends._select_categories).
  examples — в categories/<имя>.json, необязательное поле:
             {"en": "...", "ru": "..."} с разобранными примерами.
             Категория без примеров просто не имеет этого ключа —
             сборка <examples> её пропускает.

Admin API (см. categories_api.py, роуты собраны в server.py) — ТОЛЬКО
чтение, дефолты правятся вручную в inference/categories/*.json:
  GET  /categories            — list_categories(): {"order": [...]}
  GET  /categories/summaries  — get_summaries(): {имя: summary, ...}
  GET  /categories/<имя>      — get_category(имя): summary+full+compact+
                                 examples одной категории

Разовые категории (см. build_overlay ниже): вместо API, который бы
навсегда менял общий для всех реестр (риск: любой с доступом к серверу
мог свободно переопределить дефолты для всех последующих запросов),
клиент передаёт определение категории прямо в теле POST /analyze
(поле "categories"). build_overlay() строит из него локальный,
изолированный оверлей (order/summaries/registry) поверх текущих
дефолтов — ничего не пишет на диск, не вызывает reload(), не трогает
глобальный реестр ниже. Оверлей живёт ровно один HTTP-запрос и
передаётся явным параметром через backends.py/prompt.py — не как
общее состояние, — поэтому параллельные запросы разных клиентов с
разными "categories" никогда не видят категории друг друга.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections import namedtuple
from pathlib import Path

_DIR = Path(__file__).resolve().parent / "categories"
_lock = threading.Lock()

# Изолированный, разовый "вид" реестра — снимок order/summaries/registry,
# который build_overlay() строит поверх дефолтов для одного запроса (см.
# докстринг модуля). Передаётся явным параметром, а не хранится глобально.
CategoryOverlay = namedtuple("CategoryOverlay", ["order", "summaries", "registry"])

# Имя категории == имя файла на диске (см. докстринг выше) — поэтому
# ограничиваем его тем, что безопасно как имя файла и не позволяет выйти
# из inference/categories/ (никаких "..", "/", пробелов и т.п.).
_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")

_UNSET = object()  # сентинел: поле не передано в запросе (в отличие от None)

_order: list[str] = []
_summaries: dict[str, str] = {}
_registry: dict[str, dict] = {}


def _load(directory: Path) -> tuple[list[str], dict[str, str], dict[str, dict]]:
    index_path = directory / "index.json"
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)

    order = index["order"]
    summaries = index["summaries"]

    missing_summary = [name for name in order if name not in summaries or not summaries[name].strip()]
    if missing_summary:
        raise ValueError(f"categories/index.json: нет summary для {missing_summary}")

    registry: dict[str, dict] = {}
    for name in order:
        cat_path = directory / f"{name}.json"
        if not cat_path.exists():
            raise ValueError(f"categories/index.json ссылается на {name!r}, но файла {cat_path.name} нет")
        with open(cat_path, "r", encoding="utf-8") as f:
            cat = json.load(f)
        for field in ("full", "compact"):
            if not isinstance(cat.get(field), str) or not cat[field].strip():
                raise ValueError(f"categories/{name}.json: поле {field!r} пустое или отсутствует")
        registry[name] = cat

    # Файлы категорий, которые лежат в папке, но не упомянуты в index.json,
    # молча игнорируются (а не подхватываются в случайном порядке) — это
    # оставляет index.json единственным местом, которое решает, что вообще
    # активно, и позволяет держать в папке, например, черновик новой
    # категории, ещё не включённой в работу.
    known = set(registry) | {"index"}
    extra = sorted(p.stem for p in directory.glob("*.json") if p.stem not in known)
    if extra:
        logging.info(
            "categories: в папке есть файлы, не перечисленные в index.json (проигнорированы): %s",
            ", ".join(extra),
        )

    return order, summaries, registry


def reload(directory: Path = _DIR) -> None:
    """Перечитывает inference/categories/ с диска и атомарно подменяет реестр.

    Если что-то битое (index.json не парсится, ссылается на
    несуществующий файл категории, у категории нет summary/full/compact)
    — реестр НЕ трогается, ошибка летит наверх вызывающему коду
    (будущему admin-эндпоинту), а не роняет уже работающий сервер.
    """
    order, summaries, registry = _load(directory)
    with _lock:
        global _order, _summaries, _registry
        _order = order
        _summaries = summaries
        _registry = registry
    logging.info(
        "categories: загружено %d категорий из %s: %s",
        len(order), directory, ", ".join(order),
    )


reload()


def order(overlay: CategoryOverlay | None = None) -> list[str]:
    """Список имён категорий в каноническом порядке (порядок появления
    в промпте) — единственный источник порядка для обоих проходов.

    overlay — см. build_overlay(); None (по умолчанию) означает "только
    дефолты", как и раньше."""
    return list(overlay.order if overlay is not None else _order)


def is_valid(name: str, overlay: CategoryOverlay | None = None) -> bool:
    return name in (overlay.registry if overlay is not None else _registry)


def _selected_or_all(selected: list[str] | None, order_list: list[str]) -> list[str]:
    """Пересекает выбор с order_list, сохраняя канонический порядок.
    None означает "все категории" (используется в compact-fallback и
    как обратно совместимое поведение по умолчанию)."""
    if selected is None:
        return list(order_list)
    wanted = set(selected)
    return [name for name in order_list if name in wanted]


def summaries_block(overlay: CategoryOverlay | None = None) -> str:
    """<category name="...">summary</category> для КАЖДОЙ
    зарегистрированной категории — список кандидатов для первого
    (классифицирующего) вызова. Не фильтруется: фильтрация — это и
    есть то, что делает первый вызов.

    Собирается целиком из index.json (через _summaries), либо, если
    передан overlay, из его summaries — файлы отдельных категорий
    (full/compact/examples) для этого не нужны."""
    order_list = overlay.order if overlay is not None else _order
    summaries = overlay.summaries if overlay is not None else _summaries
    parts = [
        f'<category name="{name}">{summaries[name]}</category>'
        for name in order_list
    ]
    return "\n".join(parts)


def full_signals_block(selected: list[str] | None = None, overlay: CategoryOverlay | None = None) -> str:
    order_list = overlay.order if overlay is not None else _order
    registry = overlay.registry if overlay is not None else _registry
    names = _selected_or_all(selected, order_list)
    return "\n\n".join(registry[name]["full"] for name in names)


def compact_signals_block(selected: list[str] | None = None, overlay: CategoryOverlay | None = None) -> str:
    order_list = overlay.order if overlay is not None else _order
    registry = overlay.registry if overlay is not None else _registry
    names = _selected_or_all(selected, order_list)
    return "\n\n".join(registry[name]["compact"] for name in names)


_EXAMPLES_INTRO = {
    "en": (
        "Each example shows only the analysis fields; write all fields from output_format.\n"
        "The examples are in English; write your own free-text values in __OUTPUT_LANGUAGE__ "
        "(feature codes W1-W6 and category names stay as shown)."
    ),
    "ru": (
        "Each example shows only the analysis fields; write all fields from output_format.\n"
        "The text values in the examples are in Russian, like your own text values."
    ),
}


def examples_block(selected: list[str] | None, lang: str, overlay: CategoryOverlay | None = None) -> str:
    """Собирает сцены-примеры для выбранных категорий на нужном языке.

    Возвращает "" (пустую строку), если ни у одной выбранной категории
    нет примеров на этот язык — например, в compact-fallback, где
    examples намеренно не подгружаются вовсе, или когда первый вызов
    выбрал только категории без примеров. Вызывающая сторона
    (prompt.py) должна аккуратно убрать секцию <examples> целиком,
    а не оставлять пустую пару тегов.
    """
    order_list = overlay.order if overlay is not None else _order
    registry = overlay.registry if overlay is not None else _registry
    names = _selected_or_all(selected, order_list)
    scenes = [
        registry[name]["examples"][lang]
        for name in names
        if registry[name].get("examples") and lang in registry[name]["examples"]
    ]
    if not scenes:
        return ""
    intro = _EXAMPLES_INTRO.get(lang, _EXAMPLES_INTRO["en"])
    return "\n\n".join([intro, *scenes]) + "\n"


# ---------------------------------------------------------------------------
# Admin API — чтение (см. categories_api.py: GET-хендлеры)
# ---------------------------------------------------------------------------

class CategoryError(ValueError):
    """Ошибка валидации входных данных API (имя категории, отсутствующие
    обязательные поля, некорректный order и т.п.). Отдельный тип, чтобы
    categories_api.py мог поймать именно её и вернуть 400, не путая с
    программистской ошибкой (которая должна остаться 500)."""


def list_categories() -> dict:
    """{"order": [...]} — список имён в каноническом порядке. Ответ
    GET /categories."""
    return {"order": list(_order)}


def get_summaries() -> dict[str, str]:
    """summary каждой категории, отдельно от полного содержимого — ответ
    GET /categories/summaries."""
    return dict(_summaries)


def get_category(name: str) -> dict | None:
    """Полное содержимое одной категории: summary+full+compact+examples —
    ответ GET /categories/<имя>. None, если категории с таким именем нет."""
    if name not in _registry:
        return None
    cat = _registry[name]
    result = {
        "name": name,
        "summary": _summaries[name],
        "full": cat["full"],
        "compact": cat["compact"],
    }
    if cat.get("examples"):
        result["examples"] = cat["examples"]
    return result


# ---------------------------------------------------------------------------
# Разовые категории — build_overlay (см. докстринг модуля и analyze.py:
# handle_analyze). Ничего здесь не пишет на диск и не трогает
# _order/_summaries/_registry — только строит и возвращает локальные копии.
# ---------------------------------------------------------------------------

def _merge_examples(current: dict | None, incoming) -> dict | None:
    """Сливает {"en": ..., "ru": ...} из запроса с уже сохранёнными
    примерами (только при подмене существующей категории). incoming
    целиком None/{} — убрать все примеры; отдельный язык со значением
    None внутри incoming — убрать только его."""
    if incoming is None or incoming == {}:
        return None
    if not isinstance(incoming, dict):
        raise CategoryError("examples: ожидается объект вида {'en': '...', 'ru': '...'} или null")
    merged = dict(current or {})
    for lang, text in incoming.items():
        if text is None:
            merged.pop(lang, None)
            continue
        if not isinstance(text, str) or not text.strip():
            raise CategoryError(f"examples[{lang!r}]: ожидается непустая строка или null")
        merged[lang] = text
    return merged or None


def _merge_category_payload(
    name: str,
    summary: str | None,
    full: str | None,
    compact: str | None,
    examples,
    current_summary: str | None,
    current_record: dict | None,
) -> tuple[str, dict]:
    """Валидирует одну запись из "categories" в теле /analyze и
    сливает её с текущей дефолтной категорией того же имени (если
    есть — current_summary/current_record; иначе оба None и summary/
    full/compact обязательны). Возвращает (summary, {"full", "compact",
    "examples"?}) — готовую запись для CategoryOverlay.registry."""
    new_summary = summary if summary is not None else current_summary
    new_full = full if full is not None else (current_record or {}).get("full")
    new_compact = compact if compact is not None else (current_record or {}).get("compact")
    new_examples = (
        _merge_examples((current_record or {}).get("examples"), examples)
        if examples is not _UNSET else (current_record or {}).get("examples")
    )

    for field, value in (("summary", new_summary), ("full", new_full), ("compact", new_compact)):
        if not isinstance(value, str) or not value.strip():
            raise CategoryError(f"categories[{name!r}]: поле {field!r} обязательно и не может быть пустым")

    record = {"full": new_full, "compact": new_compact}
    if new_examples:
        record["examples"] = new_examples
    return new_summary, record


def build_overlay(extra: list[dict] | None) -> CategoryOverlay | None:
    """Строит разовый оверлей категорий для одного вызова /analyze
    поверх текущих дефолтов (см. докстринг модуля). Ничего не пишет на
    диск, не вызывает reload(), не трогает _order/_summaries/_registry.

    extra — список объектов {"name", "summary"?, "full"?, "compact"?,
    "examples"?} (то же тело, что раньше принимал upsert_category, без
    "position"). Имя, совпадающее с уже существующей категорией,
    частично заменяет её ТОЛЬКО на время этого вызова: непереданные
    поля берутся от дефолта, место категории в order не меняется.
    Новое имя требует summary+full+compact и добавляется в конец order.

    None или [] возвращает None — вызывающая сторона (backends.py/
    prompt.py) в этом случае должна работать с дефолтами напрямую,
    без лишнего копирования реестра на каждый запрос без "categories".

    Бросает CategoryError при некорректном имени/теле — analyze.py
    ловит её и отвечает 400, не давая один плохой запрос уронить сервер
    (реестр по умолчанию тут в принципе не трогается, поэтому откатывать
    нечего — в отличие от прежнего upsert_category)."""
    if not extra:
        return None
    if not isinstance(extra, list):
        raise CategoryError("categories: ожидается список объектов")

    order = list(_order)
    summaries = dict(_summaries)
    registry = dict(_registry)

    for item in extra:
        if not isinstance(item, dict) or not item.get("name"):
            raise CategoryError("categories: каждый элемент должен быть объектом с непустым полем 'name'")
        name = item["name"]
        if not _NAME_RE.match(name):
            raise CategoryError(
                f"Некорректное имя категории {name!r}: разрешены только латинские буквы, цифры и '_'"
            )

        examples = item["examples"] if "examples" in item else _UNSET
        new_summary, record = _merge_category_payload(
            name, item.get("summary"), item.get("full"), item.get("compact"), examples,
            summaries.get(name), registry.get(name),
        )
        summaries[name] = new_summary
        registry[name] = record
        if name not in order:
            order.append(name)

    return CategoryOverlay(order=order, summaries=summaries, registry=registry)