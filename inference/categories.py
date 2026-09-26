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

Admin API (см. categories_api.py, роуты собраны в server.py):
  GET  /categories            — list_categories(): {"order": [...]}
  GET  /categories/summaries  — get_summaries(): {имя: summary, ...}
  GET  /categories/<имя>      — get_category(имя): summary+full+compact+
                                 examples одной категории
  POST /categories/order      — set_order(...): переставляет order
                                 целиком (та же перестановка имён)
  POST /categories/<имя>      — upsert_category(...): создаёт (если имени
                                 нет) или частично обновляет (если есть)
                                 summary/full/compact/examples/position
пишущие функции сами кладут файлы в inference/categories/ и вызывают
reload(); при ошибке валидации откатывают файлы на диске к состоянию до
вызова и перечитывают реестр заново, так что уже работающий сервер не
остаётся с половиной изменений.
"""

from __future__ import annotations

import json
import logging
import re
import threading
from pathlib import Path

_DIR = Path(__file__).resolve().parent / "categories"
_lock = threading.Lock()
_write_lock = threading.Lock()  # сериализует upsert_category/set_order между собой

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


def order() -> list[str]:
    """Список имён категорий в каноническом порядке (порядок появления
    в промпте) — единственный источник порядка для обоих проходов."""
    return list(_order)


def is_valid(name: str) -> bool:
    return name in _registry


def _selected_or_all(selected: list[str] | None) -> list[str]:
    """Пересекает выбор с реестром, сохраняя канонический порядок.
    None означает "все категории" (используется в compact-fallback и
    как обратно совместимое поведение по умолчанию)."""
    if selected is None:
        return list(_order)
    wanted = set(selected)
    return [name for name in _order if name in wanted]


def summaries_block() -> str:
    """<category name="...">summary</category> для КАЖДОЙ
    зарегистрированной категории — список кандидатов для первого
    (классифицирующего) вызова. Не фильтруется: фильтрация — это и
    есть то, что делает первый вызов.

    Собирается целиком из index.json (через _summaries) — файлы
    отдельных категорий (full/compact/examples) для этого не нужны."""
    parts = [
        f'<category name="{name}">{_summaries[name]}</category>'
        for name in _order
    ]
    return "\n".join(parts)


def full_signals_block(selected: list[str] | None = None) -> str:
    names = _selected_or_all(selected)
    return "\n\n".join(_registry[name]["full"] for name in names)


def compact_signals_block(selected: list[str] | None = None) -> str:
    names = _selected_or_all(selected)
    return "\n\n".join(_registry[name]["compact"] for name in names)


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


def examples_block(selected: list[str] | None, lang: str) -> str:
    """Собирает сцены-примеры для выбранных категорий на нужном языке.

    Возвращает "" (пустую строку), если ни у одной выбранной категории
    нет примеров на этот язык — например, в compact-fallback, где
    examples намеренно не подгружаются вовсе, или когда первый вызов
    выбрал только категории без примеров. Вызывающая сторона
    (prompt.py) должна аккуратно убрать секцию <examples> целиком,
    а не оставлять пустую пару тегов.
    """
    names = _selected_or_all(selected)
    scenes = [
        _registry[name]["examples"][lang]
        for name in names
        if _registry[name].get("examples") and lang in _registry[name]["examples"]
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
# Admin API — запись (см. categories_api.py: POST-хендлеры)
# ---------------------------------------------------------------------------

def _atomic_write_json(path: Path, data: dict) -> None:
    """Пишет во временный файл рядом и атомарно переименовывает поверх
    целевого — чтобы конкурентный reload() (в этом же процессе или
    случайно запущенный извне) никогда не увидел наполовину записанный
    JSON."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def _merge_examples(current: dict | None, incoming) -> dict | None:
    """Сливает {"en": ..., "ru": ...} из запроса с уже сохранёнными
    примерами. incoming целиком None/{} — убрать все примеры; отдельный
    язык со значением None внутри incoming — убрать только его."""
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


def upsert_category(
    name: str,
    *,
    summary: str | None = None,
    full: str | None = None,
    compact: str | None = None,
    examples=_UNSET,
    position: int | None = None,
    directory: Path = _DIR,
) -> dict:
    """Создаёт новую категорию или частично обновляет существующую;
    пишет index.json и categories/<имя>.json, затем reload(). Ответ
    POST /categories/<имя>.

    - Новой категории (имени нет в реестре) обязательны summary, full и
      compact — как и при загрузке с диска (см. _load).
    - У существующей категории можно передать любое подмножество полей —
      остальные остаются как на диске.
    - examples: см. _merge_examples.
    - position: 0-based индекс в общем order. Для новой категории по
      умолчанию — конец списка; для существующей, если не передан,
      позиция не меняется.

    При CategoryError (в том числе если reload() внезапно не пропустил
    уже как будто провалидированные данные — например, из-за гонки с
    другим процессом, трогающим те же файлы) откатывает файлы на диске
    к состоянию до вызова и перечитывает реестр — сервер остаётся с тем,
    что было. Возвращает get_category(name) при успехе.
    """
    if not _NAME_RE.match(name):
        raise CategoryError(
            f"Некорректное имя категории {name!r}: разрешены только латинские буквы, цифры и '_'"
        )

    with _write_lock:
        exists = name in _registry
        current = _registry.get(name, {})

        new_summary = summary if summary is not None else _summaries.get(name)
        new_full = full if full is not None else current.get("full")
        new_compact = compact if compact is not None else current.get("compact")
        new_examples = _merge_examples(current.get("examples"), examples) if examples is not _UNSET else current.get("examples")

        for field, value in (("summary", new_summary), ("full", new_full), ("compact", new_compact)):
            if not isinstance(value, str) or not value.strip():
                raise CategoryError(f"{name!r}: поле {field!r} обязательно и не может быть пустым")

        new_order = list(_order)
        if name not in new_order:
            idx = len(new_order) if position is None else max(0, min(position, len(new_order)))
            new_order.insert(idx, name)
        elif position is not None:
            new_order.remove(name)
            idx = max(0, min(position, len(new_order)))
            new_order.insert(idx, name)

        index_path = directory / "index.json"
        cat_path = directory / f"{name}.json"
        index_backup = index_path.read_bytes() if index_path.exists() else None
        cat_backup = cat_path.read_bytes() if cat_path.exists() else None

        new_summaries = dict(_summaries)
        new_summaries[name] = new_summary
        cat_record = {"full": new_full, "compact": new_compact}
        if new_examples:
            cat_record["examples"] = new_examples

        try:
            _atomic_write_json(index_path, {"order": new_order, "summaries": new_summaries})
            _atomic_write_json(cat_path, cat_record)
            reload(directory)
        except Exception:
            if index_backup is not None:
                index_path.write_bytes(index_backup)
            if cat_backup is not None:
                cat_path.write_bytes(cat_backup)
            elif cat_path.exists():
                cat_path.unlink()
            reload(directory)
            raise

    return get_category(name)


def set_order(new_order, directory: Path = _DIR) -> list[str]:
    """Полностью заменяет порядок категорий (index.json: "order"), не
    меняя ни одной другой части ни одной категории. Ответ
    POST /categories/order.

    new_order обязан быть перестановкой ровно тех же имён, что уже в
    реестре — добавлять или убирать категории тут нельзя (для добавления
    есть upsert_category, для удаления пока нет эндпоинта — см. TODO).
    Как и upsert_category, откатывает файл на диске при ошибке.
    """
    if not isinstance(new_order, list) or not all(isinstance(n, str) for n in new_order):
        raise CategoryError("order: ожидается список строк (имён категорий)")
    if len(new_order) != len(_order) or set(new_order) != set(_order):
        raise CategoryError(
            f"order должен быть перестановкой текущих категорий {sorted(_order)}, "
            f"получено {sorted(set(new_order))}"
        )

    with _write_lock:
        index_path = directory / "index.json"
        backup = index_path.read_bytes()
        try:
            _atomic_write_json(index_path, {"order": new_order, "summaries": dict(_summaries)})
            reload(directory)
        except Exception:
            index_path.write_bytes(backup)
            reload(directory)
            raise

    return list(_order)