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

TODO (не сейчас, но заложено в архитектуру): admin-эндпоинт
GET/POST /categories, который пишет/удаляет файлы в inference/categories/
и дёргает reload() — позволит менять список категорий, их порядок,
summary и правила на лету, без выкладки нового кода.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

_DIR = Path(__file__).resolve().parent / "categories"
_lock = threading.Lock()

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
