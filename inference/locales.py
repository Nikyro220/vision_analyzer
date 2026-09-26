import os, json, logging

LOCALES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "locales")
PAGES_DIR = os.path.join(LOCALES_DIR, "pages")
DEFAULT_LANG = "en"
LOCALES: dict[str, dict] = {}  # {"ru": {...}, "en": {...}}
PAGES: dict[str, dict[str, str]] = {}  # {"index.body": {"en": "...", "ru": "..."}}


def load_locales(directory=LOCALES_DIR):
    if not os.path.isdir(directory):
        logging.warning("locales: директория не найдена: %s", directory)
        return
    for fname in os.listdir(directory):
        if fname.endswith(".json"):
            lang = fname[:-5]
            with open(os.path.join(directory, fname), "r", encoding="utf-8") as f:
                LOCALES[lang] = json.load(f)


def load_pages(directory=PAGES_DIR):
    """Читает locales/pages/<имя>.<lang>.txt — длинные "страничные"
    тексты (например, index.body — помощь по GET /), которые неудобно
    держать в JSON одной экранированной строкой в одну строку файла.

    В отличие от LOCALES, тексты страниц НЕ проходят через str.format()
    (см. get_page/config._page) — литеральные { и } в примерах (curl с
    JSON-телом и т.п.) можно писать как есть, без удвоения "{{"/"}}"."""
    global PAGES
    PAGES = {}
    if not os.path.isdir(directory):
        logging.warning("locales: директория страниц не найдена: %s", directory)
        return
    for fname in os.listdir(directory):
        if not fname.endswith(".txt"):
            continue
        stem = fname[:-4]
        name, sep, lang = stem.rpartition(".")
        if not sep:
            logging.warning("locales: пропускаю файл страницы с именем без языка: %r", fname)
            continue
        with open(os.path.join(directory, fname), "r", encoding="utf-8") as f:
            PAGES.setdefault(name, {})[lang] = f.read()


def get_page(name: str, lang: str | None = None) -> str:
    if lang is None:
        lang = DEFAULT_LANG
    variants = PAGES.get(name, {})
    if lang in variants:
        return variants[lang]
    if DEFAULT_LANG in variants:
        return variants[DEFAULT_LANG]
    return f"???{name}???"


def set_default_lang(lang: str) -> None:
    """Меняет DEFAULT_LANG во время работы сервера.

    Достаточно вызвать это (или напрямую присвоить locales.DEFAULT_LANG = ...)
    в любой момент — get_locale/get_formatted читают текущее значение
    DEFAULT_LANG на каждый вызов, а не на момент импорта.
    """
    global DEFAULT_LANG
    if lang not in LOCALES:
        logging.warning(
            "locales: язык %r не загружен (доступны: %s), меняю всё равно",
            lang, list(LOCALES.keys()),
        )
    DEFAULT_LANG = lang
    logging.info("locales: язык по умолчанию переключён на %r", lang)


def format(s, **kwargs):
    try:
        return s.format(**kwargs)
    except (KeyError, IndexError) as e:
        logging.warning("locales: не удалось подставить %s в строку %r", e, s)
        return s


def get_locale(k, lang=None):
    if lang is None:
        lang = DEFAULT_LANG
    if lang in LOCALES and k in LOCALES[lang]:
        return LOCALES[lang][k]
    if k in LOCALES.get(DEFAULT_LANG, {}):
        return LOCALES[DEFAULT_LANG][k]
    return f"???{k}???"


def get_formatted(k, lang=None, **kwargs):
    return format(get_locale(k, lang), **kwargs)


load_locales()
load_pages()