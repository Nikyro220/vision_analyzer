"""Темы оформления: по одному JSON-файлу на тему в vision_app/themes/<id>.json.

Формат файла:
    label, hint  — название и подпись карточки в «Оформлении»;
    group, order — группа («Тёмные», «Светлые», «Контрастные») и место в ней: у базовых тем 1–2,
                   дальше по алфавиту названий (так темы идут в выпадающем списке);
    mode         — "dark" | "light" (идёт в color-scheme и в расчёт производных цветов в theme.js);
    vars         — CSS-переменные палитры. Если среди них есть --accent, тема меняет акцент.

Из этих файлов собирается и CSS (/themes.css), и список пресетов для theme.js.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

THEMES_DIR = Path(__file__).parent / "themes"
GROUP_ORDER = ("Тёмные", "Светлые", "Контрастные")
REQUIRED_VARS = ("--bg", "--bg-panel", "--text")

_cache: dict = {"sig": None, "themes": [], "css": "", "version": ""}


def _read(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise ValueError(f"{path.name}: некорректный JSON ({exc})") from exc
    for key in ("label", "hint", "group", "mode", "vars"):
        if key not in data:
            raise ValueError(f"{path.name}: нет поля «{key}»")
    if data["mode"] not in ("dark", "light"):
        raise ValueError(f"{path.name}: mode должен быть dark или light")
    missing = [v for v in REQUIRED_VARS if v not in data["vars"]]
    if missing:
        raise ValueError(f"{path.name}: нет переменных {', '.join(missing)}")
    return {"id": path.stem, "order": 0, **data}


def _group_rank(group: str) -> int:
    return GROUP_ORDER.index(group) if group in GROUP_ORDER else len(GROUP_ORDER)


def _refresh() -> None:
    files = sorted(THEMES_DIR.glob("*.json"))
    sig = tuple((p.name, p.stat().st_mtime_ns) for p in files)
    if sig == _cache["sig"]:
        return
    themes = sorted((_read(p) for p in files), key=lambda t: (_group_rank(t["group"]), t["order"], t["id"]))
    blocks = []
    for t in themes:
        selector = f'[data-theme="{t["id"]}"]'
        if t["id"] == "dark":  # базовая палитра: действует и без data-theme
            selector = f":root,\n{selector}"
        lines = [f"{selector} {{", f'    color-scheme: {t["mode"]};']
        lines += [f"    {name}: {value};" for name, value in t["vars"].items()]
        blocks.append("\n".join(lines) + "\n}")
    css = "/* Собрано из vision_app/themes/*.json — править там. */\n" + "\n\n".join(blocks) + "\n"
    _cache.update(sig=sig, themes=themes, css=css, version=hashlib.sha1(css.encode()).hexdigest()[:10])


def load_themes() -> list[dict]:
    _refresh()
    return _cache["themes"]


def themes_css() -> str:
    _refresh()
    return _cache["css"]


def themes_version() -> str:
    _refresh()
    return _cache["version"]


def theme_presets() -> list[dict]:
    """Пресеты для theme.js (window.VT_PRESETS); «Авто» добавляет сам theme.js."""
    out = []
    for t in load_themes():
        p = {k: t[k] for k in ("id", "label", "hint", "group", "mode")}
        p["panel"] = t["vars"]["--bg-panel"]
        if "--accent" in t["vars"]:
            p["accent"] = t["vars"]["--accent"]
        out.append(p)
    return out
