"""Конвертер «примеров сцен» категории: текст <-> структурированные сцены.

В БД (Category.example_en / example_ru) и на сервере анализа пример по-прежнему
хранится обычным текстом:

    Scene A: описание сцены
    { "signals": [...], "rationale": "...", "risk_level": "...", "needs_human_review": ... }

    Scene B: ...

Форма панели редактирует его как список сцен (описание + список сигналов +
rationale + risk + needs_human_review), а этот модуль разбирает текст в такие
сцены и собирает обратно в тот же формат.
"""

from __future__ import annotations

import json
import re

RISK_LEVELS = ("low", "medium", "high")

_BLOCK_SPLIT_RE = re.compile(r"\n\s*\n")
_SCENE_PREFIX_RE = re.compile(r"^\s*Scene\s+[A-Za-z0-9]+\s*:\s*", re.IGNORECASE)


def empty_scene() -> dict:
    return {
        "scene": "",
        "signals": [""],
        "rationale": "",
        "risk_level": "low",
        "needs_human_review": False,
    }


def parse_examples(text: str | None) -> list[dict]:
    """Текст примера -> список сцен. Блок без разбираемого JSON превращается
    в сцену с одним описанием (ничего не теряется)."""
    scenes: list[dict] = []
    for block in _BLOCK_SPLIT_RE.split((text or "").strip()):
        block = block.strip()
        if not block:
            continue
        scene = empty_scene()
        scene["signals"] = []
        pos = block.find("\n{")
        desc, data = block, None
        if pos != -1:
            try:
                data = json.loads(block[pos + 1:])
                desc = block[:pos]
            except ValueError:
                data = None
        scene["scene"] = _SCENE_PREFIX_RE.sub("", desc.strip(), count=1)
        if isinstance(data, dict):
            for sig in data.get("signals") or []:
                detail = sig.get("detail", "") if isinstance(sig, dict) else str(sig)
                scene["signals"].append(str(detail))
            scene["rationale"] = str(data.get("rationale") or "")
            risk = data.get("risk_level")
            scene["risk_level"] = risk if risk in RISK_LEVELS else "low"
            scene["needs_human_review"] = bool(data.get("needs_human_review"))
        scenes.append(scene)
    return scenes


def _label(index: int) -> str:
    return chr(ord("A") + index) if index < 26 else str(index + 1)


def build_examples(scenes: list[dict], category_name: str) -> str:
    """Список сцен -> текст примера. Сцены без описания пропускаются."""
    blocks: list[str] = []
    for scene in scenes:
        desc = _SCENE_PREFIX_RE.sub("", (scene.get("scene") or "").strip(), count=1).strip()
        if not desc:
            continue
        details = [str(d).strip() for d in scene.get("signals") or [] if str(d).strip()]
        risk = scene.get("risk_level")
        if risk not in RISK_LEVELS:
            risk = "low"
        if details:
            rows = [
                "    "
                + json.dumps(
                    {"id": f"S-{i}", "category": category_name, "detail": d}, ensure_ascii=False
                )
                for i, d in enumerate(details, 1)
            ]
            signals = '"signals": [\n' + ",\n".join(rows) + "\n  ]"
        else:
            signals = '"signals": []'
        body = (
            "{\n"
            f"  {signals},\n"
            f'  "rationale": {json.dumps((scene.get("rationale") or "").strip(), ensure_ascii=False)},\n'
            f'  "risk_level": "{risk}",\n'
            f'  "needs_human_review": {"true" if scene.get("needs_human_review") else "false"}\n'
            "}"
        )
        blocks.append(f"Scene {_label(len(blocks))}: {desc}\n{body}")
    return "\n\n".join(blocks)


def data_to_text(raw_json: str | None, category_name: str) -> str:
    """Значение скрытого поля формы (JSON-список сцен) -> текст для БД."""
    if not (raw_json or "").strip():
        return ""
    scenes = json.loads(raw_json)
    if not isinstance(scenes, list):
        raise ValueError("expected list")
    return build_examples([s for s in scenes if isinstance(s, dict)], category_name)
