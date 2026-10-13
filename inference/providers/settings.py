"""
providers/settings.py — настройки провайдера: описание параметра (Setting) и разбор значений.

У каждого провайдера СВОЙ набор параметров генерации, СВОИ значения по умолчанию и СВОИ
допустимые значения (диапазоны, список вариантов). Значения хранятся в самом провайдере
(Provider.settings), а не в общем словаре: изменение настроек одного провайдера не
затрагивает другие.

Параметр описывается один раз в классе провайдера (Provider.settings_spec). По этому же
описанию сервер:
  - проверяет значения (Setting.parse),
  - отдаёт схему клиентам в GET /providers (Setting.describe) — веб-панель строит форму
    по ней и ничего не знает о конкретных бэкендах.

Значение по умолчанию берётся в порядке приоритета:
  1. VISION_ANALYZER_<ПРОВАЙДЕР>_<ПАРАМЕТР>   (например, VISION_ANALYZER_VLLM_TEMPERATURE) —
     действует только на этого провайдера;
  2. VISION_ANALYZER_<ПАРАМЕТР>               (например, VISION_ANALYZER_TEMPERATURE) — старое
     общее имя, оставлено для совместимости: действует на всех провайдеров, у которых такой
     параметр есть, но только если переменная задана явно;
  3. значение, записанное в самом Setting.default.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any

#: строки, которыми сбрасывают nullable-параметр в «не задан» (как раньше у num_ctx/num_predict)
RESET_WORDS = ("", "auto", "none", "null", "default")

#: допустимые значения параметра-«размышления» (think); true/false → bool, остальное — строкой
THINK_CHOICES = (
    ("true", "вкл"),
    ("false", "выкл"),
    ("low", "low"),
    ("medium", "medium"),
    ("high", "high"),
)


class SettingError(ValueError):
    """Параметр не принят. kind: 'unknown' (у провайдера нет такого параметра) или
    'invalid' (значение не подходит). Сервер превращает это в локализованный HTTP 400."""

    def __init__(self, provider: str, key: str, kind: str, reason: str = "", available: tuple[str, ...] = ()):
        super().__init__(f"{provider}.{key}: {kind} {reason}".strip())
        self.provider = provider
        self.key = key
        self.kind = kind
        self.reason = reason
        self.available = available


@dataclass(frozen=True)
class Setting:
    """Один настраиваемый параметр провайдера.

    type:  "float" | "int" | "choice".
    nullable: None означает «не передавать / решает бэкенд» (сбрасывается словами из RESET_WORDS).
    choices: для type="choice" — ((значение, подпись), ...); значения "true"/"false" хранятся как bool.
    """

    key: str
    type: str
    default: Any = None
    label: str = ""
    hint: str = ""
    min: float | None = None
    max: float | None = None
    min_exclusive: bool = False
    choices: tuple[tuple[str, str], ...] = ()
    nullable: bool = False

    # --- разбор -----------------------------------------------------------

    def _fail(self, reason: str) -> ValueError:
        return ValueError(f"{reason}; {self.allowed()}")

    def allowed(self) -> str:
        """Краткое описание допустимых значений (не зависит от языка)."""
        if self.type == "choice":
            text = "one of: " + ", ".join(v for v, _ in self.choices)
        else:
            lo = "-inf" if self.min is None else _num(self.min)
            hi = "inf" if self.max is None else _num(self.max)
            left = "(" if self.min_exclusive else "["
            text = f"{self.type} in {left}{lo}, {hi}]"
        return text + (", or auto" if self.nullable else "")

    def parse(self, raw: Any) -> Any:
        """Приводит значение из запроса/окружения к типу параметра. ValueError — не подходит."""
        if raw is None or (isinstance(raw, str) and raw.strip().lower() in RESET_WORDS):
            if self.nullable:
                return None
            raise self._fail("value is required")

        if self.type == "choice":
            return self._parse_choice(raw)
        if isinstance(raw, bool):
            raise self._fail(f"got {raw!r}")
        value = self._parse_int(raw) if self.type == "int" else self._parse_float(raw)

        too_low = self.min is not None and (value <= self.min if self.min_exclusive else value < self.min)
        if too_low or (self.max is not None and value > self.max):
            raise self._fail(f"{value} is out of range")
        return value

    def _parse_choice(self, raw: Any) -> Any:
        text = ("true" if raw else "false") if isinstance(raw, bool) else str(raw).strip().lower()
        text = {"1": "true", "yes": "true", "on": "true", "0": "false", "no": "false", "off": "false"}.get(text, text)
        if text not in {v for v, _ in self.choices}:
            raise self._fail(f"got {raw!r}")
        return {"true": True, "false": False}.get(text, text)

    def _parse_int(self, raw: Any) -> int:
        if isinstance(raw, float) and raw.is_integer():
            return int(raw)
        if isinstance(raw, int):
            return raw
        text = str(raw).strip()
        try:
            return int(text)
        except ValueError:
            raise self._fail(f"got {raw!r}") from None

    def _parse_float(self, raw: Any) -> float:
        try:
            value = float(str(raw).strip().replace(",", ".")) if isinstance(raw, str) else float(raw)
        except (TypeError, ValueError):
            raise self._fail(f"got {raw!r}") from None
        if not math.isfinite(value):
            raise self._fail(f"got {raw!r}")
        return value

    # --- значение по умолчанию -------------------------------------------

    def initial(self, provider: str) -> Any:
        """Значение по умолчанию для провайдера с учётом переменных окружения (см. докстринг модуля)."""
        names = (
            f"VISION_ANALYZER_{provider.upper()}_{self.key.upper()}",
            f"VISION_ANALYZER_{self.key.upper()}",
        )
        for name in names:
            if name in os.environ:
                try:
                    return self.parse(os.environ[name])
                except ValueError as e:
                    raise ValueError(f"{name}={os.environ[name]!r}: {e}") from None
        return self.default

    # --- описание для GET /providers -------------------------------------

    def describe(self, value: Any, default: Any) -> dict:
        return {
            "key": self.key,
            "type": self.type,
            "label": self.label or self.key,
            "hint": self.hint,
            "value": value,
            "default": default,
            "min": self.min,
            "max": self.max,
            "min_exclusive": self.min_exclusive,
            "choices": [{"value": v, "label": label} for v, label in self.choices],
            "nullable": self.nullable,
        }


def _num(x: float) -> str:
    return str(int(x)) if float(x).is_integer() else str(x)
