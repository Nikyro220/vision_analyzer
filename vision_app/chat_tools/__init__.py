"""Инструменты («тулзы») чата: модель сама решает, когда ей нужны данные.

  runner.py   — протокол и цикл хода чата (промпт, разбор JSON-вызова, повторный
                вызов модели с результатом); публичная точка входа — run_chat_turn.
  analyses.py — инструмент search_analyses (история анализов с проверкой прав).
  users.py    — инструмент search_users (только для админов).
  images.py   — инструмент analyze_image (ставит прикреплённые изображения в очередь анализа).

Новый инструмент — отдельный модуль рядом с analyses.py; в runner.py его нужно
описать в промпте и добавить в разбор вызова.
"""

from .images import ChatImage, attachment_note
from .runner import ChatTurn, delivery_message, run_chat_turn

__all__ = ["ChatImage", "ChatTurn", "attachment_note", "delivery_message", "run_chat_turn"]
