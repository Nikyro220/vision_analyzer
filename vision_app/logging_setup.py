"""Логирование Flask-части: консоль + файлы в logs/ в корне проекта.

  logs/app.log        — всё (INFO и выше), ротация в полночь,
                        старые файлы: app.log.YYYY-MM-DD, хранится 14 дней
  logs/app.error.log  — только WARNING/ERROR/CRITICAL, хранится 60 дней

Анализатор (inference/) пишет в свои файлы analyzer*.log в той же папке.
Папку можно поменять переменной окружения VISION_LOG_DIR.

Под gunicorn с несколькими воркерами ротация по времени из нескольких
процессов может гоняться за один файл — держите один воркер (--threads
для параллелизма) либо ротируйте через logrotate.
"""

from __future__ import annotations

import logging
import os
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

from flask import Flask
from flask.logging import default_handler

LOG_DIR = Path(
    os.environ.get("VISION_LOG_DIR", Path(__file__).resolve().parent.parent / "logs")
)


def setup_logging(app: Flask | None = None) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    console = logging.StreamHandler()

    everything = TimedRotatingFileHandler(
        LOG_DIR / "app.log", when="midnight", backupCount=14,
        encoding="utf-8", delay=True,
    )
    errors = TimedRotatingFileHandler(
        LOG_DIR / "app.error.log", when="midnight", backupCount=60,
        encoding="utf-8", delay=True,
    )
    errors.setLevel(logging.WARNING)

    for h in (console, everything, errors):
        h.setFormatter(fmt)

    # force=True — идемпотентно (create_app можно вызвать повторно, например
    # в тестах) и заменяет basicConfig из run.py.
    logging.basicConfig(
        level=logging.INFO, handlers=[console, everything, errors], force=True,
    )

    # Flask по умолчанию вешает на app.logger свой консольный хендлер —
    # без этого строки попадали бы в консоль дважды.
    if app is not None:
        app.logger.removeHandler(default_handler)
