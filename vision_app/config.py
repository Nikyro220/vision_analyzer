"""Настройки приложения. Всё, что важно для продакшена, берётся из переменных окружения."""

import os
from datetime import timedelta
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    # ВАЖНО: в проде обязательно задайте свой секретный ключ.
    SECRET_KEY = os.environ.get("FLASK_SECRET_KEY", "dev-insecure-change-me")

    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", f"sqlite:///{BASE_DIR / 'db.sqlite3'}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Куда сохраняются загруженные изображения (аналог MEDIA_ROOT).
    UPLOAD_FOLDER = os.environ.get("UPLOAD_FOLDER", str(BASE_DIR / "media"))

    # Жёсткий лимит на размер запроса (50 МБ). В отличие от Django-версии,
    # здесь он реально отклоняет слишком большие файлы (HTTP 413).
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024

    # --- Клиент vision_analyzer_server.py ---
    VISION_API_BASE_URL = os.environ.get("VISION_API_BASE_URL", "http://127.0.0.1:6769")
    VISION_API_TIMEOUT = int(os.environ.get("VISION_API_TIMEOUT", "500"))

    # --- Очередь анализов ---
    # Это дефолты уровня .env/окружения. Главный админ может переопределить их на
    # ходу в /panel/settings/ (settings_store.py) — тогда в дело идёт значение из БД,
    # а эти остаются запасным вариантом, если переопределения ещё/уже нет.
    QUEUE_WORKER_ENABLED = os.environ.get("QUEUE_WORKER_ENABLED", "1") == "1"
    QUEUE_WORKERS = int(os.environ.get("QUEUE_WORKERS", "1"))
    QUEUE_POLL_SECONDS = float(os.environ.get("QUEUE_POLL_SECONDS", "5"))
    QUEUE_MAX_FILES_PER_UPLOAD = int(os.environ.get("QUEUE_MAX_FILES_PER_UPLOAD", "20"))
    QUEUE_MAX_PENDING_PER_USER = int(os.environ.get("QUEUE_MAX_PENDING_PER_USER", "30"))

    # Часовой пояс для отображения дат (в БД всё хранится в UTC).
    TIMEZONE = os.environ.get("APP_TIMEZONE", "Asia/Aqtobe")

    # Создавать таблицы автоматически при старте. Если вы перейдёте на
    # Flask-Migrate (flask db upgrade) — поставьте AUTO_CREATE_DB=0.
    AUTO_CREATE_DB = os.environ.get("AUTO_CREATE_DB", "1") == "1"

    # Сессии и cookies
    PERMANENT_SESSION_LIFETIME = timedelta(days=14)
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"
    SESSION_COOKIE_SECURE = os.environ.get("SESSION_COOKIE_SECURE", "0") == "1"

    # CSRF-токен живёт столько же, сколько сессия (форма может долго висеть открытой).
    WTF_CSRF_TIME_LIMIT = None