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

    # --- Поиск «по смыслу» в чате (vector_search.py) ---
    # Порог cosine-сходства: ниже — результат в выдачу не попадает.
    # 0.28 — откалиброванное по реальным данным значение: религиозная одежда (никаб, хиджаб)
    # набирает ~0.30–0.31 по запросу «религиозная символика», что валидно; 0.35 их отсекал.
    # Относительный отступ (ниже) дополнительно защищает от шума на запросах с чётким топом.
    # Если выдача стала слишком шумной — поднимите через переменную окружения EMBEDDING_MIN_SIMILARITY.
    EMBEDDING_MIN_SIMILARITY = float(os.environ.get("EMBEDDING_MIN_SIMILARITY", "0.28"))
    # Отступ от лучшего результата: в выдачу идёт только то, что не ниже «лучший скор − отступ»
    # (и не ниже порога выше). 0 — выключить. 0.15 — стартовое значение, подбирается по логу
    # «vector_search: сравнено ...» (там видны топ-скоры каждого запроса).
    EMBEDDING_RELATIVE_MARGIN = float(os.environ.get("EMBEDDING_RELATIVE_MARGIN", "0.15"))
    # Сколько векторов (самых свежих анализов после фильтров) сравнивается за один запрос.
    EMBEDDING_SCAN_LIMIT = int(os.environ.get("EMBEDDING_SCAN_LIMIT", "5000"))

    # --- Чат: изображения, передаваемые модели напрямую ---
    # Сколько ПОСЛЕДНИХ вложений чата модель реально «видит» (уходят в /chat картинкой).
    # Каждая картинка — порядка 1–1.5 тыс. токенов контекста; более старые остаются в чате
    # и по-прежнему могут быть поставлены в очередь анализа, но модели напрямую не передаются.
    CHAT_CONTEXT_IMAGES = int(os.environ.get("CHAT_CONTEXT_IMAGES", "6"))

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