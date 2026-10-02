"""Настройки приложения. Всё, что важно для продакшена, берётся из переменных окружения."""

import logging
import os
from datetime import timedelta
from pathlib import Path

from flask import current_app, has_app_context

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Подтягивает переменные из файла .env в корне проекта (рядом с run.py).

    Вызывается при импорте этого модуля — раньше, чем класс Config ниже прочитает окружение, а
    credentials.py — VISION_CREDENTIALS_KEY. Поэтому .env работает для любой точки входа:
    python run.py, gunicorn wsgi:app, flask db ...

    Уже заданные переменные окружения НЕ перезаписываются (override=False): `export` и настройки
    systemd/Docker приоритетнее файла. Файла нет — ничего не происходит (всё как раньше)."""
    env_file = BASE_DIR / ".env"
    if not env_file.is_file():
        return
    try:
        from dotenv import load_dotenv
    except ImportError:
        logging.getLogger("vision_app.config").warning(
            "Найден %s, но пакет python-dotenv не установлен — файл проигнорирован "
            "(pip install python-dotenv).", env_file,
        )
        return
    load_dotenv(env_file, override=False)


_load_dotenv()


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


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
    VISION_API_TIMEOUT = _env_int("VISION_API_TIMEOUT", 500)  # /analyze и /chat
    VISION_API_HEALTH_TIMEOUT = 10  # /health и GET /embeddings, сек
    VISION_API_CALL_TIMEOUT = 15  # вспомогательные GET/POST (/models, /sampling, /categories), сек
    # /models облачного провайдера (Gemini) — отдельный, более длинный таймаут: список приходит
    # из интернета и бывает медленным. Раньше из-за него приходилось поднимать общий таймаут выше.
    VISION_API_MODELS_TIMEOUT = 60
    # Сколько хранится в БД кэш списка моделей облачного провайдера, сек (кнопка «Обновить список»
    # на странице статуса обновляет его сразу).
    MODELS_CACHE_TTL = 6 * 3600
    VISION_API_EMBED_TIMEOUT = 30  # POST /embeddings, сек
    VISION_API_ERROR_CHARS = 300  # сколько символов текста ошибки сервера показываем
    VISION_API_RAW_TEXT_CHARS = 2000  # усечение «сырого» ответа модели без валидного JSON
    VISION_API_BACKEND_NAME_CHARS = 32
    VISION_API_MODEL_NAME_CHARS = 120

    # Язык ответов модели, если не выбран другой (передаётся серверу анализа как lang=...).
    DEFAULT_LANG = "ru"

    # --- Очередь анализов ---
    # Это дефолты уровня .env/окружения. Главный админ может переопределить их на
    # ходу в /panel/settings/ (settings_store.py) — тогда в дело идёт значение из БД,
    # а эти остаются запасным вариантом, если переопределения ещё/уже нет.
    QUEUE_WORKER_ENABLED = os.environ.get("QUEUE_WORKER_ENABLED", "1") == "1"
    QUEUE_WORKERS = int(os.environ.get("QUEUE_WORKERS", "1"))
    QUEUE_POLL_SECONDS = float(os.environ.get("QUEUE_POLL_SECONDS", "5"))
    QUEUE_MAX_FILES_PER_UPLOAD = int(os.environ.get("QUEUE_MAX_FILES_PER_UPLOAD", "20"))
    QUEUE_MAX_PENDING_PER_USER = int(os.environ.get("QUEUE_MAX_PENDING_PER_USER", "30"))

    # Допустимые границы для настроек выше — их проверяют /panel/settings/ (settings_store.py)
    # и сам обработчик очереди (потолок числа потоков защищает от опечатки в 100 потоков).
    QUEUE_WORKERS_MIN, QUEUE_WORKERS_MAX = 1, 8
    QUEUE_POLL_SECONDS_MIN, QUEUE_POLL_SECONDS_MAX = 1, 300
    QUEUE_MAX_FILES_PER_UPLOAD_MIN, QUEUE_MAX_FILES_PER_UPLOAD_MAX = 1, 500
    QUEUE_MAX_PENDING_PER_USER_MIN, QUEUE_MAX_PENDING_PER_USER_MAX = 1, 1000
    VISION_API_TIMEOUT_MIN, VISION_API_TIMEOUT_MAX = 5, 3600
    SETTING_STR_MAX_LEN = 300  # длина строковых настроек в /panel/settings/

    # Внутренняя кухня обработчика очереди (queue_worker.py).
    QUEUE_ERROR_PAUSE_SECONDS = 2  # пауза после непредвиденной ошибки цикла
    QUEUE_CLAIM_RETRIES = 5  # сколько раз пробуем захватить задачу при гонке
    QUEUE_STOP_JOIN_TIMEOUT = 5  # сколько ждём завершения потока при остановке, сек

    # Подпись к снимку (контекст для модели): длина в форме загрузки, в очереди и в чате.
    CAPTION_MAX_CHARS = 500

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
    EMBEDDING_BATCH_SIZE = 32  # сервер принимает максимум 64 текста за запрос; берём с запасом
    EMBEDDING_MAX_TEXT_CHARS = 4000  # серверный лимит — 8000; модель всё равно видит только начало
    EMBEDDING_MAX_MATCHED_IDS = 500  # сколько id выше порога держим для статистики by_risk
    # Фоновая доиндексация в простое очереди (vector_search.idle_backfill).
    EMBEDDING_IDLE_BATCHES = 10  # батчей за один холостой проход
    EMBEDDING_IDLE_PAUSE_UNAVAILABLE = 600  # эмбеддинги выключены/недоступны, сек
    EMBEDDING_IDLE_PAUSE_NOTHING = 300  # индексировать было нечего, сек
    EMBEDDING_IDLE_PAUSE_RETRY = 60  # модель качается / сбой, сек

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

    # --- Файлы и каталоги внутри UPLOAD_FOLDER ---
    UPLOADS_DIR = "uploads"  # загрузки анализа: uploads/ГГГГ/ММ/ДД/<uuid>.<ext>
    THUMBS_DIR = "thumbs"  # кэш миниатюр, повторяет структуру uploads/
    CHAT_UPLOADS_DIR = "chat_uploads"  # вложения чата

    THUMB_SIZE = 96  # px; в интерфейсе показываем 36-44 px, запас под retina-экраны
    THUMB_QUALITY = 80
    THUMB_CACHE_SECONDS = 7 * 24 * 3600  # Cache-Control: max-age для миниатюр
    CHAT_ATTACHMENT_CACHE_SECONDS = 24 * 3600  # то же для вложений чата
    FILE_HASH_CHUNK_BYTES = 1024 * 1024  # размер куска при чтении файла для SHA-256

    # Фоновый подсчёт хешей файлов (image_dedup.py).
    DEDUP_IDLE_BATCH = 100  # записей за один холостой проход воркера
    DEDUP_DEFAULT_BATCH = 200  # по умолчанию в backfill_hashes()
    DEDUP_CLI_BATCH = 500  # за проход в `flask backfill-image-hashes`
    DEDUP_IDLE_PAUSE = 600  # когда считать нечего (или сбой), сек

    # --- Чат ---
    CHAT_MAX_MESSAGE_LEN = 8000  # символов в одном сообщении (см. также maxlength в chat.html)
    CHAT_MAX_HISTORY_MESSAGES = 80  # сколько последних сообщений сессии уходит модели как контекст
    CHAT_TITLE_LEN = 60  # длина автоназвания чата
    # Метка ChatSession.turn_started_at старше этого срока считается «протухшей» (сервер упал или
    # был перезапущен посреди ответа модели) — чат снова принимает сообщения.
    CHAT_TURN_STALE_SECONDS = 20 * 60
    CHAT_MAX_TOOL_CALLS = 4  # сколько раз за один ход модель может обратиться к инструментам
    CHAT_PROMPT_MAX_CHARS = 8000  # длина системного промпта чата, который админ задаёт на /panel/categories/

    # Вложения-изображения чата (chat_images.py).
    CHAT_MAX_IMAGES_PER_MESSAGE = 4
    CHAT_MAX_IMAGE_BYTES = 15 * 1024 * 1024  # на один файл; общий лимит запроса — MAX_CONTENT_LENGTH
    CHAT_ATTACHMENT_NAME_MAX = 120  # длина имени вложения для показа и промпта
    # Что уходит модели: слишком большие картинки уменьшаются (иначе — лишние токены и мегабайты
    # base64 в каждом запросе, а детали сверх этого размера модель всё равно не различит).
    CHAT_MODEL_MAX_SIDE = 1568  # px по длинной стороне
    CHAT_MODEL_PASSTHROUGH_BYTES = 1_500_000  # файл не больше этого и в допустимом формате уходит как есть
    CHAT_MODEL_JPEG_QUALITY = 88

    # Результат анализа, который отдаётся модели (chat_jobs.py).
    CHAT_JOB_FIELD_CHARS = 700
    CHAT_JOB_ANSWER_CHARS = 3000
    CHAT_JOB_CATEGORY_CHARS = 80
    CHAT_JOB_DETAIL_CHARS = 300
    CHAT_JOB_MAX_SIGNALS = 12

    # Инструмент search_analyses (chat_tools/analyses.py). Эти же числа попадают в промпт модели.
    SEARCH_SCAN_LIMIT = 500  # сколько последних строк максимум разбираем (защита от полного скана)
    SEARCH_DEFAULT_LIMIT = 5
    SEARCH_MAX_LIMIT = 15
    SEARCH_MAX_CARDS = 8  # больше карточек под одним ответом — визуальный шум
    SEARCH_DESC_CHARS = 160
    SEARCH_MAX_SINCE_DAYS = 365
    SEARCH_MAX_QUERY_CHARS = 300
    SEARCH_MAX_KEYWORDS = 8

    # Инструмент search_users (chat_tools/users.py).
    USER_SEARCH_DEFAULT_LIMIT = 10
    USER_SEARCH_MAX_LIMIT = 25
    USER_SEARCH_MAX_SINCE_DAYS = 730  # можно смотреть вглубь на 2 года

    # --- Форматы дат (локальное время, см. utils.local_dt) ---
    DATETIME_FORMAT = "%d.%m.%Y %H:%M"
    DATETIME_SHORT_FORMAT = "%d.%m %H:%M"  # без года — для компактных списков
    DATE_FORMAT = "%d.%m.%Y"
    TIME_FORMAT = "%H:%M"
    DATETIME_CHAT_FORMAT = "%d.%m · %H:%M"  # список чатов в боковой панели

    # --- Пагинация и длина списков ---
    HISTORY_PER_PAGE = 12  # «Моя история»
    PANEL_PER_PAGE = 20  # списки в админ-панели
    DASHBOARD_RECENT_LIMIT = 6  # «последние анализы» на главной
    USER_RECENT_ANALYSES_LIMIT = 10  # последние анализы в профиле / карточке пользователя
    REJECTED_FILES_SHOWN = 5  # сколько отклонённых файлов перечисляем в сообщении
    BULK_DELETE_MAX_IDS = 1000  # сколько записей можно удалить одним запросом
    MODEL_NAME_MAX_LEN = 200  # длина названия модели в форме «Статус сервера»

    # --- Безопасность ---
    PASSWORD_MIN_LENGTH = 8
    PASSWORD_USERNAME_MIN_LEN = 3  # логины короче этого не сравниваются с паролем
    PASSWORD_USERNAME_SIMILARITY = 0.7  # порог похожести пароля на логин

    # --- БД ---
    SQL_IN_CHUNK = 500  # размер пачки id в DELETE/IN (лимит числа параметров SQL)
    SQLITE_TIMEOUT = 30  # сек; обработчик очереди пишет в БД из отдельного потока

    # --- Логи (logging_setup.py) ---
    LOG_DIR = os.environ.get("VISION_LOG_DIR", str(BASE_DIR / "logs"))
    LOG_BACKUP_DAYS = 14  # app.log
    LOG_ERROR_BACKUP_DAYS = 60  # app.error.log


# Хук переопределений из БД (ставит settings_store.py). Так config.py не импортирует БД/модели,
# а `conf()` при этом видит значения, которые главный админ поменял в /panel/settings/.
NO_OVERRIDE = object()
_override_provider = None


def set_override_provider(provider) -> None:
    """provider(key) -> значение из БД или NO_OVERRIDE. Вызывается один раз из settings_store."""
    global _override_provider
    _override_provider = provider


def conf(key: str):
    """Действующее значение настройки.

    Порядок: переопределение главного админа из /panel/settings/ (БД, подхватывается на лету) ->
    app.config (учитывает create_app(config=...) и тесты) -> Config. Вне контекста приложения
    (при импорте) — просто Config. Так одно и то же имя читается одинаково и в запросе, и в
    фоновом потоке. Всё, что должно меняться без перезапуска, нужно читать через conf() В МОМЕНТ
    использования, а не сохранять в константу на уровне модуля.
    """
    if has_app_context():
        if _override_provider is not None:
            value = _override_provider(key)
            if value is not NO_OVERRIDE:
                return value
        try:
            return current_app.config[key]
        except KeyError:
            pass
    return getattr(Config, key)
