"""
config.py — конфигурация vision_analyzer_server: константы (бэкенды,
таймауты, сэмплинг) плюс общий bootstrap опциональных модулей (locales,
vision_analyzer_prompt, image_upscaler) и мелкие обёртки над ними
(_t, _current_lang, _get_system_prompt), которые нужны и backends.py,
и server.py. Собраны в одном месте, чтобы не дублировать
try/except ImportError и предупреждения в логах по разным файлам.

ВАЖНО про мутируемые "глобалы": BACKEND/OLLAMA_HOST/VLLM_URL меняются
"на лету" через POST /config (см. server.py: handle_config) присвоением
config.BACKEND = ...", config.OLLAMA_HOST = ...", config.VLLM_URL = ...".
Другие модули должны делать `import config` и обращаться именно как
`config.BACKEND` / `config.OLLAMA_HOST` / `config.VLLM_URL` — если вместо
этого сделать `from config import BACKEND`, получится локальная копия
имени, и последующие изменения через /config перестанут быть видны.
SAMPLING_DEFAULTS этой проблемы не имеет — это обычный dict, /sampling
мутирует его на месте (SAMPLING_DEFAULTS[...] = ...), поэтому его можно
спокойно читать и через `config.SAMPLING_DEFAULTS`, и через
`from config import SAMPLING_DEFAULTS`.
"""

import logging
import os
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import aiohttp


LOG_DIR = Path(
    os.environ.get("VISION_LOG_DIR", Path(__file__).resolve().parent.parent / "logs")
)


def _setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    console = logging.StreamHandler()

    everything = TimedRotatingFileHandler(
        LOG_DIR / "analyzer.log", when="midnight", backupCount=14,
        encoding="utf-8", delay=True,
    )
    errors = TimedRotatingFileHandler(
        LOG_DIR / "analyzer.error.log", when="midnight", backupCount=60,
        encoding="utf-8", delay=True,
    )
    errors.setLevel(logging.WARNING)

    for h in (console, everything, errors):
        h.setFormatter(fmt)

    # force=True — снять хендлеры, добавленные раньше, и не задвоить вывод.
    logging.basicConfig(
        level=logging.INFO, handlers=[console, everything, errors], force=True,
    )


_setup_logging()

try:
    import prompt as prompt
except ImportError:
    prompt = None
    logging.warning(
        "config: prompt.py не найден, системный промпт будет пустым"
    )

try:
    import locales
except ImportError:
    locales = None
    logging.warning(
        "config: locales.py не найден, POST /lang будет недоступен"
    )

try:
    import image_upscaler
except ImportError:
    image_upscaler = None
    logging.warning(
        "config: image_upscaler.py не найден, "
        "апскейлинг маленьких изображений отключён"
    )

try:
    import fastembed as _fastembed_probe  # noqa: F401 — только проверка наличия пакета
except ImportError:
    _fastembed_probe = None
    logging.warning(
        "config: пакет fastembed не установлен, POST /embeddings будет недоступен "
        "(pip install fastembed)"
    )


def _current_lang() -> str:
    """Текущий язык сервера (для промпта модели и текстов ответов)."""
    return locales.DEFAULT_LANG if locales is not None else "ru"


def _t(key: str, **kwargs) -> str:
    """Короткий алиас для locales.get_formatted с фолбэком, если locales.py нет."""
    if locales is None:
        return f"???{key}???"
    return locales.get_formatted(key, **kwargs)


def _page(name: str, **kwargs) -> str:
    """Длинный "страничный" текст (см. locales.py: load_pages/get_page),
    например index.body — помощь по GET /. В отличие от _t, подстановка
    плейсхолдеров вида {key} делается точечным str.replace по каждому
    переданному kwarg, а не str.format(**kwargs) — поэтому сам текст
    страницы не обязан экранировать литеральные { и } (curl-примеры с
    JSON-телом можно писать как есть, без "{{"/"}}")."""
    if locales is None:
        return f"???{name}???"
    text = locales.get_page(name, _current_lang())
    for key, value in kwargs.items():
        text = text.replace("{" + key + "}", str(value))
    return text




# С переходом на двухпроходный анализ (classify + analyze, см. backends.py:
# _select_categories/_analyze_image) backends.py сам решает, какие категории
# и какой вариант (full/compact) передать в prompt.get_system_prompt — эта
# обёртка больше не используется в backends.py напрямую. Оставлена как общий
# "весь промпт, все категории" хелпер для внешнего кода (CLI-обёртка и т.п.,
# см. prompt.py: docstring).
def _get_system_prompt(lang: str | None = None) -> str:
    if prompt is None:
        return ""
    return prompt.get_system_prompt(lang or _current_lang())


# ---------------------------------------------------------------------------
# Конфигурация бэкендов
# ---------------------------------------------------------------------------

THINK_LEVELS = ("low", "medium", "high")

def parse_think(v):
    """True / False / 'low' / 'medium' / 'high'; иначе ValueError."""
    if isinstance(v, bool):        # JSON-тело: true/false приходят как bool
        return v
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "on"):
        return True
    if s in ("0", "false", "no", "off"):
        return False
    if s in THINK_LEVELS:
        return s
    raise ValueError(v)

# Значения по умолчанию — можно переопределить переменными окружения при
# запуске (VISION_ANALYZER_BACKEND / _OLLAMA_HOST / _VLLM_URL), а также
# "на лету", без перезапуска сервера, через GET/POST /config (см. server.py).
# BACKEND/OLLAMA_HOST/VLLM_URL остаются обычными module-level переменными —
# POST /config меняет их присвоением, тем же способом, что и
# locales.set_default_lang() меняет DEFAULT_LANG.
BACKEND = os.environ.get("VISION_ANALYZER_BACKEND", "vllm").strip().lower()  # "vllm" или "ollama"
OLLAMA_HOST = os.environ.get("VISION_ANALYZER_OLLAMA_HOST", "http://127.0.0.1:11434").strip()
VLLM_URL = os.environ.get("VISION_ANALYZER_VLLM_URL", "http://host.docker.internal:8000/v1").strip()

SERVER_HOST = "0.0.0.0"
SERVER_PORT = 6769

# ---------------------------------------------------------------------------
# Эмбеддинги текста (POST /embeddings)
# ---------------------------------------------------------------------------
# Отдельная лёгкая CPU-модель (ONNX через fastembed), никак не связана с
# BACKEND/OLLAMA_HOST/VLLM_URL выше — те выбирают модель для риск-анализа
# картинок (/analyze) и диалога (/chat), а эта только превращает текст в
# вектор для последующего векторного поиска на стороне vision_app.
#
# Модель фиксирована конфигом, а не переключается "на лету" через API, в
# отличие от BACKEND: разные модели эмбеддингов дают векторы в разных,
# несовместимых друг с другом пространствах, и подмена модели без явного
# намерения незаметно бы испортила уже посчитанные векторы (см. ручку
# /embeddings/info ниже — vision_app сверяет по ней, той ли моделью
# посчитан вектор, прежде чем доверять результату).
#
# EMBEDDING_MODEL должен быть именем модели из fastembed.TextEmbedding.
# list_supported_models() — список мультиязычных вариантов (нужен русский):
#   sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384-мерный, ~0.22 ГБ, самый быстрый на CPU)
#   sentence-transformers/paraphrase-multilingual-mpnet-base-v2 (768-мерный, ~1.0 ГБ, дефолт — баланс качества/скорости)
#   intfloat/multilingual-e5-large                              (1024-мерный, ~2.24 ГБ, лучшее качество, медленнее на CPU)
EMBEDDING_MODEL = os.environ.get(
    "VISION_ANALYZER_EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
).strip()
EMBEDDING_ENABLED = os.environ.get("VISION_ANALYZER_EMBEDDING_ENABLED", "1") == "1"
EMBEDDING_MAX_CHARS = int(os.environ.get("VISION_ANALYZER_EMBEDDING_MAX_CHARS", "8000"))
# Явная папка для весов модели эмбеддингов — ТОЛЬКО сюда fastembed скачивает
# и отсюда читает модель (cache_dir), а не в системный temp/HF-кэш. Можно
# переопределить VISION_ANALYZER_EMBEDDING_DIR (например, на общий том в
# Docker). Папка в .gitignore, в репозиторий веса не попадают.
EMBEDDING_CACHE_DIR = Path(
    os.environ.get(
        "VISION_ANALYZER_EMBEDDING_DIR",
        Path(__file__).resolve().parent / "models" / "embeddings",
    )
).resolve()
# Скачивание (при первом запуске) и загрузка модели в память идут в
# отдельном фоновом потоке, который стартует вместе с сервером (см.
# embeddings.py: start_background_load, server.py: on_startup). Сервер
# отвечает на запросы сразу; пока модель не готова, POST /embeddings
# отдаёт 503 + Retry-After, а прогресс виден в GET /embeddings и /health.
# Предварительный прогрев на этапе сборки образа по-прежнему возможен —
# см. warm_embeddings.py (использует ту же папку EMBEDDING_CACHE_DIR).

SAMPLING_DEFAULTS = {
    "temperature": float(os.environ.get("VISION_ANALYZER_TEMPERATURE", 0)),
    "top_p": float(os.environ.get("VISION_ANALYZER_TOP_P", 1.0)),
    "top_k": int(os.environ.get("VISION_ANALYZER_TOP_K", 1)),
    "seed": int(os.environ.get("VISION_ANALYZER_SEED", 42)),
    "num_ctx": (
        int(os.environ["VISION_ANALYZER_NUM_CTX"])
        if os.environ.get("VISION_ANALYZER_NUM_CTX")
        else None
    ),
    "num_predict": (
        int(os.environ["VISION_ANALYZER_NUM_PREDICT"])
        if os.environ.get("VISION_ANALYZER_NUM_PREDICT")
        else None          # None = не переопределять, как num_ctx
    ),
    "think": parse_think(os.environ.get("VISION_ANALYZER_THINK", "true")),
}


REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=500)
DISCOVERY_TIMEOUT = aiohttp.ClientTimeout(total=10)