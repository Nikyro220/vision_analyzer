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

import aiohttp

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


def _current_lang() -> str:
    """Текущий язык сервера (для промпта модели и текстов ответов)."""
    return locales.DEFAULT_LANG if locales is not None else "ru"


def _t(key: str, **kwargs) -> str:
    """Короткий алиас для locales.get_formatted с фолбэком, если locales.py нет."""
    if locales is None:
        return f"???{key}???"
    return locales.get_formatted(key, **kwargs)


def _get_system_prompt(lang: str | None = None) -> str:
    if prompt is None:
        return ""
    return prompt.get_system_prompt(lang or _current_lang())


# ---------------------------------------------------------------------------
# Конфигурация бэкендов
# ---------------------------------------------------------------------------

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

# Параметры сэмплинга модели — стартовые значения тоже можно задать через
# окружение, а поменять "на лету" (без перезапуска) через GET/POST /sampling.
# Это обычный dict, а не отдельные переменные: /sampling мутирует его на
# месте (SAMPLING_DEFAULTS[...] = ...), никакой пересборки не требуется.
SAMPLING_DEFAULTS = {
    "temperature": float(os.environ.get("VISION_ANALYZER_TEMPERATURE", 0)),
    "top_p": float(os.environ.get("VISION_ANALYZER_TOP_P", 1.0)),
    "top_k": int(os.environ.get("VISION_ANALYZER_TOP_K", 1)),
    "seed": int(os.environ.get("VISION_ANALYZER_SEED", 42)),
    # num_ctx — размер контекстного окна, актуален ТОЛЬКО для Ollama (у
    # vLLM размер контекста фиксирован при запуске сервера, per-request
    # не передаётся). По умолчанию — None, т.е. вообще не переопределяем:
    # Ollama использует дефолт модели из её Modelfile, как было и до
    # появления /sampling. Поднимать num_ctx стоит только осознанно —
    # это увеличивает объём KV-cache и заметно увеличивает время prefill
    # (вплоть до частичного оффлоада на CPU, если не хватает VRAM), так
    # что задавать большое значение "на всякий случай" не стоит — только
    # когда реально нужна длинная история (/analyze -> history) и есть
    # запас по VRAM. Задать явно можно через VISION_ANALYZER_NUM_CTX или
    # POST /sampling; вернуть обратно на "не переопределять" — передать
    # num_ctx="auto" (или "none"/"default"/"").
    "num_ctx": (
        int(os.environ["VISION_ANALYZER_NUM_CTX"])
        if os.environ.get("VISION_ANALYZER_NUM_CTX")
        else None
    ),
}


REQUEST_TIMEOUT = aiohttp.ClientTimeout(total=300)
DISCOVERY_TIMEOUT = aiohttp.ClientTimeout(total=10)

RISK_EMOJI = {
    "low": "🟢",
    "medium": "🟡",
    "high": "🔴",
}


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
