"""Точка входа для разработки: python run.py"""

import atexit
import logging
import os
import subprocess
import sys
from pathlib import Path

from vision_app import create_app


logging.basicConfig(level=logging.INFO)

app = create_app()


# --- vision_analyzer (аналайзер картинок) поднимается рядом, подпроцессом ---
#
# aiohttp живёт в своём event loop, Flask dev-сервер — обычный WSGI,
# совмещать их в одном процессе/одном loop смысла нет — проще и надёжнее
# просто запустить `python server.py` отдельным процессом и следить за ним.
#
# Пути по умолчанию определяются относительно расположения run.py.
# При необходимости их можно переопределить через переменные окружения.

BASE_DIR = Path(__file__).resolve().parent


VISION_ANALYZER_DIR = Path(
    os.environ.get(
        "VISION_ANALYZER_DIR",
        BASE_DIR / "inference",
    )
).resolve()


def find_venv_python(directory: Path) -> Path | None:
    candidates = []

    if os.name == "nt":
        # Windows
        candidates = [
            directory / "venv" / "Scripts" / "python.exe",
            directory / ".venv" / "Scripts" / "python.exe",
        ]
    else:
        # Linux / macOS
        candidates = [
            directory / "venv" / "bin" / "python",
            directory / ".venv" / "bin" / "python",
        ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return None


found_python = find_venv_python(VISION_ANALYZER_DIR)

VISION_ANALYZER_PYTHON = Path(
    os.environ.get(
        "VISION_ANALYZER_PYTHON",
        found_python if found_python else sys.executable,
    )
).resolve()

_vision_analyzer_proc: subprocess.Popen | None = None


def _start_vision_analyzer() -> None:
    global _vision_analyzer_proc

    entrypoint = VISION_ANALYZER_DIR / "server.py"

    python_exe = (
        VISION_ANALYZER_PYTHON
        if VISION_ANALYZER_PYTHON.is_file()
        else Path(sys.executable)
    )

    if not entrypoint.is_file():
        logging.warning(
            "run.py: не нашёл %s — vision_analyzer не запущен "
            "(поправь VISION_ANALYZER_DIR / переменную окружения)",
            entrypoint,
        )
        return

    logging.info(
        "run.py: запускаю vision_analyzer (%s, python=%s)",
        entrypoint,
        python_exe,
    )

    _vision_analyzer_proc = subprocess.Popen(
        [str(python_exe), str(entrypoint)],
        cwd=str(VISION_ANALYZER_DIR),
    )

    atexit.register(_stop_vision_analyzer)


def _stop_vision_analyzer() -> None:
    if (
        _vision_analyzer_proc is not None
        and _vision_analyzer_proc.poll() is None
    ):
        logging.info("run.py: останавливаю vision_analyzer")

        _vision_analyzer_proc.terminate()

        try:
            _vision_analyzer_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _vision_analyzer_proc.kill()


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"

    # С включённым Flask-реloader'ом (debug=True) этот файл импортируется
    # дважды: в процессе-наблюдателе и в реальном рабочем процессе.
    # Запускаем подпроцесс только в настоящем воркере — иначе
    # vision_analyzer поднимется дважды и второй упадёт на занятом порту.

    if not debug or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        _start_vision_analyzer()

    try:
        app.run(
            host=os.environ.get("FLASK_RUN_HOST", "127.0.0.1"),
            port=int(os.environ.get("FLASK_RUN_PORT", "6967")),
            debug=debug,
        )
    finally:
        _stop_vision_analyzer()
