"""Точка входа для разработки: python run.py"""
import atexit
import logging
import os
import subprocess
import sys

from vision_app import create_app

logging.basicConfig(level=logging.INFO)

app = create_app()

# --- vision_analyzer (аналайзер картинок) поднимается рядом, подпроцессом ---
# aiohttp живёт в своём event loop, Flask dev-сервер — обычный WSGI,
# совмещать их в одном процессе/одном loop смысла нет — проще и надёжнее
# просто запустить `python server.py` отдельным процессом и следить за ним.
#
# Поправь пути под себя (или задай переменными окружения):
VISION_ANALYZER_DIR = os.environ.get(
    "VISION_ANALYZER_DIR", r"D:\vision_analyzer\inference"
)
VISION_ANALYZER_PYTHON = os.environ.get(
    "VISION_ANALYZER_PYTHON", r"D:\vision_analyzer\.venv\Scripts\python.exe"
)

_vision_analyzer_proc: subprocess.Popen | None = None


def _start_vision_analyzer() -> None:
    global _vision_analyzer_proc

    entrypoint = os.path.join(VISION_ANALYZER_DIR, "server.py")
    python_exe = VISION_ANALYZER_PYTHON if os.path.isfile(VISION_ANALYZER_PYTHON) else sys.executable

    if not os.path.isfile(entrypoint):
        logging.warning(
            "run.py: не нашёл %s — vision_analyzer не запущен "
            "(поправь VISION_ANALYZER_DIR / переменную окружения)",
            entrypoint,
        )
        return

    logging.info("run.py: запускаю vision_analyzer (%s, python=%s)", entrypoint, python_exe)
    _vision_analyzer_proc = subprocess.Popen([python_exe, entrypoint], cwd=VISION_ANALYZER_DIR)
    atexit.register(_stop_vision_analyzer)


def _stop_vision_analyzer() -> None:
    if _vision_analyzer_proc is not None and _vision_analyzer_proc.poll() is None:
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