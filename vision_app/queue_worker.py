"""Очередь анализов: фоновый поток берёт задачи по одной и отправляет их на vision-сервер.

Как это работает
----------------
* Загрузка изображения только КЛАДЁТ запись в БД со статусом «queued» — быстро, без ожидания.
* Обработчик (поток внутри процесса приложения) забирает самую старую запись, ставит
  «processing», вызывает vision-сервер, затем ставит «done» (с результатом или с ошибкой).
  Только после этого запись появляется в истории.
* Очередь хранится в БД, поэтому переживает перезагрузку страницы, закрытие вкладки
  и перезапуск приложения (прерванные задачи при старте возвращаются в очередь).
* Обрабатывается по одной задаче на КАЖДЫЙ поток-обработчик; их количество задаёт
  QUEUE_WORKERS (по умолчанию 1). QUEUE_WORKERS, QUEUE_WORKER_ENABLED и QUEUE_POLL_SECONDS
  можно менять на ходу в /panel/settings/ (главный админ) — без перезапуска приложения.

Запуск: потоки стартуют лениво, при первом запросе к приложению (ensure_worker), поэтому
не мешают служебным командам flask и родительскому процессу перезагрузчика Werkzeug.
Приложение рассчитано на ОДИН процесс (например, gunicorn -w 1 --threads 4). При нескольких
процессах задачи не задвоятся (захват атомарный), но обрабатываться будут параллельно и
между процессами — суммарное число одновременных задач тогда будет больше, чем QUEUE_WORKERS.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import timezone
from pathlib import Path

from sqlalchemy import or_, select, update

from .extensions import db
from .models import AnalysisResult, Status, utcnow
from .services import VisionApiError, analyze_image
from .settings_store import get_analysis_target, get_runtime_setting

log = logging.getLogger("vision_app.queue")

# Момент запуска этого процесса (naive UTC — так даты хранятся в БД). Задача, начатая ДО него,
# гарантированно осталась от предыдущего запуска и может быть возвращена в очередь.
_PROCESS_STARTED = utcnow().replace(tzinfo=None)

_registry_lock = threading.Lock()
_workers: dict[int, list["QueueWorker"]] = {}

_MIME_BY_EXT = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif",
    ".webp": "image/webp", ".bmp": "image/bmp", ".tif": "image/tiff", ".tiff": "image/tiff",
}


# ----------------------------------------------------------------------------
# Захват и обработка одной задачи
# ----------------------------------------------------------------------------
def claim_next() -> int | None:
    """Атомарно забирает самую старую задачу из очереди. Возвращает её id или None.

    UPDATE ... WHERE status='queued' срабатывает только для одного из конкурентов,
    поэтому задача не может быть взята дважды.
    """
    for _ in range(5):
        job_id = db.session.scalar(
            select(AnalysisResult.id)
            .where(AnalysisResult.status == Status.QUEUED)
            .order_by(AnalysisResult.id)
            .limit(1)
        )
        if job_id is None:
            return None
        result = db.session.execute(
            update(AnalysisResult)
            .where(AnalysisResult.id == job_id, AnalysisResult.status == Status.QUEUED)
            .values(status=Status.PROCESSING, started_at=utcnow())
            .execution_options(synchronize_session=False)
        )
        db.session.commit()
        if result.rowcount == 1:
            return job_id
    return None


def _run_analysis(app, image_path: str, image_mime: str, caption: str = ""):
    """Возвращает (AnalysisOutcome | None, текст_ошибки)."""
    path = Path(app.config["UPLOAD_FOLDER"]) / image_path
    try:
        data = path.read_bytes()
    except OSError:
        return None, "Файл изображения не найден на диске."

    mime = image_mime or _MIME_BY_EXT.get(path.suffix.lower(), "image/jpeg")

    # Бэкенд и модель берём В МОМЕНТ обработки, а не загрузки: выбор из «Статуса сервера» действует сразу.
    backend, model = get_analysis_target()
    db.session.rollback()  # не держим транзакцию на время долгого HTTP-запроса

    try:
        outcome = analyze_image(data, mime, lang="ru", backend=backend, model=model, caption=caption)
        return outcome, ""
    except VisionApiError as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001 — задача не должна навсегда застревать в «обрабатывается»
        log.exception("Непредвиденная ошибка при анализе задачи")
        return None, f"Внутренняя ошибка обработки: {exc}"


def _finish(job_id: int, outcome, error: str) -> None:
    values: dict = {"status": Status.DONE, "finished_at": utcnow(), "is_new": True}
    if outcome is not None:
        values.update(
            backend=outcome.backend,
            risk_level=outcome.risk_level,
            needs_human_review=outcome.needs_human_review,
            description=outcome.description,
            raw_report=outcome.raw_report,
            error="",
        )
    else:
        values["error"] = error

    result = db.session.execute(
        update(AnalysisResult)
        .where(AnalysisResult.id == job_id, AnalysisResult.status == Status.PROCESSING)
        .values(**values)
        .execution_options(synchronize_session=False)
    )
    db.session.commit()
    if result.rowcount != 1:
        log.warning("Задача %s изменилась во время обработки — результат не записан", job_id)


def process_next(app) -> bool:
    """Обрабатывает одну задачу. True — задача была, False — очередь пуста."""
    with app.app_context():
        try:
            job_id = claim_next()
            if job_id is None:
                return False
            row = db.session.execute(
                select(AnalysisResult.image_path, AnalysisResult.image_mime, AnalysisResult.caption)
                .where(AnalysisResult.id == job_id)
            ).one()
            db.session.rollback()

            log.info("Очередь: начинаю анализ задачи %s (%s)", job_id, row.image_path)
            started = time.monotonic()
            outcome, error = _run_analysis(app, row.image_path, row.image_mime, row.caption)
            _finish(job_id, outcome, error)
            log.info(
                "Очередь: задача %s завершена за %.1f с (%s)",
                job_id, time.monotonic() - started, "ошибка: " + error if error else "успешно",
            )
            return True
        finally:
            db.session.remove()


def drain(app, limit: int = 10_000) -> int:
    """Обрабатывает всё, что есть в очереди, синхронно (для тестов и служебных задач)."""
    done = 0
    while done < limit and process_next(app):
        done += 1
    return done


def requeue_interrupted(app, before=None) -> int:
    """Возвращает в очередь задачи, которые были «в обработке» при остановке предыдущего запуска.

    Трогаем только задачи, начатые раньше `before` (по умолчанию — момента старта этого процесса).
    Так запуск обработчика не выбивает задачу, которую прямо сейчас выполняет другой поток
    или процесс. (Если второй процесс стартует посреди чужой обработки, эта задача будет
    проанализирована повторно — результат не потеряется, но работа задвоится.)
    """
    cutoff = before if before is not None else _PROCESS_STARTED
    if getattr(cutoff, "tzinfo", None) is not None:
        cutoff = cutoff.astimezone(timezone.utc).replace(tzinfo=None)
    with app.app_context():
        try:
            result = db.session.execute(
                update(AnalysisResult)
                .where(
                    AnalysisResult.status == Status.PROCESSING,
                    or_(AnalysisResult.started_at.is_(None), AnalysisResult.started_at < cutoff),
                )
                .values(status=Status.QUEUED, started_at=None)
                .execution_options(synchronize_session=False)
            )
            db.session.commit()
            return result.rowcount or 0
        finally:
            db.session.remove()


# ----------------------------------------------------------------------------
# Поток-обработчик
# ----------------------------------------------------------------------------
class QueueWorker(threading.Thread):
    def __init__(self, app, index: int = 1):
        super().__init__(name=f"analysis-queue-worker-{index}", daemon=True)
        self.app = app
        self.index = index
        self._wake = threading.Event()
        self._stop_requested = threading.Event()

    def wake(self) -> None:
        self._wake.set()

    def shutdown(self) -> None:
        self._stop_requested.set()
        self._wake.set()

    def run(self) -> None:
        log.info("Очередь: обработчик #%d запущен", self.index)

        while not self._stop_requested.is_set():
            try:
                busy = process_next(self.app)
            except Exception:  # noqa: BLE001 — поток не должен умирать из-за одной ошибки
                log.exception("Очередь: обработчик #%d — ошибка цикла обработки", self.index)
                busy = False
                time.sleep(2)
            if not busy:
                # Читаем интервал заново на каждом холостом цикле (а не один раз при
                # запуске потока) — так изменение в /panel/settings/ действует сразу,
                # а не только для новых потоков.
                with self.app.app_context():
                    poll = float(get_runtime_setting("QUEUE_POLL_SECONDS"))
                self._wake.wait(timeout=poll)
                self._wake.clear()
        log.info("Очередь: обработчик #%d остановлен", self.index)

def _real_app(app):
    """Из прокси current_app достаёт настоящий объект приложения (поток не живёт в контексте запроса)."""
    getter = getattr(app, "_get_current_object", None)
    return getter() if getter else app


def worker_enabled(app) -> bool:
    """Обработчик выключен в тестах и при QUEUE_WORKER_ENABLED=False (.env или /panel/settings/)."""
    if app.config.get("TESTING"):
        return False
    with app.app_context():
        return bool(get_runtime_setting("QUEUE_WORKER_ENABLED"))


def worker_count(app) -> int:
    """Сколько потоков-обработчиков держать одновременно (QUEUE_WORKERS: .env или /panel/settings/)."""
    with app.app_context():
        try:
            n = int(get_runtime_setting("QUEUE_WORKERS"))
        except (TypeError, ValueError):
            n = 1
    return min(max(n, 1), 8)  # разумный потолок, чтобы опечатка в настройке не завела 100 потоков


# Приложения, для которых уже когда-либо выполнялся _bootstrap_once (возврат в очередь
# задач, прерванных предыдущим запуском). Делается один раз за всё время жизни процесса,
# а не при каждом включении QUEUE_WORKER_ENABLED через /panel/settings/.
_bootstrapped: set[int] = set()


def _bootstrap_once(app, key: int) -> None:
    if key in _bootstrapped:
        return
    _bootstrapped.add(key)
    try:
        n = requeue_interrupted(app)
        if n:
            log.info("Очередь: возвращено в очередь прерванных задач: %d", n)
    except Exception:  # noqa: BLE001
        log.exception("Очередь: не удалось вернуть прерванные задачи")


def ensure_worker(app) -> list[QueueWorker]:
    """Приводит число живых потоков к worker_count(app): запускает недостающие,
    останавливает лишние (например, после уменьшения QUEUE_WORKERS в настройках),
    заменяет умершие. Если обработка выключена (QUEUE_WORKER_ENABLED=False) —
    останавливает все потоки и возвращает пустой список. Ничего не блокирует:
    остановка — это только сигнал потоку, без ожидания его завершения, поэтому
    вызов из обработчика HTTP-запроса не подвисает."""
    app = _real_app(app)
    key = id(app)

    if not worker_enabled(app):
        with _registry_lock:
            workers = _workers.pop(key, [])
        for worker in workers:
            worker.shutdown()
        return []

    target = worker_count(app)
    workers = _workers.get(key) or []
    if len(workers) == target and all(w.is_alive() for w in workers):
        return workers

    with _registry_lock:
        workers = [w for w in _workers.get(key, []) if w.is_alive()]
        _bootstrap_once(app, key)

        while len(workers) > target:  # QUEUE_WORKERS уменьшили — лишним просто говорим остановиться
            workers.pop().shutdown()

        while len(workers) < target:  # QUEUE_WORKERS увеличили (или первый запуск) — стартуем недостающих
            worker = QueueWorker(app, index=len(workers) + 1)
            worker.start()
            workers.append(worker)

        _workers[key] = workers
        return workers


def wake_worker(app) -> None:
    """Будит обработчиков после добавления задач (и запускает их, если они не запущены)."""
    for worker in ensure_worker(app):
        worker.wake()


def stop_worker(app) -> None:
    app = _real_app(app)
    with _registry_lock:
        workers = _workers.pop(id(app), [])
    for worker in workers:
        worker.shutdown()
    for worker in workers:
        worker.join(timeout=5)