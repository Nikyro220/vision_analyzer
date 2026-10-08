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
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import func, or_, select, update

from . import image_dedup, vector_search
from .categories_store import build_categories_payload
from .config import Config, conf
from .extensions import db
from .models import AnalysisResult, Status, utcnow
from .services import AnalysisOutcome, LinkOutcome, VisionApiError, analyze_image, analyze_link
from .settings_store import get_analysis_target, get_runtime_setting

log = logging.getLogger("vision_app.queue")

# Момент запуска этого процесса (naive UTC — так даты хранятся в БД). Задача, начатая ДО него,
# гарантированно осталась от предыдущего запуска и может быть возвращена в очередь.
_PROCESS_STARTED = utcnow().replace(tzinfo=None)

_registry_lock = threading.Lock()
_workers: dict[int, list["QueueWorker"]] = {}

_EXT_BY_MIME = {
    "image/jpeg": ".jpg", "image/png": ".png", "image/gif": ".gif", "image/webp": ".webp",
    "image/bmp": ".bmp", "image/tiff": ".tiff",
}

_MIME_BY_EXT = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".gif": "image/gif",
    ".webp": "image/webp", ".bmp": "image/bmp", ".tif": "image/tiff", ".tiff": "image/tiff",
}


def queue_limit_hit(user_id: int, adding: int = 1) -> tuple[str, int, int] | None:
    """Лимиты очереди перед постановкой ``adding`` задач.

    None — можно. Иначе (``"user"`` | ``"total"``, сколько_сейчас, лимит): ``user`` — упёрлись в
    лимит самого пользователя (QUEUE_MAX_PENDING_PER_USER), ``total`` — в общий потолок очереди
    (QUEUE_MAX_PENDING_TOTAL). Считаются незавершённые задачи (status != done)."""
    per_user = get_runtime_setting("QUEUE_MAX_PENDING_PER_USER")
    total_limit = get_runtime_setting("QUEUE_MAX_PENDING_TOTAL")
    unfinished = AnalysisResult.status != Status.DONE
    mine = db.session.scalar(
        select(func.count(AnalysisResult.id)).where(AnalysisResult.user_id == user_id, unfinished)
    )
    if mine + adding > per_user:
        return "user", mine, per_user
    total = db.session.scalar(select(func.count(AnalysisResult.id)).where(unfinished))
    if total + adding > total_limit:
        return "total", total, total_limit
    return None


# ----------------------------------------------------------------------------
# Захват и обработка одной задачи
# ----------------------------------------------------------------------------
def claim_next() -> int | None:
    """Атомарно забирает самую старую задачу из очереди. Возвращает её id или None.

    UPDATE ... WHERE status='queued' срабатывает только для одного из конкурентов,
    поэтому задача не может быть взята дважды.
    """
    for _ in range(conf("QUEUE_CLAIM_RETRIES")):
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
    # Категории — тоже в момент обработки: правки в /panel/categories/ применяются
    # сразу, без перезапуска, даже к задачам, которые уже стояли в очереди.
    categories = build_categories_payload()
    db.session.rollback()  # не держим транзакцию на время долгого HTTP-запроса

    try:
        outcome = analyze_image(
            data, mime, backend=backend, model=model, caption=caption, categories=categories,
        )
        return outcome, ""
    except VisionApiError as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001 — задача не должна навсегда застревать в «обрабатывается»
        log.exception("Непредвиденная ошибка при анализе задачи")
        return None, f"Внутренняя ошибка обработки: {exc}"


def _outcome_values(outcome: AnalysisOutcome) -> dict:
    return dict(
        backend=outcome.backend,
        risk_level=outcome.risk_level,
        needs_human_review=outcome.needs_human_review,
        description=outcome.description,
        raw_report=outcome.raw_report,
        error="",
    )


def _finish(job_id: int, outcome, error: str) -> None:
    values: dict = {"status": Status.DONE, "finished_at": utcnow(), "is_new": True}
    if outcome is not None:
        values.update(_outcome_values(outcome))
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
    elif outcome is not None:
        # Вектор описания для поиска по смыслу. Best-effort: не вышло (модель эмбеддингов ещё
        # качается, сервер недоступен) — анализ уже сохранён, вектор добьёт idle_backfill.
        vector_search.index_analysis(job_id)


# ----------------------------------------------------------------------------
# Задачи-ссылки: картинку скачивает сервер анализа (yt-dlp / Open Graph)
# ----------------------------------------------------------------------------
def _run_link_analysis(url: str):
    """Возвращает (LinkOutcome | None, текст_ошибки)."""
    backend, model = get_analysis_target()
    categories = build_categories_payload()
    db.session.rollback()  # не держим транзакцию на время долгого HTTP-запроса
    try:
        return analyze_link(url, backend=backend, model=model, categories=categories), ""
    except VisionApiError as exc:
        return None, str(exc)
    except Exception as exc:  # noqa: BLE001 — задача не должна навсегда застревать в «обрабатывается»
        log.exception("Непредвиденная ошибка при анализе ссылки")
        return None, f"Внутренняя ошибка обработки: {exc}"


def _ext_for(mime: str) -> str:
    ext = _EXT_BY_MIME.get(mime)
    if ext:
        return ext
    tail = re.sub(r"[^a-z0-9]", "", mime.split("/")[-1].lower())[:8]
    return f".{tail}" if tail else ".jpg"


def _finish_link(app, job_id: int, user_id: int, url: str, result: LinkOutcome | None, error: str) -> None:
    """Сохраняет результат задачи-ссылки.

    Первая картинка поста заполняет саму задачу, остальные (карусель, галерея) становятся
    новыми готовыми записями того же пользователя. Без картинок — обычная ошибка задачи.
    """
    if result is None or not result.items:
        _finish(job_id, None, error or "Не удалось получить изображение по ссылке.")
        return

    root = Path(app.config["UPLOAD_FOLDER"])
    now = datetime.now(timezone.utc)
    written: list[Path] = []
    saved: list[tuple] = []  # (LinkItem, относительный путь, sha256)
    try:
        for item in result.items:
            rel = f"{Config.UPLOADS_DIR}/{now:%Y/%m/%d}/{uuid.uuid4().hex}{_ext_for(item.mime)}"
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(item.image_bytes)
            written.append(target)
            saved.append((item, rel, image_dedup.sha256_bytes(item.image_bytes)))
    except OSError as exc:
        for path in written:
            path.unlink(missing_ok=True)
        log.exception("Очередь: не удалось сохранить картинку из ссылки (задача %s)", job_id)
        _finish(job_id, None, f"Не удалось сохранить скачанное изображение: {exc}")
        return

    first, first_rel, first_hash = saved[0]
    finished = utcnow()
    updated = db.session.execute(
        update(AnalysisResult)
        .where(AnalysisResult.id == job_id, AnalysisResult.status == Status.PROCESSING)
        .values(
            status=Status.DONE, finished_at=finished, is_new=True,
            image_path=first_rel, image_mime=first.mime, image_hash=first_hash,
            original_name=first.name[:255], caption=first.caption,
            **_outcome_values(first.outcome),
        )
        .execution_options(synchronize_session=False)
    )
    if updated.rowcount != 1:
        db.session.rollback()
        for path in written:
            path.unlink(missing_ok=True)
        log.warning("Задача %s изменилась во время обработки — результат не записан", job_id)
        return

    extra_rows = []
    for item, rel, digest in saved[1:]:
        row = AnalysisResult(
            user_id=user_id, image_path=rel, original_name=item.name[:255], image_mime=item.mime,
            image_hash=digest, caption=item.caption, source_url=url, status=Status.DONE,
            started_at=finished, finished_at=finished, is_new=True, **_outcome_values(item.outcome),
        )
        db.session.add(row)
        extra_rows.append(row)
    db.session.commit()

    for row_id in [job_id, *(r.id for r in extra_rows)]:
        vector_search.index_analysis(row_id)
    if result.errors:
        log.info("Ссылка %s: часть картинок не обработана: %s", url, "; ".join(result.errors))


def process_next(app) -> bool:
    """Обрабатывает одну задачу. True — задача была, False — очередь пуста."""
    with app.app_context():
        try:
            job_id = claim_next()
            if job_id is None:
                return False
            row = db.session.execute(
                select(
                    AnalysisResult.image_path, AnalysisResult.image_mime, AnalysisResult.caption,
                    AnalysisResult.source_url, AnalysisResult.user_id,
                )
                .where(AnalysisResult.id == job_id)
            ).one()
            db.session.rollback()

            started = time.monotonic()
            if row.source_url and not row.image_path:  # анализ по ссылке: файла ещё нет
                log.info("Очередь: начинаю анализ задачи %s (ссылка %s)", job_id, row.source_url)
                link_result, error = _run_link_analysis(row.source_url)
                _finish_link(app, job_id, row.user_id, row.source_url, link_result, error)
            else:
                log.info("Очередь: начинаю анализ задачи %s (%s)", job_id, row.image_path)
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
                time.sleep(conf("QUEUE_ERROR_PAUSE_SECONDS"))
            if not busy:
                vector_search.idle_backfill(self.app)  # добирает векторы, пропущенные при анализе
                image_dedup.idle_backfill(self.app)  # считает хеши файлов у старых записей
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
            n = conf("QUEUE_WORKERS_MIN")
    # разумный потолок, чтобы опечатка в настройке не завела 100 потоков
    return min(max(n, conf("QUEUE_WORKERS_MIN")), conf("QUEUE_WORKERS_MAX"))


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
        worker.join(timeout=conf("QUEUE_STOP_JOIN_TIMEOUT"))