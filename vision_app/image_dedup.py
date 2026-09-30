"""Один и тот же файл, загруженный несколько раз, — это ОДИН снимок, а не несколько.

Каждая загрузка создаёт отдельный AnalysisResult (с новым случайным именем файла на диске),
поэтому повторно загруженный `33.jpeg` в поиске выглядит как несколько разных снимков. Чтобы
их различать, у анализа хранится `image_hash` — SHA-256 содержимого файла:

* новые загрузки получают его сразу (blueprints/analyzer.py: _enqueue_uploads);
* старые записи добирает backfill_hashes() — в простое воркера очереди (idle_backfill) или
  вручную: `flask backfill-image-hashes`;
* поиск в чате (chat_tools/analyses.py) склеивает анализы с одинаковым хешем: показывает один
  снимок и перечисляет номера остальных анализов этого же файла.

Хеш байт ловит только ТОЧНЫЕ копии файла. Пересохранённая, пережатая или обрезанная картинка
получит другой хеш и будет считаться другим снимком (для этого понадобился бы перцептивный хеш).
Запись без хеша (NULL — ещё не посчитан) и с пустой строкой (файл на диске потерян) никогда
не склеивается ни с чем.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from pathlib import Path

from flask import current_app
from sqlalchemy import select, update

from .config import Config
from .extensions import db
from .models import AnalysisResult

log = logging.getLogger("vision_app.image_dedup")

_CHUNK = Config.FILE_HASH_CHUNK_BYTES


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def backfill_hashes(limit: int | None = Config.DEDUP_DEFAULT_BATCH) -> tuple[int, int]:
    """Считает image_hash для записей, где его ещё нет. Возвращает (посчитано, файл потерян).
    Потерянные файлы получают пустую строку — чтобы не пытаться снова и не склеивать их между
    собой. Нужен app context."""
    root = Path(current_app.config["UPLOAD_FOLDER"])
    stmt = (
        select(AnalysisResult.id, AnalysisResult.image_path)
        .where(AnalysisResult.image_hash.is_(None))
        .order_by(AnalysisResult.id)
    )
    if limit:
        stmt = stmt.limit(limit)
    rows = db.session.execute(stmt).all()

    hashed = missing = 0
    for row_id, rel_path in rows:
        try:
            value = _hash_file(root / rel_path) if rel_path else ""
        except OSError:
            value = ""
        if value:
            hashed += 1
        else:
            missing += 1
        db.session.execute(
            update(AnalysisResult).where(AnalysisResult.id == row_id).values(image_hash=value)
            .execution_options(synchronize_session=False)
        )
    db.session.commit()
    if rows:
        log.info("image_dedup: хеши посчитаны: %d, файл не найден: %d", hashed, missing)
    return hashed, missing


_idle_lock = threading.Lock()
_next_idle_run = 0.0


def idle_backfill(app) -> None:
    """Вызывается потоком-обработчиком очереди в простое. Дёшево, если считать нечего.
    Не бросает исключений."""
    global _next_idle_run
    now = time.monotonic()
    if now < _next_idle_run or not _idle_lock.acquire(blocking=False):
        return
    try:
        with app.app_context():
            hashed, missing = backfill_hashes(limit=Config.DEDUP_IDLE_BATCH)
        _next_idle_run = 0.0 if (hashed or missing) else now + Config.DEDUP_IDLE_PAUSE
    except Exception:  # noqa: BLE001
        log.exception("image_dedup: сбой фонового подсчёта хешей")
        _next_idle_run = now + Config.DEDUP_IDLE_PAUSE
    finally:
        _idle_lock.release()


def dedupe_ordered(items: list[tuple[int, str | None]]) -> tuple[list[int], dict[int, list[int]]]:
    """items — (id анализа, image_hash) в порядке предпочтения (для выдачи это порядок показа:
    свежие первыми или лучшие по сходству первыми). Возвращает (id представителей в том же
    порядке, {id представителя: [id остальных анализов этого же файла]}).
    Анализы без хеша (None или "") склеиваться не могут."""
    reps: list[int] = []
    others: dict[int, list[int]] = {}
    seen: dict[str, int] = {}
    for analysis_id, image_hash in items:
        if not image_hash:
            reps.append(analysis_id)
        elif image_hash in seen:
            others.setdefault(seen[image_hash], []).append(analysis_id)
        else:
            seen[image_hash] = analysis_id
            reps.append(analysis_id)
    return reps, others
