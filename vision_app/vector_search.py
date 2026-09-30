"""Векторный поиск по описаниям анализов («найди снимки с человеком в красной куртке»).

Как устроено
------------
* Вектор считает сервер анализа (POST /embeddings, см. inference/embeddings.py), здесь он
  только хранится и сравнивается. Хранилище — таблица analysis_embeddings в той же БД
  (models.AnalysisEmbedding): по одному float32-вектору на анализ, L2-нормализованному, поэтому
  cosine-сходство равно скалярному произведению. Сравнение — numpy, без внешней векторной БД:
  на тысячах и десятках тысяч анализов этого хватает (5000 векторов по 768 float32 ≈ 15 МБ).
* Запись: воркер очереди после завершения анализа вызывает index_analysis() (best-effort —
  сбой эмбеддинга анализ не ломает), а в простое добирает пропущенное через idle_backfill():
  анализы без вектора или с вектором другой модели. Вручную — `flask reindex-embeddings`.
* Поиск: semantic_search() применяет ТЕ ЖЕ SQL-условия (права, риск, даты), что и обычный
  режим тулза, и только потом ранжирует оставшихся кандидатов по сходству. Права никогда не
  зависят от векторов.
* Текст для эмбеддинга — description (+ caption, если есть). У модели эмбеддингов ограниченное
  окно (для multilingual-mpnet реестр fastembed указывает усечение на 384 токенах), длинный хвост
  она не увидит, поэтому description идёт первым.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sqlalchemy import delete, func, or_, select
from sqlalchemy.exc import IntegrityError

from . import services
from .config import Config
from .extensions import db
from .models import AnalysisEmbedding, AnalysisResult, Status, utcnow
from .services import EmbeddingNotReady, VisionApiError

log = logging.getLogger("vision_app.vector_search")

# Размеры батча/текста/выдачи и паузы фоновой индексации — в config.py (EMBEDDING_*).
_BATCH = Config.EMBEDDING_BATCH_SIZE
_MAX_TEXT_CHARS = Config.EMBEDDING_MAX_TEXT_CHARS
_MAX_MATCHED_IDS = Config.EMBEDDING_MAX_MATCHED_IDS


# ----------------------------------------------------------------------------
# Текст и (де)сериализация векторов
# ----------------------------------------------------------------------------
def embedding_text(description: str | None, caption: str | None) -> str:
    """Что именно эмбеддим. Пустая строка — эмбеддить нечего."""
    parts = [(description or "").strip(), (caption or "").strip()]
    return "\n".join(p for p in parts if p)[:_MAX_TEXT_CHARS]


def _text_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize(vec) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    return arr / norm if norm > 1e-12 else arr


def to_blob(vec) -> bytes:
    return normalize(vec).tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


def _upsert(analysis_id: int, text: str, vec, model: str) -> None:
    """Создаёт или обновляет вектор в сессии (commit — на вызывающем)."""
    arr = normalize(vec)
    row = db.session.get(AnalysisEmbedding, analysis_id)
    if row is None:
        db.session.add(
            AnalysisEmbedding(
                analysis_id=analysis_id, model=model, dim=int(arr.shape[0]),
                vector=arr.tobytes(), text_hash=_text_hash(text),
            )
        )
    else:
        row.model = model
        row.dim = int(arr.shape[0])
        row.vector = arr.tobytes()
        row.text_hash = _text_hash(text)
        row.created_at = utcnow()


def _commit_quietly() -> bool:
    """commit, который переживает гонку двух потоков, добавляющих один и тот же вектор."""
    try:
        db.session.commit()
        return True
    except IntegrityError:
        db.session.rollback()
        log.info("vector_search: вектор уже записан другим потоком — пропускаю")
        return False


# ----------------------------------------------------------------------------
# Индексация
# ----------------------------------------------------------------------------
def index_analysis(analysis_id: int) -> bool:
    """Считает и сохраняет вектор ОДНОГО анализа. Никогда не бросает исключений:
    True — вектор записан, False — не вышло (модель ещё качается, сервер недоступен,
    нечего эмбеддить); недостающее потом добьёт backfill. Нужен app context."""
    try:
        row = db.session.execute(
            select(AnalysisResult.description, AnalysisResult.caption, AnalysisResult.status)
            .where(AnalysisResult.id == analysis_id)
        ).one_or_none()
        if row is None or row.status != Status.DONE:
            return False
        text = embedding_text(row.description, row.caption)
        if not text:
            return False
        vectors, model = services.embed_texts([text])
        _upsert(analysis_id, text, vectors[0], model)
        return _commit_quietly()
    except EmbeddingNotReady:
        db.session.rollback()
        log.info("vector_search: анализ %s не проиндексирован сразу — модель эмбеддингов ещё не готова", analysis_id)
    except VisionApiError as exc:
        db.session.rollback()
        log.warning("vector_search: анализ %s не проиндексирован: %s", analysis_id, exc)
    except Exception:  # noqa: BLE001 — индексация не должна ронять воркер
        db.session.rollback()
        log.exception("vector_search: неожиданный сбой индексации анализа %s", analysis_id)
    return False


@dataclass
class BackfillResult:
    indexed: int = 0
    reason: str = ""  # пусто — дошли до конца; иначе почему остановились раньше


def _delete_orphans() -> int:
    """Векторы удалённых анализов (SQLite не выполняет ON DELETE CASCADE без PRAGMA)."""
    result = db.session.execute(
        delete(AnalysisEmbedding)
        .where(AnalysisEmbedding.analysis_id.not_in(select(AnalysisResult.id)))
        .execution_options(synchronize_session=False)
    )
    db.session.commit()
    return result.rowcount or 0


def backfill(*, everything: bool = False, max_batches: int | None = None) -> BackfillResult:
    """Индексирует анализы без вектора или с вектором другой модели (everything=True —
    пересчитывает вообще все). Идёт батчами по _BATCH по возрастанию id (keyset), поэтому
    не зацикливается на строках, которые не удалось обработать. Нужен app context."""
    try:
        status = services.get_embedding_status()
    except VisionApiError as exc:
        return BackfillResult(reason=f"сервер анализа недоступен: {exc}")
    state = status.get("state")
    if state != "ready":
        return BackfillResult(reason=f"state={state}")
    model = status.get("model") or ""

    _delete_orphans()

    total, batches, last_id = 0, 0, 0
    while max_batches is None or batches < max_batches:
        stmt = (
            select(AnalysisResult.id, AnalysisResult.description, AnalysisResult.caption)
            .outerjoin(AnalysisEmbedding, AnalysisEmbedding.analysis_id == AnalysisResult.id)
            .where(
                AnalysisResult.id > last_id,
                AnalysisResult.status == Status.DONE,
                AnalysisResult.description != "",
            )
            .order_by(AnalysisResult.id)
            .limit(_BATCH)
        )
        if not everything:
            stmt = stmt.where(
                or_(AnalysisEmbedding.analysis_id.is_(None), AnalysisEmbedding.model != model)
            )
        rows = db.session.execute(stmt).all()
        if not rows:
            break
        last_id = rows[-1].id

        items = [(r.id, embedding_text(r.description, r.caption)) for r in rows]
        items = [(aid, text) for aid, text in items if text]  # сервер выкидывает пустые -> порядок бы поплыл
        if items:
            try:
                vectors, used_model = services.embed_texts([t for _, t in items])
            except EmbeddingNotReady:
                return BackfillResult(indexed=total, reason="state=loading")
            except VisionApiError as exc:
                return BackfillResult(indexed=total, reason=str(exc))
            for (aid, text), vec in zip(items, vectors):
                _upsert(aid, text, vec, used_model)
            if _commit_quietly():
                total += len(items)
        batches += 1

    if total:
        log.info("vector_search: проиндексировано анализов: %d (модель %s)", total, model)
    return BackfillResult(indexed=total)


_idle_lock = threading.Lock()
_next_idle_run = 0.0


def idle_backfill(app) -> None:
    """Вызывается потоком-обработчиком очереди, когда очередь пуста. Дёшево, если делать
    нечего: следующий проход не раньше чем через несколько минут. Потоков-обработчиков
    может быть несколько — работает один (остальные сразу выходят). Не бросает исключений."""
    global _next_idle_run
    now = time.monotonic()
    if now < _next_idle_run or not _idle_lock.acquire(blocking=False):
        return
    try:
        with app.app_context():
            result = backfill(max_batches=Config.EMBEDDING_IDLE_BATCHES)
        if result.reason:
            # модель качается/сервер лежит/эмбеддинги выключены — не долбим сервер каждые 5 с
            _next_idle_run = now + (
                Config.EMBEDDING_IDLE_PAUSE_UNAVAILABLE
                if result.reason.startswith(("state=disabled", "state=unavailable"))
                else Config.EMBEDDING_IDLE_PAUSE_RETRY
            )
        elif result.indexed:
            _next_idle_run = 0.0  # могло остаться ещё — продолжим на следующем холостом цикле
        else:
            _next_idle_run = now + Config.EMBEDDING_IDLE_PAUSE_NOTHING
    except Exception:  # noqa: BLE001
        log.exception("vector_search: сбой фоновой индексации")
        _next_idle_run = now + Config.EMBEDDING_IDLE_PAUSE_RETRY
    finally:
        _idle_lock.release()


# ----------------------------------------------------------------------------
# Поиск
# ----------------------------------------------------------------------------
def get_vector(analysis_id: int) -> tuple[np.ndarray, str] | None:
    row = db.session.get(AnalysisEmbedding, analysis_id)
    return (from_blob(row.vector), row.model) if row is not None else None


def embed_query(text: str) -> tuple[np.ndarray, str]:
    """Эмбеддинг поискового запроса -> (нормализованный вектор, модель).
    Бросает EmbeddingNotReady / VisionApiError."""
    vectors, model = services.embed_texts([text])
    return normalize(vectors[0]), model


@dataclass
class SemanticResult:
    hits: list[tuple[int, float]] = field(default_factory=list)  # (analysis_id, similarity), лучшие первыми
    ranked: list[tuple[int, float]] = field(default_factory=list)  # ВСЕ выше порога, лучшие первыми (до _MAX_MATCHED_IDS)
    matched: int = 0  # сколько всего выше порога
    scanned: int = 0  # сколько векторов сравнили
    best_score: float | None = None  # лучший скор, даже если он ниже порога


def semantic_search(
    conditions: list,
    query_vector: np.ndarray,
    model: str,
    *,
    limit: int,
    min_similarity: float,
    scan_limit: int,
    relative_margin: float = 0.0,
    exclude_ids: set[int] | None = None,
    raw_report_filter: Callable[[dict], bool] | None = None,
    text_filter: Callable[[str], bool] | None = None,
) -> SemanticResult:
    """Ранжирует по сходству анализы, прошедшие SQL-условия `conditions` (права и фильтры
    строит вызывающий — здесь их не ослабить). Сравниваются только векторы модели `model`
    той же размерности; берутся scan_limit самых свежих. raw_report_filter — фильтр по
    категориям, которые лежат в JSON и в SQL не выражаются. text_filter — точный фильтр по тексту
    (description + caption), например по ключевым словам: отбирает кандидатов, а сходство
    только сортирует их.

    Отсечение двойное: сходство не ниже min_similarity (абсолютный порог) И не ниже
    «лучший скор − relative_margin» (0 — выключено). Один абсолютный порог не работает: у
    коротких общих запросов («человек с шапкой») почти все описания набирают заметный скор
    из-за слова «человек», и в выдачу попадает всё подряд. Относительное отсечение держит
    только то, что близко к лучшему совпадению, а на запросах, где лучший скор низкий,
    его роль играет абсолютный порог."""
    cols = [AnalysisEmbedding.analysis_id, AnalysisEmbedding.vector]
    if raw_report_filter is not None:
        cols.append(AnalysisResult.raw_report)
    if text_filter is not None:
        cols += [AnalysisResult.description, AnalysisResult.caption]
    stmt = (
        select(*cols)
        .join(AnalysisResult, AnalysisResult.id == AnalysisEmbedding.analysis_id)
        .where(
            *conditions,
            AnalysisEmbedding.model == model,
            AnalysisEmbedding.dim == int(query_vector.shape[0]),
        )
        .order_by(AnalysisResult.created_at.desc())
        .limit(scan_limit)
    )

    ids: list[int] = []
    vectors: list[np.ndarray] = []
    for row in db.session.execute(stmt):
        if exclude_ids and row.analysis_id in exclude_ids:
            continue
        if raw_report_filter is not None and not raw_report_filter(row.raw_report or {}):
            continue
        if text_filter is not None and not text_filter(f"{row.description or ''}\n{row.caption or ''}"):
            continue
        ids.append(row.analysis_id)
        vectors.append(from_blob(row.vector))

    result = SemanticResult(scanned=len(ids))
    if not ids:
        return result

    scores = np.vstack(vectors) @ query_vector
    result.best_score = float(scores.max())
    cutoff = min_similarity
    if relative_margin > 0:
        cutoff = max(cutoff, result.best_score - relative_margin)
    idx = np.flatnonzero(scores >= cutoff)
    idx = idx[np.argsort(-scores[idx])]
    # Только id и числа (текст запроса не логируем) — по этой строке подбирают пороги.
    top = np.argsort(-scores)[:8]
    log.info(
        "vector_search: сравнено %d, лучший %.2f, отсечка %.2f (порог %.2f, отступ %.2f), прошло %d; топ: %s",
        len(ids), result.best_score, cutoff, min_similarity, relative_margin, idx.size,
        ", ".join(f"#{ids[i]}={scores[i]:.2f}" for i in top),
    )
    result.matched = int(idx.size)
    result.ranked = [(ids[i], float(scores[i])) for i in idx[:_MAX_MATCHED_IDS]]
    result.hits = [(ids[i], float(scores[i])) for i in idx[:limit]]
    return result


def count_not_indexed(conditions: list, model: str) -> int:
    """Сколько анализов, подходящих под условия, поиском по смыслу пока не охвачено
    (нет вектора текущей модели) — модель может честно предупредить об этом пользователя."""
    stmt = (
        select(func.count())
        .select_from(AnalysisResult)
        .outerjoin(AnalysisEmbedding, AnalysisEmbedding.analysis_id == AnalysisResult.id)
        .where(
            *conditions,
            AnalysisResult.description != "",
            or_(AnalysisEmbedding.analysis_id.is_(None), AnalysisEmbedding.model != model),
        )
    )
    return int(db.session.scalar(stmt) or 0)
