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
* Кэш матрицы: векторы текущей модели лежат в памяти процесса одной numpy-матрицей (см.
  «Кэш матрицы векторов» ниже), чтобы не читать и не разбирать до scan_limit blob'ов на каждый
  запрос. Кэш — только справочник «id -> вектор»: кандидатов (с правами и фильтрами) по-прежнему
  отбирает SQL, поэтому устаревшая или лишняя строка в кэше права обойти не может. Отключается
  настройкой EMBEDDING_MEMORY_CACHE (тогда векторы читаются из БД, как раньше).
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
from .config import conf
from .extensions import db
from .models import AnalysisEmbedding, AnalysisResult, Status, utcnow
from .services import EmbeddingNotReady, VisionApiError

log = logging.getLogger("vision_app.vector_search")

# Размеры батча/текста/выдачи и паузы фоновой индексации — в config.py (EMBEDDING_*); читаются через
# conf() в момент использования, чтобы правки в /panel/settings/ действовали без перезапуска.
_FALLBACK_PAUSE = 60  # сек; только если не удалось даже прочитать настройку


# ----------------------------------------------------------------------------
# Текст и (де)сериализация векторов
# ----------------------------------------------------------------------------
def _report_parts(raw_report) -> tuple[list[str], str, list[str]]:
    """(body_marks, text_on_image, детали сигналов) из raw_report. Терпимо к чужой форме данных:
    у старых анализов этих ключей нет, а ответ модели может нарушить схему."""
    if not isinstance(raw_report, dict):
        return [], "", []
    marks: list[str] = []
    check = raw_report.get("body_marks_check")  # результат точечной проверки по картинке (chat_tools/body_marks.py)
    for source in (raw_report.get("body_marks"), check.get("body_marks") if isinstance(check, dict) else None):
        if isinstance(source, str):
            source = [source]
        for m in source if isinstance(source, list) else []:
            m = str(m).strip()
            if m and m not in marks:
                marks.append(m)
    on_image = raw_report.get("text_on_image")
    on_image = on_image.strip() if isinstance(on_image, str) else ""
    signals = raw_report.get("signals")
    details = [
        str(s["detail"]).strip()
        for s in (signals if isinstance(signals, list) else [])
        if isinstance(s, dict) and str(s.get("detail") or "").strip()
    ]
    return marks, on_image, details


def searchable_text(description: str | None, caption: str | None, raw_report=None) -> str:
    """Весь текст анализа, по которому ищем ключевые слова: описание, подпись, метки на теле,
    текст на изображении и детали сигналов. Без усечения. Порядок здесь не важен — для эмбеддинга
    он важен, см. embedding_text()."""
    marks, on_image, details = _report_parts(raw_report)
    parts = [(description or "").strip(), (caption or "").strip(), *marks, on_image, *details]
    return "\n".join(p for p in parts if p)


def embedding_text(description: str | None, caption: str | None, raw_report=None) -> str:
    """Что именно эмбеддим. Пустая строка — эмбеддить нечего. Окно модели ограничено (~384 токена),
    поэтому короткие и ценные части (метки на теле, текст на изображении) идут ПЕРЕД длинным
    описанием: иначе усечение отрежет именно их."""
    marks, on_image, details = _report_parts(raw_report)
    parts = [*marks, on_image, (description or "").strip(), (caption or "").strip(), *details]
    return "\n".join(p for p in parts if p)[:int(conf("EMBEDDING_MAX_TEXT_CHARS"))]


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
            select(AnalysisResult.description, AnalysisResult.caption, AnalysisResult.raw_report, AnalysisResult.status)
            .where(AnalysisResult.id == analysis_id)
        ).one_or_none()
        if row is None or row.status != Status.DONE:
            return False
        text = embedding_text(row.description, row.caption, row.raw_report)
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
    пересчитывает вообще все). Идёт батчами по EMBEDDING_BATCH_SIZE по возрастанию id (keyset), поэтому
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
            select(AnalysisResult.id, AnalysisResult.description, AnalysisResult.caption, AnalysisResult.raw_report)
            .outerjoin(AnalysisEmbedding, AnalysisEmbedding.analysis_id == AnalysisResult.id)
            .where(
                AnalysisResult.id > last_id,
                AnalysisResult.status == Status.DONE,
                AnalysisResult.description != "",
            )
            .order_by(AnalysisResult.id)
            .limit(int(conf("EMBEDDING_BATCH_SIZE")))
        )
        if not everything:
            stmt = stmt.where(
                or_(AnalysisEmbedding.analysis_id.is_(None), AnalysisEmbedding.model != model)
            )
        rows = db.session.execute(stmt).all()
        if not rows:
            break
        last_id = rows[-1].id

        items = [(r.id, embedding_text(r.description, r.caption, r.raw_report)) for r in rows]
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
            result = backfill(max_batches=int(conf("EMBEDDING_IDLE_BATCHES")))
            # паузы читаем внутри контекста — иначе не увидим настройки из БД
            pause_unavailable = conf("EMBEDDING_IDLE_PAUSE_UNAVAILABLE")
            pause_retry = conf("EMBEDDING_IDLE_PAUSE_RETRY")
            pause_nothing = conf("EMBEDDING_IDLE_PAUSE_NOTHING")
        if result.reason:
            # модель качается/сервер лежит/эмбеддинги выключены — не долбим сервер каждые 5 с
            _next_idle_run = now + (
                pause_unavailable
                if result.reason.startswith(("state=disabled", "state=unavailable"))
                else pause_retry
            )
        elif result.indexed:
            _next_idle_run = 0.0  # могло остаться ещё — продолжим на следующем холостом цикле
        else:
            _next_idle_run = now + pause_nothing
    except Exception:  # noqa: BLE001
        log.exception("vector_search: сбой фоновой индексации")
        _next_idle_run = now + _FALLBACK_PAUSE
    finally:
        _idle_lock.release()


# ----------------------------------------------------------------------------
# Кэш матрицы векторов (в памяти процесса)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class _Snapshot:
    """Неизменяемый слепок всех векторов одной модели: после создания его никто не правит, поэтому
    параллельные поиски читают его без блокировок, а обновление — это подмена ссылки целиком."""

    model: str
    dim: int
    fingerprint: tuple  # (число строк, максимальный created_at) на момент загрузки
    row_of: dict  # analysis_id -> номер строки в matrix
    matrix: np.ndarray  # (N, dim) float32, строки L2-нормализованы


_snapshot: _Snapshot | None = None
_snapshot_lock = threading.Lock()  # только на время (пере)загрузки: чтобы N потоков не грузили одно и то же


def clear_cache() -> None:
    """Сбросить кэш (следующий поиск загрузит матрицу заново). В коде приложения не нужен —
    актуальность проверяет _fingerprint(); пригодится в тестах и при ручной отладке."""
    global _snapshot
    with _snapshot_lock:
        _snapshot = None


def _fingerprint(model: str, dim: int) -> tuple:
    """Дешёвый «отпечаток» таблицы векторов этой модели: (count, max(created_at)).
    Меняется при любой записи: новая строка поднимает max и count, пересчёт вектора (_upsert)
    обновляет created_at, удаление уменьшает count. Проверка идёт по БД, а не по флагу в памяти,
    поэтому работает и когда писатель — другой процесс (несколько воркеров WSGI, CLI reindex)."""
    count, latest = db.session.execute(
        select(func.count(), func.max(AnalysisEmbedding.created_at)).where(
            AnalysisEmbedding.model == model, AnalysisEmbedding.dim == dim
        )
    ).one()
    return int(count or 0), latest


def _load_snapshot(model: str, dim: int) -> _Snapshot:
    started = time.monotonic()
    # Отпечаток берём ДО чтения строк: если кто-то запишет вектор в промежутке, в слепке будет
    # не меньше, чем в отпечатке, а следующая проверка увидит расхождение и перезагрузит — то
    # есть в худшем случае лишняя загрузка, но не устаревшие данные под свежим отпечатком.
    fingerprint = _fingerprint(model, dim)
    stmt = select(AnalysisEmbedding.analysis_id, AnalysisEmbedding.vector).where(
        AnalysisEmbedding.model == model, AnalysisEmbedding.dim == dim
    )
    ids: list[int] = []
    blobs: list[bytes] = []
    want = dim * 4  # float32
    for aid, blob in db.session.execute(stmt):
        if len(blob) != want:  # битая строка: пропускаем, а не роняем весь поиск
            log.warning("vector_search: вектор анализа %s повреждён (%d байт, ожидалось %d) — пропущен", aid, len(blob), want)
            continue
        ids.append(aid)
        blobs.append(blob)
    matrix = (
        np.frombuffer(b"".join(blobs), dtype=np.float32).reshape(len(ids), dim)
        if ids else np.empty((0, dim), dtype=np.float32)
    )
    log.info(
        "vector_search: кэш векторов загружен: %d × %d (%.1f МБ), модель %s, %d мс",
        len(ids), dim, matrix.nbytes / 1e6, model, (time.monotonic() - started) * 1000,
    )
    return _Snapshot(model, dim, fingerprint, {aid: i for i, aid in enumerate(ids)}, matrix)


def _get_snapshot(model: str, dim: int, *, force: bool = False) -> _Snapshot:
    """Актуальный слепок для (model, dim). Без force — перезагружает, только если отпечаток
    таблицы изменился. Держим слепок одной модели: векторы разных моделей несовместимы, а
    поиск идёт только по «текущей»."""
    global _snapshot
    snap = _snapshot
    if not force and snap is not None and snap.model == model and snap.dim == dim:
        if snap.fingerprint == _fingerprint(model, dim):
            return snap
    with _snapshot_lock:
        snap = _snapshot
        if not force and snap is not None and snap.model == model and snap.dim == dim:
            # Пока ждали замок, другой поток мог уже перезагрузить.
            if snap.fingerprint == _fingerprint(model, dim):
                return snap
        snap = _load_snapshot(model, dim)
        _snapshot = snap
        return snap


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
    ranked: list[tuple[int, float]] = field(default_factory=list)  # ВСЕ выше порога, лучшие первыми (до EMBEDDING_MAX_MATCHED_IDS)
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
    use_cache = bool(conf("EMBEDDING_MEMORY_CACHE"))
    dim = int(query_vector.shape[0])
    # С кэшем blob'ы из БД не читаем: SQL отдаёт только id кандидатов (права и фильтры — те же).
    cols = [AnalysisEmbedding.analysis_id] if use_cache else [AnalysisEmbedding.analysis_id, AnalysisEmbedding.vector]
    if raw_report_filter is not None or text_filter is not None:
        cols.append(AnalysisResult.raw_report)
    if text_filter is not None:
        cols += [AnalysisResult.description, AnalysisResult.caption]
    stmt = (
        select(*cols)
        .join(AnalysisResult, AnalysisResult.id == AnalysisEmbedding.analysis_id)
        .where(
            *conditions,
            AnalysisEmbedding.model == model,
            AnalysisEmbedding.dim == dim,
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
        if text_filter is not None and not text_filter(searchable_text(row.description, row.caption, row.raw_report)):
            continue
        ids.append(row.analysis_id)
        if not use_cache:
            vectors.append(from_blob(row.vector))

    if use_cache and ids:
        snap = _get_snapshot(model, dim)
        rows = [snap.row_of.get(i) for i in ids]
        if None in rows:
            # Вектор записан уже после загрузки слепка (или в другом процессе) — один раз
            # перечитываем; кого и после этого нет, пропускаем (как анализ без вектора).
            snap = _get_snapshot(model, dim, force=True)
            rows = [snap.row_of.get(i) for i in ids]
            kept = [(i, r) for i, r in zip(ids, rows) if r is not None]
            ids = [i for i, _ in kept]
            rows = [r for _, r in kept]
        vectors_matrix = snap.matrix[rows] if ids else None
    else:
        vectors_matrix = np.vstack(vectors) if vectors else None

    result = SemanticResult(scanned=len(ids))
    if not ids:
        return result

    scores = vectors_matrix @ query_vector
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
    result.ranked = [(ids[i], float(scores[i])) for i in idx[:int(conf("EMBEDDING_MAX_MATCHED_IDS"))]]
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
