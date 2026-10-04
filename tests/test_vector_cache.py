"""Кэш матрицы векторов (vision_app/vector_search.py): результаты те же, что без кэша, и он не
отдаёт устаревшее после записи / пересчёта / удаления.

Запуск из корня репозитория:  python -m pytest tests/test_vector_cache.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import delete, select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision_app import create_app, vector_search  # noqa: E402
from vision_app.extensions import db  # noqa: E402
from vision_app.models import AnalysisEmbedding, AnalysisResult, Status, User  # noqa: E402

DIM, MODEL = 16, "test-model"


@pytest.fixture()
def app(tmp_path):
    app = create_app({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'test.db'}",
        "UPLOAD_FOLDER": str(tmp_path / "uploads"),
        "QUEUE_WORKER_ENABLED": False,
        "SECRET_KEY": "test",
        "WTF_CSRF_ENABLED": False,
    })
    with app.app_context():
        db.create_all()
        vector_search.clear_cache()
        yield app
        db.session.remove()
        vector_search.clear_cache()


def _vec(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=DIM).astype(np.float32)


def _seed(n: int, user_id: int = 1) -> list[int]:
    if db.session.get(User, user_id) is None:
        db.session.add(User(id=user_id, username=f"u{user_id}", password_hash="x"))
        db.session.commit()
    ids = []
    for i in range(n):
        row = AnalysisResult(
            user_id=user_id, image_path=f"x/{i}.png", status=Status.DONE, description=f"d{i}",
        )
        db.session.add(row)
        db.session.flush()
        vector_search._upsert(row.id, f"d{i}", _vec(row.id), MODEL)
        ids.append(row.id)
    db.session.commit()
    return ids


def _search(app, query_seed: int, *, use_cache: bool, conditions=None, **kw):
    app.config["EMBEDDING_MEMORY_CACHE"] = use_cache
    return vector_search.semantic_search(
        conditions or [AnalysisResult.status == Status.DONE],
        vector_search.normalize(_vec(query_seed)), MODEL,
        limit=kw.pop("limit", 10), min_similarity=kw.pop("min_similarity", -1.0),
        scan_limit=kw.pop("scan_limit", 5000), **kw,
    )


def _same(a, b):
    assert [i for i, _ in a.ranked] == [i for i, _ in b.ranked]
    np.testing.assert_allclose([s for _, s in a.ranked], [s for _, s in b.ranked], atol=1e-6)
    assert (a.scanned, a.matched) == (b.scanned, b.matched)


def test_cache_matches_db_path(app):
    ids = _seed(200)
    for seed in (1, 7, 123):
        _same(_search(app, seed, use_cache=True), _search(app, seed, use_cache=False))
    # и с отсечками, и с ограничением окна
    _same(
        _search(app, 3, use_cache=True, min_similarity=0.1, relative_margin=0.3, scan_limit=50),
        _search(app, 3, use_cache=False, min_similarity=0.1, relative_margin=0.3, scan_limit=50),
    )
    assert _search(app, 3, use_cache=True).scanned == len(ids)


def test_permissions_still_come_from_sql(app):
    mine = _seed(20, user_id=1)
    _seed(20, user_id=2)
    only_mine = [AnalysisResult.status == Status.DONE, AnalysisResult.user_id == 1]
    res = _search(app, 5, use_cache=True, conditions=only_mine)
    assert res.scanned == 20 and {i for i, _ in res.ranked} <= set(mine)
    _same(res, _search(app, 5, use_cache=False, conditions=only_mine))


def test_cache_is_reused_when_nothing_changed(app, monkeypatch):
    _seed(30)
    _search(app, 1, use_cache=True)
    loads = []
    real = vector_search._load_snapshot
    monkeypatch.setattr(vector_search, "_load_snapshot", lambda *a: loads.append(1) or real(*a))
    for seed in range(5):
        _search(app, seed, use_cache=True)
    assert loads == []  # пять поисков подряд — ни одной перезагрузки


def test_new_vector_is_visible(app):
    _seed(30)
    _search(app, 1, use_cache=True)  # прогрели кэш
    new_id = _seed(1)[0]
    res = _search(app, new_id, use_cache=True)  # запрос == вектор нового анализа
    assert res.ranked[0][0] == new_id and res.scanned == 31


def test_recomputed_vector_is_not_stale(app):
    ids = _seed(30)
    target = ids[5]
    before = _search(app, 999, use_cache=True)
    # пересчёт вектора у существующего анализа (как делает _upsert при смене описания/модели)
    vector_search._upsert(target, "new text", _vec(999), MODEL)
    db.session.commit()
    after = _search(app, 999, use_cache=True)
    assert after.ranked[0][0] == target and after.ranked[0][1] == pytest.approx(1.0, abs=1e-5)
    assert dict(before.ranked)[target] < 0.99
    _same(after, _search(app, 999, use_cache=False))


def test_deleted_analysis_never_returned(app):
    ids = _seed(30)
    _search(app, 1, use_cache=True)
    gone = ids[0]
    db.session.execute(delete(AnalysisEmbedding).where(AnalysisEmbedding.analysis_id == gone))
    db.session.execute(delete(AnalysisResult).where(AnalysisResult.id == gone))
    db.session.commit()
    res = _search(app, gone, use_cache=True)
    assert gone not in {i for i, _ in res.ranked} and res.scanned == 29


def test_vector_written_after_snapshot_triggers_reload(app):
    """Кандидат есть в SQL, но его нет в слепке (запись из другого процесса между проверкой
    отпечатка и выборкой) — слепок перечитывается, а не теряет кандидата."""
    _seed(10)
    _search(app, 1, use_cache=True)
    new_id = _seed(1)[0]
    # подсовываем СТАРЫЙ отпечаток, чтобы _get_snapshot без force посчитал слепок актуальным
    old = vector_search._snapshot
    vector_search._snapshot = vector_search._Snapshot(
        old.model, old.dim, vector_search._fingerprint(MODEL, DIM), old.row_of, old.matrix
    )
    res = _search(app, new_id, use_cache=True)
    assert res.ranked[0][0] == new_id


def test_empty_and_other_model(app):
    assert _search(app, 1, use_cache=True).scanned == 0
    _seed(5)
    app.config["EMBEDDING_MEMORY_CACHE"] = True
    other = vector_search.semantic_search(
        [AnalysisResult.status == Status.DONE], vector_search.normalize(_vec(1)), "other-model",
        limit=5, min_similarity=-1.0, scan_limit=100,
    )
    assert other.scanned == 0
