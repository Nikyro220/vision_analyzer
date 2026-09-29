"""Инструмент `search_analyses`: чтение истории анализов для чата.

Это слой данных для системы «тулзов» (см. runner.py): модель сама решает,
нужна ли ей история анализов, и присылает JSON-вызов с аргументами; здесь эти
аргументы валидируются, применяются права доступа, выполняется SQL-запрос и
собирается результат (JSON для модели + карточки-ссылки для интерфейса).

Принципы:

  - Аргументы от модели — НЕДОВЕРЕННЫЙ ввод. Всё приводится к безопасным типам
    и диапазонам, неизвестное игнорируется (с предупреждением в результате).
  - Права применяются здесь, на сервере, и модель на них повлиять не может:
    user_id в аргументах не принимается вообще; `own_only` умеет только СУЖАТЬ
    выборку. Полный доступ ко всей истории — только у head_admin, остальные
    видят исключительно свои анализы.
  - Инструмент только читает. Худший исход злоупотребления (например, текст на
    картинке, который пытается «управлять» моделью) — пользователь увидит свои
    же данные.
  - Карточки строятся из тех же строк БД, что вернулись модели, а не из её
    текста, поэтому ссылка не может ускользнуть на несуществующий анализ.

Два режима поиска. Структурные вопросы (риск, категории, даты, «сколько…») — обычный
SQL: данные структурированы, фильтры точнее и проще. Вопросы про СОДЕРЖИМОЕ снимков
(«человек в красной куртке», «похожие на анализ #42») в фильтры не выражаются —
для них есть аргументы `query` и `similar_to`: те же SQL-условия и права, а затем
ранжирование кандидатов по векторному сходству описаний (см. vector_search.py).
Векторы права не определяют: сначала SQL отбирает, что пользователю вообще можно видеть.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from flask import current_app, url_for
from sqlalchemy import select

from .. import image_dedup, vector_search
from ..extensions import db
from ..models import RISK_LABELS, AnalysisResult, Category, RiskLevel, Status, User
from ..services import EmbeddingNotReady, VisionApiError
from ..utils import local_dt

TOOL_NAME = "search_analyses"

_SCAN_LIMIT = 500  # сколько последних строк максимум разбираем (защита от полного скана)
_DEFAULT_LIMIT = 5
_MAX_LIMIT = 15
_MAX_CARDS = 8  # больше карточек под одним ответом — визуальный шум
_DESC_LEN = 160
_MAX_SINCE_DAYS = 365
_MAX_QUERY_CHARS = 300
_MAX_KEYWORDS = 8
_VALID_RISKS = {RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.UNKNOWN}

# Порядок выдачи: по умолчанию от новых к старым; «самый первый / самый старый анализ» —
# order=oldest. Модель иногда присылает синонимы — принимаем самые очевидные.
_ORDER_NEWEST = "newest"
_ORDER_OLDEST = "oldest"
_ORDER_ALIASES = {
    "newest": _ORDER_NEWEST, "new": _ORDER_NEWEST, "latest": _ORDER_NEWEST, "recent": _ORDER_NEWEST,
    "desc": _ORDER_NEWEST, "newest_first": _ORDER_NEWEST,
    "oldest": _ORDER_OLDEST, "old": _ORDER_OLDEST, "earliest": _ORDER_OLDEST, "first": _ORDER_OLDEST,
    "asc": _ORDER_OLDEST, "oldest_first": _ORDER_OLDEST,
}


@dataclass
class ToolResult:
    text: str  # JSON-строка, которая уходит модели
    references: list = field(default_factory=list)  # карточки для фронта


# ---------------------------------------------------------------------------
# Категории (нужны и для описания инструмента в промпте, и для валидации)
# ---------------------------------------------------------------------------


def active_categories() -> list[tuple[str, str]]:
    """[(name, title)] активных категорий — что модель может подставить в `categories`."""
    rows = db.session.scalars(select(Category).where(Category.is_active.is_(True))).all()
    return [(c.name, c.title) for c in rows]


# ---------------------------------------------------------------------------
# Валидация аргументов
# ---------------------------------------------------------------------------


def _int_arg(value, lo: int, hi: int) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    if number < lo:
        return None  # заведомо некорректное значение (0, отрицательное) — игнорируем, а не подгоняем
    return min(hi, number)


def _bool_arg(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "да"}
    return bool(value)


def _fold(text: str) -> str:
    """Для сравнения слов: нижний регистр и ё=е (в описаниях модель пишет и «тёмные», и «темные»)."""
    return text.lower().replace("ё", "е")


def _stem(word: str) -> str:
    """Грубое «основа слова»: отрезаем окончание, чтобы «кепка» находила «кепке», «кепку», «кепкой».
    Совпадение потом идёт по НАЧАЛУ слова, так что длинная основа безопаснее короткой."""
    n = len(word)
    if n >= 6:
        return word[:-2]
    if n == 5:
        return word[:-1]
    return word


_KEYWORDS_FALLBACK = (
    "по указанным словам точных совпадений в описаниях нет — показан поиск по смыслу (приблизительный): "
    "слова подбирала модель, и они могли не совпасть с формулировками в описаниях; это НЕ значит, что "
    "подходящих снимков нет"
)


def _keywords_arg(value, warnings: list[str]) -> tuple[list[str], list[str]]:
    """Ключевые слова от модели -> (основы для точного поиска, исходные слова для запасного
    поиска по смыслу). Принимает список или строку через запятую."""
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, list):
        return [], []
    stems: list[str] = []
    originals: list[str] = []
    for item in value:
        clean = " ".join(str(item).split())[:40]
        words = _fold(clean).split()
        if not words or sum(len(w) for w in words) < 3:
            continue
        stem = " ".join(_stem(w) for w in words)
        if stem not in stems:
            stems.append(stem)
            originals.append(clean.lower())
    if len(stems) > _MAX_KEYWORDS:
        warnings.append(f"ключевых слов больше {_MAX_KEYWORDS} — лишние отброшены")
        stems, originals = stems[:_MAX_KEYWORDS], originals[:_MAX_KEYWORDS]
    return stems, originals


def _keyword_matcher(stems: list[str]):
    """text -> True, если в тексте есть ХОТЯ БЫ ОДНО из слов (по началу слова, без учёта регистра и ё)."""
    # Слова фразы склоняются все («головной убор» -> «головным убором»): между основами пропускаем окончание.
    patterns = [re.compile(r"(?<!\w)" + r"\w*\s+".join(re.escape(w) for w in stem.split())) for stem in stems]

    def match(text: str) -> bool:
        folded = _fold(text)
        return any(pat.search(folded) for pat in patterns)

    return match


def _row_text(row: AnalysisResult) -> str:
    return f"{row.description or ''}\n{row.caption or ''}"


def _query_arg(value) -> str | None:
    """Текст семантического запроса: приходит от модели (а значит, косвенно от пользователя или
    даже от текста на картинке) — только строка, схлопнутые пробелы, ограниченная длина."""
    if not isinstance(value, str):
        return None
    text = " ".join(value.split())[:_MAX_QUERY_CHARS]
    return text or None


def normalize_args(raw) -> tuple[dict, list[str]]:
    """Приводит сырые аргументы от модели к безопасному виду. Возвращает
    (очищенные аргументы, список предупреждений для модели)."""
    warnings: list[str] = []
    if not isinstance(raw, dict):
        warnings.append("args должен быть объектом — использованы значения по умолчанию")
        raw = {}

    known = {
        "limit", "since_days", "risk_level", "categories", "needs_review", "own_only", "count_only",
        "query", "similar_to", "keywords", "order",
    }
    unknown = sorted(str(k) for k in raw if k not in known)
    if unknown:
        warnings.append("неизвестные аргументы проигнорированы: " + ", ".join(unknown))

    args: dict = {
        "limit": _int_arg(raw.get("limit"), 1, _MAX_LIMIT) or _DEFAULT_LIMIT,
        "since_days": _int_arg(raw.get("since_days"), 1, _MAX_SINCE_DAYS),
        "risk_level": None,
        "categories": [],
        "needs_review": _bool_arg(raw.get("needs_review")),
        "own_only": _bool_arg(raw.get("own_only")),
        "count_only": _bool_arg(raw.get("count_only")),
        "query": _query_arg(raw.get("query")),
        "similar_to": _int_arg(raw.get("similar_to"), 1, 2_000_000_000),
        "order": _ORDER_NEWEST,
    }
    order = raw.get("order")
    if order not in (None, ""):
        resolved = _ORDER_ALIASES.get(str(order).strip().lower())
        if resolved:
            args["order"] = resolved
        else:
            warnings.append(f"order '{order}' не распознан (допустимо: newest, oldest) — использован newest")
    args["keywords"], keyword_words = _keywords_arg(raw.get("keywords"), warnings)
    args["keywords_text"] = ", ".join(keyword_words)  # для запасного поиска по смыслу, если точные слова ничего не нашли
    if args["query"] and args["similar_to"]:
        warnings.append("указаны и query, и similar_to — query проигнорирован")
        args["query"] = None
    if args["order"] == _ORDER_OLDEST and (args["query"] or args["similar_to"]):
        warnings.append("order игнорируется при query/similar_to — записи упорядочены по сходству, а не по дате")

    risk = raw.get("risk_level")
    if risk not in (None, ""):
        risk = str(risk).strip().lower()
        if risk in _VALID_RISKS:
            args["risk_level"] = risk
        else:
            warnings.append(f"risk_level '{risk}' не распознан и проигнорирован")

    cats = raw.get("categories")
    if isinstance(cats, str):
        cats = [cats]
    if isinstance(cats, list) and cats:
        lookup: dict[str, str] = {}
        for name, title in active_categories():
            lookup[name.strip().lower()] = name
            lookup[title.strip().lower()] = name
        for item in cats[:10]:
            found = lookup.get(str(item).strip().lower())
            if found:
                if found not in args["categories"]:
                    args["categories"].append(found)
            else:
                warnings.append(f"категория '{item}' не найдена и проигнорирована")

    return args, warnings


# ---------------------------------------------------------------------------
# Выборка + права доступа
# ---------------------------------------------------------------------------


def _in_scope_condition(user: User, own_only: bool):
    """Полный доступ ко всей истории — только у главного администратора.
    Обычный admin по видимости в чате приравнен к рядовому пользователю.
    own_only может только сузить выборку до своих анализов."""
    if user.is_head_admin and not own_only:
        return None
    return AnalysisResult.user_id == user.id


def _row_categories(row: AnalysisResult) -> set[str]:
    signals = (row.raw_report or {}).get("signals") or []
    return {s.get("category") for s in signals if isinstance(s, dict) and s.get("category")}


def _conditions(user: User, args: dict) -> list:
    """SQL-условия (права + фильтры) — общие для обычного и семантического поиска."""
    conditions = [AnalysisResult.status == Status.DONE]
    scope = _in_scope_condition(user, args["own_only"])
    if scope is not None:
        conditions.append(scope)
    if args["risk_level"]:
        conditions.append(AnalysisResult.risk_level == args["risk_level"])
    if args["needs_review"]:
        conditions.append(AnalysisResult.needs_human_review.is_(True))
    if args["since_days"]:
        conditions.append(AnalysisResult.created_at >= datetime.now(timezone.utc) - timedelta(days=args["since_days"]))
    return conditions


def _fetch_rows(user: User, args: dict) -> list[AnalysisResult]:
    # id — вторым ключом: у анализов, созданных в одну секунду, порядок иначе не определён.
    if args["order"] == _ORDER_OLDEST:
        ordering = (AnalysisResult.created_at.asc(), AnalysisResult.id.asc())
    else:
        ordering = (AnalysisResult.created_at.desc(), AnalysisResult.id.desc())
    stmt = select(AnalysisResult).where(*_conditions(user, args)).order_by(*ordering).limit(_SCAN_LIMIT)
    rows = list(db.session.scalars(stmt).all())

    if args["categories"]:
        wanted = set(args["categories"])
        rows = [r for r in rows if _row_categories(r) & wanted]
    if args["keywords"]:
        matches = _keyword_matcher(args["keywords"])
        rows = [r for r in rows if matches(_row_text(r))]
    return rows


# ---------------------------------------------------------------------------
# Результат: JSON для модели + карточки для интерфейса
# ---------------------------------------------------------------------------


def _clean(text: str) -> str:
    """Описания генерирует модель по картинке (а значит, их содержимое может
    быть подсунуто текстом на изображении) — не даём им имитировать служебные
    маркеры протокола."""
    return (text or "").replace("[TOOL", "(TOOL").replace("[/TOOL", "(/TOOL")


def _record(
    row: AnalysisResult, show_user: bool, similarity: float | None = None, same_image: list[int] | None = None
) -> dict:
    desc = _clean((row.description or "").strip().replace("\n", " "))
    if len(desc) > _DESC_LEN:
        desc = desc[: _DESC_LEN - 1] + "…"
    rec = {
        "id": row.id,
        "date": local_dt(row.created_at, "%d.%m.%Y %H:%M"),
        "risk": row.risk_level_display,
        "needs_review": bool(row.needs_human_review),
        "categories": sorted(_row_categories(row)),
        "description": desc,
    }
    if similarity is not None:
        rec["similarity"] = round(similarity, 2)
    if same_image:
        # тот же ФАЙЛ проанализирован ещё раз(ы) — это один снимок, а не несколько
        rec["same_image_analyses"] = sorted(same_image)
    if show_user and row.user:
        rec["user"] = row.user.username
    return rec


def _build_reference_cards(rows: list[AnalysisResult], show_user: bool) -> list[dict]:
    cards = []
    for row in rows[:_MAX_CARDS]:
        card = {
            "id": row.id,
            "url": url_for("analyzer.result_detail", pk=row.id),
            "thumb_url": url_for("analyzer.thumb", filename=row.image_path) if row.image_path else "",
            "label": row.original_name or f"Анализ #{row.id}",
            "risk_level": row.risk_level,
            "risk_label": row.risk_level_display,
            "date": local_dt(row.created_at, "%d.%m.%Y %H:%M"),
        }
        if show_user and row.user:
            card["username"] = row.user.username
        cards.append(card)
    return cards


def _error(message: str, **extra) -> ToolResult:
    return ToolResult(json.dumps({"error": message, **extra}, ensure_ascii=False))


def _search_semantic(user: User, args: dict, warnings: list[str]) -> ToolResult:
    """Режим `query` / `similar_to`: SQL-условия те же, что в обычном режиме, затем ранжирование
    кандидатов по сходству векторов описаний. При недоступной модели эмбеддингов возвращаем
    ЧЕСТНУЮ ошибку, а не «просто последние записи»: на «были ли снимки с ножом?» ответ «вот
    последние» был бы ложью."""
    conditions = _conditions(user, args)
    cfg = current_app.config
    min_similarity = float(cfg.get("EMBEDDING_MIN_SIMILARITY", 0.35))
    unavailable = (
        "поиск по смыслу сейчас недоступен ({reason}) — скажи об этом пользователю и предложи "
        "поиск по фильтрам (риск, категории, даты)"
    )

    try:
        exclude_ids: set[int] = set()
        same_as_source: list[int] = []
        if args["similar_to"]:
            scope = _in_scope_condition(user, args["own_only"])
            src_conditions = [AnalysisResult.id == args["similar_to"], AnalysisResult.status == Status.DONE]
            if scope is not None:
                src_conditions.append(scope)
            source = db.session.scalar(select(AnalysisResult).where(*src_conditions))
            if source is None:
                return _error(f"анализ #{args['similar_to']} не найден или недоступен пользователю")
            got = vector_search.get_vector(source.id)
            if got is None and vector_search.index_analysis(source.id):
                got = vector_search.get_vector(source.id)
            if got is None:
                return _error(f"анализ #{source.id} пока не проиндексирован для поиска по смыслу (нет описания или модель эмбеддингов недоступна)")
            query_vector, model = got
            exclude_ids = {source.id}
            if source.image_hash:
                # повторные анализы ТОГО ЖЕ файла — не «похожие», а он сам; исключаем и сообщаем отдельно
                same_as_source = sorted(
                    db.session.scalars(
                        select(AnalysisResult.id).where(
                            AnalysisResult.image_hash == source.image_hash,
                            AnalysisResult.id != source.id,
                            *([scope] if scope is not None else []),
                        )
                    ).all()
                )
                exclude_ids |= set(same_as_source)
        else:
            query_vector, model = vector_search.embed_query(args["query"])

        wanted = set(args["categories"])
        # Если есть keywords, кандидатов отбирает точный текстовый фильтр, а сходство только
        # сортирует их: отсечки по скору тут вредны (у «кепки» скор бывает ниже, чем у не-кепки).
        def run(gated: bool):
            return vector_search.semantic_search(
                conditions, query_vector, model,
                limit=args["limit"],
                min_similarity=-1.0 if gated else min_similarity,
                relative_margin=0.0 if gated else float(cfg.get("EMBEDDING_RELATIVE_MARGIN", 0.15)),
                text_filter=_keyword_matcher(args["keywords"]) if gated else None,
                scan_limit=int(cfg.get("EMBEDDING_SCAN_LIMIT", 5000)),
                exclude_ids=exclude_ids,
                raw_report_filter=(
                    (lambda report: bool({s.get("category") for s in (report.get("signals") or []) if isinstance(s, dict)} & wanted))
                    if wanted else None
                ),
            )

        gated = bool(args["keywords"])
        result = run(gated)
        if gated and result.scanned == 0:
            # Точный фильтр не нашёл ни одного описания с этими словами. Слова подбирает LLM и легко
            # промахивается («религиозные элементы» -> «крест, икона» при том, что на снимках никаб) —
            # поэтому «ноль» тут не ответ: откатываемся на поиск по смыслу и честно предупреждаем.
            warnings.append(_KEYWORDS_FALLBACK)
            result = run(False)
        not_indexed = vector_search.count_not_indexed(conditions, model)

        # Склеиваем повторные анализы одного и того же файла: остаётся лучший по сходству.
        meta: dict[int, tuple] = {}
        if result.ranked:
            for aid, ihash, risk, review in db.session.execute(
                select(
                    AnalysisResult.id, AnalysisResult.image_hash,
                    AnalysisResult.risk_level, AnalysisResult.needs_human_review,
                ).where(AnalysisResult.id.in_([i for i, _ in result.ranked]))
            ):
                meta[aid] = (ihash, risk, review)
        rep_ids, same_image = image_dedup.dedupe_ordered(
            [(aid, meta.get(aid, (None,))[0]) for aid, _ in result.ranked]
        )
        score_of = dict(result.ranked)
        shown_ids = rep_ids[: args["limit"]]

        rows_by_id: dict[int, AnalysisResult] = {}
        if shown_ids:
            found = db.session.scalars(select(AnalysisResult).where(AnalysisResult.id.in_(shown_ids))).all()
            rows_by_id = {r.id: r for r in found}
        by_risk: dict[str, int] = {}
        needs_review = 0
        for aid in rep_ids:
            _, risk, review = meta.get(aid, (None, "unknown", False))
            label = RISK_LABELS.get(risk, risk)
            by_risk[label] = by_risk.get(label, 0) + 1
            needs_review += 1 if review else 0
    except EmbeddingNotReady:
        return _error(unavailable.format(reason="модель эмбеддингов ещё загружается, повторить можно через минуту"))
    except VisionApiError as exc:
        current_app.logger.warning("search_analyses: эмбеддинги недоступны: %s", exc)
        return _error(unavailable.format(reason="сервер эмбеддингов не отвечает"))
    except Exception:  # noqa: BLE001 — сбой инструмента не должен ронять чат
        current_app.logger.exception("search_analyses: ошибка семантического поиска")
        return _error("не удалось выполнить поиск по смыслу")

    show_user = user.is_head_admin and not args["own_only"]
    payload: dict = {
        "mode": "semantic",
        "scope": "все пользователи" if show_user else "только анализы самого пользователя",
        "total_matched": len(rep_ids),
        "by_risk": by_risk,
        "needs_review_count": needs_review,
        "compared": result.scanned,
    }
    if len(result.ranked) > len(rep_ids):
        payload["duplicates_merged"] = len(result.ranked) - len(rep_ids)
    if args["similar_to"]:
        payload["similar_to"] = args["similar_to"]
        if same_as_source:
            payload["same_image_as_source"] = same_as_source
    else:
        payload["query"] = args["query"]
    if args["keywords"]:
        payload["keywords"] = args["keywords"]
    if result.matched == 0 and result.best_score is not None:
        payload["best_similarity"] = round(result.best_score, 2)  # ниже порога — «по смыслу ничего достаточно похожего»
    if not_indexed:
        payload["not_indexed"] = not_indexed
        payload["note"] = f"{not_indexed} анализов ещё не проиндексированы для поиска по смыслу и в результат не вошли"
    if warnings:
        payload["warnings"] = warnings

    references: list = []
    if not args["count_only"]:
        shown = [(rows_by_id[i], score_of[i]) for i in shown_ids if i in rows_by_id]
        payload["records"] = [
            _record(r, show_user, similarity=score, same_image=same_image.get(r.id)) for r, score in shown
        ]
        payload["records_shown"] = len(shown)
        references = _build_reference_cards([r for r, _ in shown], show_user)

    return ToolResult(json.dumps(payload, ensure_ascii=False), references)


def search_analyses(user: User, raw_args) -> ToolResult:
    """Точка входа инструмента: аргументы от модели -> результат."""
    args, warnings = normalize_args(raw_args)

    # Категории запрошены, но ни одна не распознана: молча отдать всё без фильтра
    # было бы обманом («были ли анализы с X?» -> «вот всё»). Возвращаем ошибку со
    # списком доступных категорий — модель может повторить вызов.
    if isinstance(raw_args, dict) and raw_args.get("categories") and not args["categories"]:
        return ToolResult(
            json.dumps(
                {
                    "error": "указанные категории не найдены",
                    "available_categories": [name for name, _ in active_categories()],
                },
                ensure_ascii=False,
            )
        )

    if args["query"] or args["similar_to"]:
        return _search_semantic(user, args, warnings)

    try:
        rows = _fetch_rows(user, args)
    except Exception:  # noqa: BLE001 — сбой инструмента не должен ронять чат
        current_app.logger.exception("search_analyses: ошибка выборки")
        return ToolResult(json.dumps({"error": "не удалось получить данные из базы"}, ensure_ascii=False))

    if not rows and args["keywords"] and args["keywords_text"]:
        fallback_args = {
            **args, "keywords": [], "keywords_text": "", "similar_to": None,
            "query": args["keywords_text"][:_MAX_QUERY_CHARS],
        }
        fallback = _search_semantic(user, fallback_args, [*warnings, _KEYWORDS_FALLBACK])
        if "error" not in json.loads(fallback.text):  # эмбеддинги недоступны — честный «0» ниже лучше ошибки
            return fallback

    # Один и тот же файл, загруженный несколько раз, — один снимок: показываем первый по порядку
    # выдачи (при order=newest — самый свежий анализ, при oldest — самый ранний), номера
    # остальных перечисляем в same_image_analyses.
    raw_count = len(rows)
    rep_ids, same_image = image_dedup.dedupe_ordered([(r.id, r.image_hash) for r in rows])
    by_id = {r.id: r for r in rows}
    rows = [by_id[i] for i in rep_ids]

    show_user = user.is_head_admin and not args["own_only"]
    by_risk: dict[str, int] = {}
    for row in rows:
        label = RISK_LABELS.get(row.risk_level, row.risk_level)
        by_risk[label] = by_risk.get(label, 0) + 1

    payload: dict = {
        "scope": "все пользователи" if show_user else "только анализы самого пользователя",
        "order": args["order"],  # newest — от новых к старым, oldest — от старых к новым
        "total_matched": len(rows),
        "by_risk": by_risk,
        "needs_review_count": sum(1 for r in rows if r.needs_human_review),
    }
    if args["keywords"]:
        payload["keywords"] = args["keywords"]  # какие основы слов реально искались
    if raw_count > len(rows):
        payload["duplicates_merged"] = raw_count - len(rows)  # повторные анализы тех же файлов, склеены
    if raw_count >= _SCAN_LIMIT:
        which = "самых старых" if args["order"] == _ORDER_OLDEST else "самых свежих"
        payload["note"] = f"учтены только {_SCAN_LIMIT} {which} записей"
    if warnings:
        payload["warnings"] = warnings

    references: list = []
    if not args["count_only"]:
        shown = rows[: args["limit"]]
        payload["records"] = [_record(r, show_user, same_image=same_image.get(r.id)) for r in shown]
        payload["records_shown"] = len(shown)
        references = _build_reference_cards(shown, show_user)

    return ToolResult(json.dumps(payload, ensure_ascii=False), references)


# ---------------------------------------------------------------------------
# TODO (не реализовано):
#
# 1. Если analysis_results вырастет настолько, что скан signals в Python
#    станет узким местом — вынести категории в таблицу analysis_categories
#    (analysis_id, category_name) с индексом и фильтровать JOIN'ом в SQL.
# 2. Если анализов станет на порядок больше EMBEDDING_SCAN_LIMIT — кэш матрицы
#    векторов в памяти процесса или sqlite-vec (см. vector_search.py).
# ---------------------------------------------------------------------------
