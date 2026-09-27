"""Retrieval-слой для чата: превращает вопрос пользователя вида «какие анализы
были на этой неделе», «сколько высокого риска», «были ли анализы с оружием»
в выборку из таблицы analysis_results и краткую текстовую сводку, которая
подмешивается в system-промпт модели (POST /chat уже поддерживает поле
'system' — см. inference/chat.py: get_chat_system_prompt).

Почему НЕ векторный RAG (embeddings + похожесть), а структурированные
SQL-фильтры:

  - Данные УЖЕ структурированы: risk_level, needs_human_review, signals[].category
    (см. inference/prompts/analyze_system.txt — схема отчёта), created_at, user_id.
    Вопросы вида «анализы за неделю» / «с высоким риском» / «с объектом X» —
    это фильтры по полям, а не поиск по смыслу свободного текста. SQL-фильтр
    даёт точный и объяснимый результат; embedding-поиск для этого — оверкил
    и источник ложных срабатываний (найдёт «похожее», а не то, что спросили).
  - Не нужно поднимать векторную БД/считать embeddings при каждом новом
    анализе, не нужно решать, чем эмбеддить (сервер анализа — LLM для
    изображений, отдельной embedding-модели в проекте нет).
  - Права доступа (свои анализы vs все) естественно ложатся на WHERE user_id=,
    а не на пост-фильтрацию top-k результатов векторного поиска.

Где векторный поиск потенциально пригодится ПОЗЖЕ (сюда не входит, см. TODO
в конце файла) — «нечёткий» поиск по полю description/rationale, когда
формулировка вопроса не совпадает ни с одной категорией и не сводится к
фильтру («покажи анализы, похожие на потасовку у входа»).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from flask import current_app, url_for
from sqlalchemy import select

from .extensions import db
from .models import RISK_LABELS, AnalysisResult, Category, RiskLevel, Status, User
from .utils import local_dt

# ---------------------------------------------------------------------------
# 1. Нужен ли вообще ретрив по этому сообщению
# ---------------------------------------------------------------------------

# Специально широкий и с запасом — ложное срабатывание стоит дёшево (лишний
# кусок system-промпта), а пропуск настоящего аналитического вопроса — дорого
# (модель отвечает "не знаю" или выдумывает). Легко расширять по мере того,
# как станет понятно, какие формулировки реально пишут пользователи.
_TRIGGER_RE = re.compile(
    r"анализ|риск|объект|категор|сигнал|провер|недавн|последн|стат"
    r"истик|сколько|за (сегодня|вчера|неделю|месяц|день)|историст",
    re.IGNORECASE,
)


def wants_analysis_context(message: str) -> bool:
    return bool(_TRIGGER_RE.search(message or ""))


# ---------------------------------------------------------------------------
# 2. Разбор фильтров из текста вопроса
# ---------------------------------------------------------------------------

_RISK_RE = {
    RiskLevel.HIGH: re.compile(r"выс(?:окий|окого|окая)\s+риск|высок\w*\s+уровен", re.IGNORECASE),
    RiskLevel.MEDIUM: re.compile(r"средн\w*\s+риск|средн\w*\s+уровен", re.IGNORECASE),
    RiskLevel.LOW: re.compile(r"низк\w*\s+риск|низк\w*\s+уровен", re.IGNORECASE),
}

_REVIEW_RE = re.compile(r"на\s+проверк|требу\w*\s+проверк|нужна\s+проверка|human.?review", re.IGNORECASE)

_DAYS_N_RE = re.compile(r"за\s+(последн\w*\s+)?(\d{1,3})\s*д", re.IGNORECASE)

_PERIOD_DAYS = (
    (re.compile(r"сегодня", re.IGNORECASE), 1),
    (re.compile(r"вчера", re.IGNORECASE), 2),
    (re.compile(r"недел", re.IGNORECASE), 7),
    (re.compile(r"месяц", re.IGNORECASE), 31),
)


@dataclass
class ParsedQuery:
    risk_level: str | None = None
    review_only: bool = False
    since: datetime | None = None
    category_names: list[str] = field(default_factory=list)


def _parse_since(text: str) -> datetime | None:
    match = _DAYS_N_RE.search(text)
    if match:
        days = int(match.group(2))
        return datetime.now(timezone.utc) - timedelta(days=days)
    for pattern, days in _PERIOD_DAYS:
        if pattern.search(text):
            return datetime.now(timezone.utc) - timedelta(days=days)
    return None


def _matched_categories(text: str) -> list[str]:
    """Ищет в тексте вопроса упоминания активных категорий (по title/name),
    чтобы можно было спросить «были анализы с оружием» без знания того, как
    категория называется технически в signals[].category."""
    text_low = text.lower()
    names: list[str] = []
    categories = db.session.scalars(select(Category).where(Category.is_active.is_(True))).all()
    for cat in categories:
        candidates = {cat.title.strip().lower(), cat.name.strip().lower()}
        if any(c and c in text_low for c in candidates):
            names.append(cat.name)
    return names


def parse_query(text: str) -> ParsedQuery:
    parsed = ParsedQuery()
    for level, pattern in _RISK_RE.items():
        if pattern.search(text):
            parsed.risk_level = level
            break
    parsed.review_only = bool(_REVIEW_RE.search(text))
    parsed.since = _parse_since(text)
    parsed.category_names = _matched_categories(text)
    return parsed


# ---------------------------------------------------------------------------
# 3. Выборка + права доступа
# ---------------------------------------------------------------------------

# Сколько строк максимум разбираем на предмет категорий/агрегатов (защита от
# полного table-скана на очень большой истории) и сколько показываем в сводке
# построчно — числа отдельные, т.к. агрегаты считаем по всей выборке, а
# в текст модели кладём только "хвост" самых свежих.
_SCAN_LIMIT = 500
_DETAIL_LIMIT = 15
_CARD_LIMIT = 6  # сколько кликабельных карточек-ссылок реально прикладываем к ответу
_MAX_CONTEXT_CHARS = 3500


def _in_scope_condition(user: User):
    """Полный доступ ко всей истории — только у главного администратора
    (по явному решению: обычный admin по правам видимости в чате приравнен
    к рядовому пользователю и видит только свои анализы). Если это решение
    поменяется — единственное место для правки: user.is_head_admin ниже."""
    if user.is_head_admin:
        return None
    return AnalysisResult.user_id == user.id


def _fetch_rows(user: User, parsed: ParsedQuery) -> list[AnalysisResult]:
    conditions = [AnalysisResult.status == Status.DONE]
    scope = _in_scope_condition(user)
    if scope is not None:
        conditions.append(scope)
    if parsed.risk_level:
        conditions.append(AnalysisResult.risk_level == parsed.risk_level)
    if parsed.review_only:
        conditions.append(AnalysisResult.needs_human_review.is_(True))
    if parsed.since:
        conditions.append(AnalysisResult.created_at >= parsed.since)

    stmt = (
        select(AnalysisResult)
        .where(*conditions)
        .order_by(AnalysisResult.created_at.desc())
        .limit(_SCAN_LIMIT)
    )
    rows = list(db.session.scalars(stmt).all())

    if parsed.category_names:
        wanted = set(parsed.category_names)
        rows = [r for r in rows if _row_categories(r) & wanted]
    return rows


def _row_categories(row: AnalysisResult) -> set[str]:
    signals = (row.raw_report or {}).get("signals") or []
    return {s.get("category") for s in signals if isinstance(s, dict) and s.get("category")}


# ---------------------------------------------------------------------------
# 4. Текстовая сводка для system-промпта
# ---------------------------------------------------------------------------


def _format_row(row: AnalysisResult, show_user: bool) -> str:
    when = local_dt(row.created_at, "%d.%m.%Y %H:%M")
    cats = ", ".join(sorted(_row_categories(row))) or "—"
    who = f" · пользователь: {row.user.username}" if show_user and row.user else ""
    desc = (row.description or "").strip().replace("\n", " ")
    if len(desc) > 140:
        desc = desc[:139] + "…"
    review = " · требует проверки" if row.needs_human_review else ""
    return f"#{row.id} · {when} · риск: {row.risk_level_display}{review}{who} · категории: {cats} · {desc or '(без описания)'}"


def _build_reference_cards(rows: list[AnalysisResult], show_user: bool) -> list[dict]:
    """Карточки-ссылки на конкретные анализы, которые уйдут в ответ ОТДЕЛЬНО от
    текста (см. AnalysisRAGContext) — рендерятся на фронте как кликабельный
    блок с миниатюрой (chat.js: renderRefCards). Строятся из тех же строк
    БД, что и текстовая сводка выше, а не из текста, который напишет модель —
    так ссылка гарантированно ведёт на существующий анализ и не может
    "поплыть"/сгаллюцинироваться, если бы модель писала URL сама текстом."""
    cards = []
    for row in rows[:_CARD_LIMIT]:
        card = {
            "id": row.id,
            "url": url_for("analyzer.result_detail", pk=row.id),
            "thumb_url": url_for("analyzer.media", filename=row.image_path) if row.image_path else "",
            "label": row.original_name or f"Анализ #{row.id}",
            "risk_level": row.risk_level,
            "risk_label": row.risk_level_display,
            "date": local_dt(row.created_at, "%d.%m.%Y %H:%M"),
        }
        if show_user and row.user:
            card["username"] = row.user.username
        cards.append(card)
    return cards


@dataclass
class AnalysisRAGContext:
    system: str = ""  # подмешивается в system-промпт модели ('' — ретрив не сработал)
    references: list = field(default_factory=list)  # карточки для фронта, см. _build_reference_cards


def build_analysis_context(user: User, message: str) -> AnalysisRAGContext:
    """Главная точка входа: система (для параметра 'system' чата) + карточки
    ссылок на конкретные анализы. Пустой AnalysisRAGContext(), если вопрос не
    похож на аналитический — не тратим контекст модели на каждое сообщение."""
    if not wants_analysis_context(message):
        return AnalysisRAGContext()

    parsed = parse_query(message)
    try:
        rows = _fetch_rows(user, parsed)
    except Exception:  # noqa: BLE001 — сбой ретрива не должен ронять весь чат
        current_app.logger.exception("analysis_query: не удалось построить контекст для чата")
        return AnalysisRAGContext()

    show_user = user.is_head_admin
    total = len(rows)
    by_risk: dict[str, int] = {}
    for row in rows:
        by_risk[row.risk_level] = by_risk.get(row.risk_level, 0) + 1
    review_count = sum(1 for row in rows if row.needs_human_review)

    scope_note = (
        "по всем пользователям (роль позволяет видеть всю историю)"
        if show_user
        else "только по анализам этого пользователя"
    )
    header = [
        f"Данные из базы анализов ({scope_note}).",
        f"Найдено записей по фильтру запроса: {total}"
        + (f" (показаны {min(total, _DETAIL_LIMIT)} самых свежих)" if total > _DETAIL_LIMIT else "")
        + ".",
        "Разбивка по уровню риска: "
        + (", ".join(f"{RISK_LABELS.get(k, k)}={v}" for k, v in by_risk.items()) or "нет данных")
        + f"; требуют проверки: {review_count}.",
    ]

    lines = [_format_row(r, show_user) for r in rows[:_DETAIL_LIMIT]]

    body = "\n".join(header) + ("\n\nПоследние записи:\n" + "\n".join(lines) if lines else "")
    if len(body) > _MAX_CONTEXT_CHARS:
        body = body[: _MAX_CONTEXT_CHARS - 1] + "…"

    system_text = (
        "Ниже — фактическая выборка из базы данных анализов, уже отфильтрованная "
        "под вопрос пользователя и его права доступа. Отвечай на основе ЭТИХ данных, "
        "ничего не выдумывай сверх них; если записей 0 — так и скажи. Не показывай "
        "данные других пользователей, если это не разрешено явно в scope ниже. "
        "Ссылки на сами анализы добавлять не нужно — интерфейс уже приложит их "
        "отдельно под твоим ответом.\n\n" + body
    )

    return AnalysisRAGContext(system=system_text, references=_build_reference_cards(rows, show_user))


# ---------------------------------------------------------------------------
# TODO (по мере надобности, не реализовано сейчас):
#
# 1. Векторный/семантический поиск по description — если появятся вопросы
#    вроде "покажи анализы, похожие на Х", которые не сводятся к фильтру по
#    категории/риску/дате. Тогда: считать embedding description при сохранении
#    AnalysisResult (например через тот же backend, если он умеет отдавать
#    эмбеддинги, либо отдельной лёгкой моделью), хранить в отдельной таблице
#    (sqlite: расширение sqlite-vec; postgres: pgvector) и добавить сюда
#    top-k семантический подзапрос как ещё один источник строк.
# 2. Более точный парсинг дат (диапазоны "с 1 по 15 сентября", конкретные даты) —
#    сейчас есть только относительные периоды.
# 3. Если объём analysis_results вырастет настолько, что _SCAN_LIMIT/скан по
#    JSON-полю signals в Python станет узким местом — вынести категории из
#    signals в отдельную нормализованную таблицу analysis_categories
#    (analysis_id, category_name) с индексом, фильтровать JOIN'ом в SQL, а не
#    построчно в Python.
# ---------------------------------------------------------------------------