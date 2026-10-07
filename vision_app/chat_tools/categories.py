"""Инструмент `manage_category`: чтение и изменение категорий оценивания из чата.

Категории — это правила, по которым сервер анализа ищет сигналы на изображениях: вся таблица
уходит в /analyze при КАЖДОМ анализе (Category.to_overlay_dict), то есть текст категории — это
часть промпта для vision-модели и он влияет на анализы всех пользователей. Поэтому защита здесь
та же, что у manage_user, плюс несколько правил, специфичных для категорий:

  1. Чтение (list / get) выполняется сразу. Любое изменение (create / update / enable / disable /
     delete) — только заявка ChatAction со статусом pending и карточка с кнопками «Подтвердить» /
     «Отмена». Модель ничего не меняет сама.
  2. Права — как в панели (`staff_required`): администраторы панели, не только главный.
     Проверяются при создании заявки и ПОВТОРНО при подтверждении.
  3. Новая категория всегда создаётся выключенной (черновик): в анализах она не участвует, пока её
     не включат отдельной заявкой.
  4. Валидация — теми же правилами, что у формы панели (CategoryForm): формат и уникальность имени,
     обязательные поля. Техническое имя после создания не меняется: по нему категория записана в
     уже сохранённых анализах.
  5. Заявка на update хранит прежние значения полей (`before`): если категорию тем временем
     изменили (в панели или другой заявкой), подтверждение не перезапишет чужую правку. Они же —
     запись для отката; при удалении в заявке остаётся полный снимок категории.
  6. Удаление подтверждается ещё и вводом имени категории в карточке.
  7. runner.py не даёт готовить изменения в ходе, где модель уже читала чужие данные (анализы,
     пользователей, картинки): в них может быть текст с «инструкциями». Чтение самих категорий этот
     запрет не включает — их пишут администраторы.

ChatAction.action хранит имя с префиксом `cat_` (колонка String(16)); для create target_id = 0.
Таблица chat_actions — заодно журнал, как и для действий над пользователями.
"""

from __future__ import annotations

import json
from datetime import timedelta

from flask import current_app, url_for
from sqlalchemy import func, select, update
from werkzeug.datastructures import MultiDict

from ..config import conf
from ..extensions import db
from ..forms import CategoryForm
from ..models import Category, ChatAction, ChatActionStatus, utcnow
from ..utils import local_dt
from .manage import _aware, _error, _expire_stale, _finish
from .users import ToolResult

TOOL_NAME = "manage_category"

ACTION_LIST = "list"
ACTION_GET = "get"
ACTION_CREATE = "create"
ACTION_UPDATE = "update"
ACTION_ENABLE = "enable"
ACTION_DISABLE = "disable"
ACTION_DELETE = "delete"
READ_ACTIONS = (ACTION_LIST, ACTION_GET)
WRITE_ACTIONS = (ACTION_CREATE, ACTION_UPDATE, ACTION_ENABLE, ACTION_DISABLE, ACTION_DELETE)
ACTIONS = READ_ACTIONS + WRITE_ACTIONS

DB_PREFIX = "cat_"
DB_ACTIONS = frozenset(DB_PREFIX + a for a in WRITE_ACTIONS)

EDIT_FIELDS = ("title", "summary", "full", "compact", "position")
CREATE_FIELDS = ("title", "summary", "full", "compact")
_FIELD_LABELS = {
    "title": "Название",
    "summary": "Summary",
    "full": "Full",
    "compact": "Compact",
    "position": "Позиция",
}
# Пределы длины: текст категории попадает в промпт и целиком показывается в карточке.
_LIMITS = {"title": 150, "summary": 800, "full": 4000, "compact": 2500}
_TITLES = {
    ACTION_CREATE: "Создать категорию",
    ACTION_UPDATE: "Изменить категорию",
    ACTION_ENABLE: "Включить категорию",
    ACTION_DISABLE: "Выключить категорию",
    ACTION_DELETE: "Удалить категорию",
}
_OLD_CLIP = 300  # прежнее значение в карточке показываем сокращённо, новое — целиком


def is_category_action(db_action: str) -> bool:
    return db_action in DB_ACTIONS


def _short(db_action: str) -> str:
    return db_action[len(DB_PREFIX):] if db_action.startswith(DB_PREFIX) else db_action


def _clip(text, limit: int) -> str:
    text = str(text or "").strip()
    return text if len(text) <= limit else text[:limit].rstrip() + "… (сокращено)"


def _is_staff(user) -> bool:
    return user is not None and bool(getattr(user, "is_panel_staff", False))


# ---------------------------------------------------------------------------
# Поиск и разбор аргументов
# ---------------------------------------------------------------------------


def find_category(ref) -> Category | None:
    """Категория по точному техническому имени (без учёта регистра) или по id. Частичных
    совпадений нет: изменять «похожую» категорию нельзя."""
    if isinstance(ref, bool) or ref is None:
        return None
    if isinstance(ref, int):
        return db.session.get(Category, ref)
    text = str(ref).strip().lstrip("#")
    if not text:
        return None
    found = db.session.scalar(select(Category).where(func.lower(Category.name) == text.lower()))
    if found is None and text.isdigit():
        found = db.session.get(Category, int(text))
    return found


def _form(data: dict, current_id: int | None = None) -> CategoryForm:
    """Та же форма, что в панели; CSRF не нужен — это не HTTP-отправка."""
    return CategoryForm(formdata=MultiDict(data), current_id=current_id, meta={"csrf": False})


def _text_arg(raw, key: str) -> tuple[str | None, str | None]:
    """(значение, ошибка) для строкового поля."""
    value = raw.get(key)
    if value is None or isinstance(value, (dict, list, bool)):
        return None, f"поле '{key}' должно быть строкой"
    text = str(value).strip()
    limit = _LIMITS.get(key)
    if limit and len(text) > limit:
        return None, f"поле '{key}' длиннее {limit} символов ({len(text)}): сократи текст"
    return text, None


def _form_problems(form: CategoryForm, fields) -> str | None:
    form.validate()
    problems = [msg for field, errs in form.errors.items() if field in fields for msg in errs]
    return "; ".join(dict.fromkeys(problems)) if problems else None


def _check_create(raw: dict) -> tuple[dict, str | None]:
    params: dict = {}
    for key in ("name",) + CREATE_FIELDS:
        value, err = _text_arg(raw, key) if key != "name" else _name_arg(raw)
        if err:
            return {}, err
        params[key] = value
    form = _form({**params, "full_extra": "", "compact_extra": "", "position": ""})
    err = _form_problems(form, ("name",) + CREATE_FIELDS)
    if err:
        return {}, err
    return {k: (getattr(form, k).data or "").strip() for k in ("name",) + CREATE_FIELDS}, None


def _name_arg(raw: dict) -> tuple[str | None, str | None]:
    value = raw.get("name")
    if value is None or isinstance(value, (dict, list, bool)) or not str(value).strip():
        return None, "для create нужен name — техническое имя (латиница, цифры, «_»)"
    return str(value).strip(), None


def _clean_changes(raw) -> tuple[dict, list[str]]:
    warnings: list[str] = []
    if not isinstance(raw, dict):
        return {}, ["changes должен быть объектом {поле: значение}"]
    changes: dict = {}
    for key, value in raw.items():
        if key not in EDIT_FIELDS:
            warnings.append(f"поле '{key}' менять нельзя, доступно: " + ", ".join(EDIT_FIELDS))
            continue
        if key == "position":
            if isinstance(value, bool) or not isinstance(value, (int, str)) or not str(value).strip().lstrip("-").isdigit():
                warnings.append("position должен быть целым числом")
                continue
            changes[key] = int(str(value).strip())
            continue
        text, err = _text_arg({key: value}, key)
        if err:
            warnings.append(err)
            continue
        changes[key] = text
    return changes, warnings


def _current_values(cat: Category) -> dict:
    return {
        "title": cat.title or "", "summary": cat.summary or "", "full": cat.full or "",
        "compact": cat.compact or "", "position": cat.position,
    }


def _check_update(cat: Category, changes: dict) -> tuple[dict, str | None]:
    """(очищенные изменения, ошибка). Валидируются только изменяемые поля."""
    if not changes:
        return {}, "не указано, что менять (changes)"
    data = {
        "name": cat.name, **{k: v for k, v in _current_values(cat).items() if k != "position"},
        "full_extra": cat.full_extra or "", "compact_extra": cat.compact_extra or "",
        "position": str(cat.position),
    }
    data.update({k: str(v) for k, v in changes.items()})
    err = _form_problems(_form(data, current_id=cat.id), tuple(changes))
    if err:
        return {}, err
    current = _current_values(cat)
    cleaned = {k: (v.strip() if isinstance(v, str) else v) for k, v in changes.items()}
    cleaned = {k: v for k, v in cleaned.items() if v != current[k]}
    if not cleaned:
        return {}, "указанные значения уже совпадают с текущими"
    return cleaned, None


# ---------------------------------------------------------------------------
# Чтение
# ---------------------------------------------------------------------------


def _list() -> ToolResult:
    rows = db.session.scalars(select(Category).order_by(Category.position, Category.id)).all()
    items = [
        {
            "id": c.id, "name": c.name, "title": c.title, "is_active": c.is_active,
            "position": c.position, "summary": _clip(c.summary, 300),
        }
        for c in rows
    ]
    return ToolResult(json.dumps({"total": len(items), "categories": items}, ensure_ascii=False))


def _get(cat: Category | None) -> ToolResult:
    if cat is None:
        return _error("категория не найдена (нужно точное имя из list или id)")
    payload = {
        "id": cat.id, "name": cat.name, "title": cat.title, "is_active": cat.is_active,
        "position": cat.position, "summary": cat.summary, "full": cat.full, "compact": cat.compact,
        "has_extra_blocks": bool((cat.full_extra or "").strip() or (cat.compact_extra or "").strip()),
        "has_examples": bool((cat.example_en or "").strip() or (cat.example_ru or "").strip()),
    }
    return ToolResult(json.dumps(payload, ensure_ascii=False))


# ---------------------------------------------------------------------------
# Проверка прав (при создании заявки и при выполнении)
# ---------------------------------------------------------------------------


def check_allowed(admin, cat: Category | None, action: str, params: dict) -> str | None:
    """Текст ошибки или None. Одни и те же правила для заявки и для её выполнения."""
    if not _is_staff(admin):
        return "управлять категориями могут только администраторы панели"
    if action not in WRITE_ACTIONS:
        return f"неизвестное действие '{action}', доступно: " + ", ".join(WRITE_ACTIONS)
    if action == ACTION_CREATE:
        return None
    if cat is None:
        return "категория не найдена (нужно точное имя из list или id)"
    if action == ACTION_ENABLE and cat.is_active:
        return "категория уже включена"
    if action == ACTION_DISABLE and not cat.is_active:
        return "категория уже выключена"
    return None


def _snapshot(cat: Category) -> dict:
    return {
        "name": cat.name, "title": cat.title, "summary": cat.summary, "full": cat.full,
        "compact": cat.compact, "full_extra": cat.full_extra, "compact_extra": cat.compact_extra,
        "example_en": cat.example_en, "example_ru": cat.example_ru, "position": cat.position,
        "is_active": cat.is_active,
    }


def _describe(action: str, name: str, params: dict) -> tuple[str, list[str]]:
    """(краткое описание, строки-детали для карточки)."""
    details: list[str] = []
    if action == ACTION_CREATE:
        details.append(f"Название: {params['title'] or '—'}")
        details.append(f"Summary: {params['summary']}")
        details.append(f"Full: {params['full']}")
        details.append(f"Compact: {params['compact']}")
        details.append("Будет создана ВЫКЛЮЧЕННОЙ (черновик): в анализах не участвует, пока её не включат отдельной заявкой")
        return f"Создать категорию «{name}»", details
    if action == ACTION_UPDATE:
        before = params.get("before") or {}
        for key, new in (params.get("changes") or {}).items():
            old = before.get(key, "")
            old_text = str(old) if key == "position" else _clip(old, _OLD_CLIP)
            details.append(f"{_FIELD_LABELS[key]}: {old_text or '—'} → {new if new != '' else '—'}")
        details.append("Изменение повлияет на анализы всех пользователей, если категория включена")
        return f"Изменить категорию «{name}»", details
    if action == ACTION_ENABLE:
        details.append("Категория начнёт передаваться на сервер анализа и влиять на анализы всех пользователей")
        return f"Включить категорию «{name}»", details
    if action == ACTION_DISABLE:
        details.append("Категория перестанет участвовать в новых анализах; сама она сохранится и её можно включить снова")
        return f"Выключить категорию «{name}»", details
    details.append("Категория будет удалена безвозвратно; в уже сохранённых анализах её название останется")
    details.append("Если нужно лишь убрать её из анализа — выключите вместо удаления")
    return f"Удалить категорию «{name}»", details


# ---------------------------------------------------------------------------
# Карточка под ответом
# ---------------------------------------------------------------------------


def action_card(a: ChatAction) -> dict:
    """Карточка заявки (рисует static/js/chat.js по kind == "action"); поля те же, что у manage.action_card."""
    action = _short(a.action)
    exists = bool(a.target_id) and db.session.get(Category, a.target_id) is not None
    return {
        "kind": "action",
        "id": a.id,
        "action": a.action,
        "title": _TITLES.get(action, action),
        "summary": a.summary,
        "details": list((a.params or {}).get("details") or []),
        "status": a.status,
        "result": a.result or "",
        "danger": action == ACTION_DELETE,
        "type_to_confirm": a.target_label if action == ACTION_DELETE else "",
        "confirm_hint": "Для удаления введите имя категории: ",
        "target_url": url_for("panel.category_edit", pk=a.target_id) if exists else "",
        "target_link_label": "страница категории",
        "state_url": url_for("chat.action_state", action_id=a.id),
        "confirm_url": url_for("chat.action_confirm", action_id=a.id),
        "cancel_url": url_for("chat.action_cancel", action_id=a.id),
        "expires": local_dt(a.expires_at, "chat") if a.expires_at else "",
    }


# ---------------------------------------------------------------------------
# Создание заявки (то, что вызывает модель)
# ---------------------------------------------------------------------------


def manage_category(user, raw_args, session_id: int | None = None) -> ToolResult:
    """Точка входа: чтение возвращается сразу, изменения — только заявкой на подтверждение."""
    if not _is_staff(user):
        return _error("инструмент доступен только администраторам панели")
    if not isinstance(raw_args, dict):
        return _error("args должен быть объектом")

    action = str(raw_args.get("action") or "").strip().lower()
    if action not in ACTIONS:
        return _error("нужен action: " + ", ".join(ACTIONS))
    if action == ACTION_LIST:
        return _list()

    cat = None if action == ACTION_CREATE else find_category(raw_args.get("category"))
    if action == ACTION_GET:
        return _get(cat)

    params: dict = {}
    warnings: list[str] = []
    name = cat.name if cat is not None else ""
    if action == ACTION_CREATE:
        params, err = _check_create(raw_args)
        if err:
            return _error(err)
        name = params["name"]
    elif action == ACTION_UPDATE and cat is not None:
        changes, warnings = _clean_changes(raw_args.get("changes"))
        changes, err = _check_update(cat, changes)
        if err:
            return _error(err, **({"warnings": warnings} if warnings else {}))
        current = _current_values(cat)
        params = {"changes": changes, "before": {k: current[k] for k in changes}}
    elif action == ACTION_DELETE and cat is not None:
        params = {"snapshot": _snapshot(cat)}

    err = check_allowed(user, cat, action, params)
    if err:
        return _error(err)

    db_action = DB_PREFIX + action
    target_id = cat.id if cat is not None else 0
    _expire_stale(user.id)
    pending = db.session.scalars(
        select(ChatAction).where(ChatAction.admin_id == user.id, ChatAction.status == ChatActionStatus.PENDING)
    ).all()
    same = next(
        (p for p in pending if p.action == db_action and p.target_id == target_id and p.target_label == name
         and {k: v for k, v in (p.params or {}).items() if k != "details"} == params),
        None,
    )
    if same is None and len(pending) >= conf("CHAT_ACTION_MAX_PENDING"):
        db.session.commit()
        return _error(
            f"уже есть {len(pending)} неподтверждённых заявок: подтвердите или отмените их в карточках выше, "
            "затем повторите"
        )

    summary, details = _describe(action, name, params)
    if same is None:
        same = ChatAction(
            admin_id=user.id, session_id=session_id, action=db_action, target_id=target_id,
            target_label=name, params={**params, "details": details}, summary=summary[:400],
            status=ChatActionStatus.PENDING,
            expires_at=utcnow() + timedelta(minutes=conf("CHAT_ACTION_TTL_MINUTES")),
        )
        db.session.add(same)
    db.session.commit()

    payload = {
        "status": "awaiting_confirmation",
        "action_id": same.id,
        "summary": summary,
        "note": (
            "NOTHING HAS BEEN CHANGED YET. The request is shown to the admin as a card with Confirm and "
            f"Cancel buttons and stays valid for {conf('CHAT_ACTION_TTL_MINUTES')} minutes. Tell the admin "
            "that the action is prepared and waits for confirmation."
        ),
    }
    if action == ACTION_CREATE:
        payload["note"] += " The category will be created disabled; enabling it is a separate request when the admin asks."
    if action == ACTION_DELETE:
        payload["note"] += " For delete, the admin must also type the category name in the card."
    if warnings:
        payload["warnings"] = warnings
    return ToolResult(json.dumps(payload, ensure_ascii=False), references=[action_card(same)])


# ---------------------------------------------------------------------------
# Подтверждение (вызывает manage.confirm_action по нажатию кнопки, а не модель)
# ---------------------------------------------------------------------------


def confirm_action(admin, a: ChatAction, typed_text: str = "") -> tuple[bool, str]:
    """Выполняет заявку. Возвращает (успех, сообщение). Права и параметры проверяются заново."""
    if a.admin_id != admin.id:
        return False, "заявка принадлежит другому администратору"
    if a.status == ChatActionStatus.PENDING and _aware(a.expires_at) < utcnow():
        _finish(a, ChatActionStatus.EXPIRED, "Срок подтверждения истёк. Попросите подготовить действие заново.")
        return False, a.result

    action = _short(a.action)
    if action == ACTION_DELETE and (typed_text or "").strip() != a.target_label:
        return False, f"для удаления введите имя «{a.target_label}» точно как оно написано"

    # Атомарный захват: двойной клик или две вкладки не выполнят заявку дважды.
    claimed = db.session.execute(
        update(ChatAction)
        .where(ChatAction.id == a.id, ChatAction.status == ChatActionStatus.PENDING)
        .values(status=ChatActionStatus.RUNNING)
    ).rowcount
    db.session.commit()
    db.session.refresh(a)
    if not claimed:
        return False, "заявка уже обработана"

    params = {k: v for k, v in (a.params or {}).items() if k != "details"}
    cat = db.session.get(Category, a.target_id) if a.target_id else None
    err = check_allowed(admin, cat, action, params)
    if err:
        _finish(a, ChatActionStatus.FAILED, f"Не выполнено: {err}.")
        return False, a.result

    try:
        message, err = _apply(action, cat, params)
    except Exception:  # noqa: BLE001
        db.session.rollback()
        current_app.logger.exception("chat action %s: сбой выполнения", a.id)
        _finish(a, ChatActionStatus.FAILED, "Не выполнено: внутренняя ошибка.")
        return False, a.result
    if err:
        db.session.rollback()
        _finish(a, ChatActionStatus.FAILED, f"Не выполнено: {err}.")
        return False, a.result

    current_app.logger.info(
        "chat action %s выполнено: %s (id %s) → %s категории %s (id %s)",
        a.id, admin.username, admin.id, action, a.target_label, a.target_id,
    )
    _finish(a, ChatActionStatus.DONE, message)
    return True, message


def _apply(action: str, cat: Category | None, params: dict) -> tuple[str, str | None]:
    """(сообщение, ошибка). Ошибка — ожидаемый отказ (категорию изменили, имя занято)."""
    if action == ACTION_CREATE:
        form = _form({**params, "full_extra": "", "compact_extra": "", "position": ""})
        err = _form_problems(form, ("name",) + CREATE_FIELDS)
        if err:
            return "", err
        row = Category(
            name=params["name"], title=params["title"], summary=params["summary"], full=params["full"],
            compact=params["compact"], position=db.session.scalar(select(func.count(Category.id))) or 0,
            is_active=False,
        )
        db.session.add(row)
        db.session.commit()
        return f"Категория «{row.name}» создана выключенной (черновик). Включить её можно отдельной заявкой.", None

    name = cat.name
    if action == ACTION_UPDATE:
        before = params.get("before") or {}
        current = _current_values(cat)
        if any(current.get(k) != v for k, v in before.items()):
            return "", "категория изменилась после подготовки заявки — попросите подготовить изменение заново"
        changes, err = _check_update(cat, params.get("changes") or {})
        if err:
            return "", err
        for key, value in changes.items():
            setattr(cat, key, value)
        db.session.commit()
        return f"Категория «{name}» обновлена: " + ", ".join(_FIELD_LABELS[k] for k in changes) + ".", None
    if action == ACTION_ENABLE:
        cat.is_active = True
        db.session.commit()
        return f"Категория «{name}» включена.", None
    if action == ACTION_DISABLE:
        cat.is_active = False
        db.session.commit()
        return f"Категория «{name}» выключена.", None
    db.session.delete(cat)
    db.session.commit()
    return f"Категория «{name}» удалена.", None
