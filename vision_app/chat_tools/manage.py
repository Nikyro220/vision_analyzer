"""Инструмент `manage_user`: изменение, смена роли, блокировка и удаление пользователей из чата.

Только для главного администратора — и модель НИЧЕГО не выполняет сама. Защита построена так:

  1. Инструмент лишь создаёт заявку (ChatAction, статус pending) и показывает под ответом карточку
     с кнопками «Подтвердить» / «Отмена». Изменения происходят только после нажатия кнопки
     (POST /chat/actions/<id>/confirm: вход, CSRF, тот же администратор, срок заявки не истёк).
     Ни один текст в данных (описание анализа, ник пользователя, надпись на картинке) не может
     «нажать» кнопку за человека.
  2. Права проверяются и при создании заявки, и повторно при выполнении — теми же правилами, что и
     в панели (User.can_manage / assignable_roles): нельзя трогать себя, роль выдаётся только из
     разрешённых.
  3. Удаление аккаунта подтверждается ещё и вводом его логина в карточке (в панели — SQL-командой).
  4. Заявка живёт CHAT_ACTION_TTL_MINUTES минут; неподтверждённых заявок у админа не больше
     CHAT_ACTION_MAX_PENDING.
  5. runner.py не даёт вызвать инструмент в ходе, где модель уже читала данные пользователей или
     анализов (там может быть чужой текст с «инструкциями»), и в служебных ходах без сообщения админа.

Таблица chat_actions — заодно журнал: кто, что, над кем и чем закончилось.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from flask import current_app, url_for
from flask_babel import gettext, lazy_gettext
from sqlalchemy import func, select, update
from werkzeug.datastructures import MultiDict

from ..config import conf
from ..extensions import db
from ..forms import AccountForm
from ..models import (
    DISPLAY_STYLE_LABELS,
    ROLE_LABELS,
    ChatAction,
    ChatActionStatus,
    DisplayStyle,
    Role,
    User,
    utcnow,
)
from ..utils import local_dt
from .users import ToolResult

TOOL_NAME = "manage_user"

ACTION_SET_ROLE = "set_role"
ACTION_BLOCK = "block"
ACTION_UNBLOCK = "unblock"
ACTION_DELETE = "delete"
ACTION_EDIT = "edit"
ACTIONS = (ACTION_EDIT, ACTION_SET_ROLE, ACTION_BLOCK, ACTION_UNBLOCK, ACTION_DELETE)

EDIT_FIELDS = ("username", "email", "family_name", "given_name", "middle_name", "nickname")
_FIELD_LABELS = {
    "username": lazy_gettext("account.field.login"),
    "email": "Email",
    "family_name": lazy_gettext("forms.account.family_name"),
    "given_name": lazy_gettext("forms.account.given_name"),
    "middle_name": lazy_gettext("forms.account.middle_name"),
    "nickname": lazy_gettext("account.field.nickname_title"),
}
_TITLES = {
    ACTION_EDIT: lazy_gettext("account.edit.title"),
    ACTION_SET_ROLE: lazy_gettext("chat.action.title.set_role"),
    ACTION_BLOCK: lazy_gettext("panel.user.block"),
    ACTION_UNBLOCK: lazy_gettext("panel.user.unblock"),
    ACTION_DELETE: lazy_gettext("account.delete"),
}


def _error(message: str, **extra) -> ToolResult:
    return ToolResult(json.dumps({"error": message, **extra}, ensure_ascii=False))


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Разбор аргументов и поиск цели
# ---------------------------------------------------------------------------


def find_target(ref) -> User | None:
    """Пользователь по логину (точно, без учёта регистра) или по id. Частичных совпадений нет:
    для необратимых действий «похожий» пользователь недопустим."""
    if isinstance(ref, bool) or ref is None:
        return None
    if isinstance(ref, int):
        return db.session.get(User, ref)
    text = str(ref).strip().lstrip("#")
    if not text:
        return None
    found = db.session.scalar(select(User).where(func.lower(User.username) == text.lower()))
    if found is None and text.isdigit():
        found = db.session.get(User, int(text))
    return found


def _clean_changes(raw) -> tuple[dict, list[str]]:
    warnings: list[str] = []
    if not isinstance(raw, dict):
        return {}, ["changes должен быть объектом {поле: значение}"]
    changes: dict = {}
    for key, value in raw.items():
        if key not in EDIT_FIELDS:
            warnings.append(f"поле '{key}' менять нельзя, доступно: " + ", ".join(EDIT_FIELDS))
            continue
        if value is None or isinstance(value, (dict, list, bool)):
            warnings.append(f"значение поля '{key}' должно быть строкой")
            continue
        changes[key] = str(value).strip()[:254]
    return changes, warnings


def _edit_form(target: User, changes: dict) -> AccountForm:
    """Те же правила, что у формы редактирования в панели (AccountForm): формат логина, email,
    уникальность логина, длины. Незатронутые поля берутся из текущих данных."""
    style = target.display_style if target.display_style in dict(DISPLAY_STYLE_LABELS) else DisplayStyle.NICKNAME
    data = {
        "username": target.username, "email": target.email or "", "family_name": target.family_name or "",
        "given_name": target.given_name or "", "middle_name": target.middle_name or "",
        "nickname": target.nickname or "", "display_style": style, "color": "",
    }
    data.update(changes)
    return AccountForm(formdata=MultiDict(data), current_id=target.id, meta={"csrf": False})


def _check_edit(target: User, changes: dict) -> tuple[dict, str | None]:
    """(очищенные изменения, ошибка). Валидируются только изменяемые поля."""
    if not changes:
        return {}, "не указано, что менять (changes)"
    form = _edit_form(target, changes)
    form.validate()
    problems = [str(msg) for field, errs in form.errors.items() if field in changes for msg in errs]
    if problems:
        return {}, "; ".join(dict.fromkeys(problems))
    cleaned = {k: (getattr(form, k).data or "").strip() for k in changes}
    cleaned = {k: v for k, v in cleaned.items() if v != ((getattr(target, k) or "").strip())}
    if not cleaned:
        return {}, "указанные значения уже совпадают с текущими"
    return cleaned, None


# ---------------------------------------------------------------------------
# Проверка прав (при создании заявки и при выполнении)
# ---------------------------------------------------------------------------


def check_allowed(admin: User, target: User | None, action: str, params: dict) -> str | None:
    """Текст ошибки или None. Одни и те же правила для заявки и для её выполнения."""
    if not admin.is_head_admin:
        return gettext("chat.action.error.head_only")
    if target is None:
        return gettext("chat.action.error.user_not_found")
    if target.id == admin.id:
        return gettext("chat.action.error.self")
    if not admin.can_manage(target):
        return gettext("chat.action.error.no_rights")
    if action == ACTION_SET_ROLE:
        role = params.get("role")
        if role not in admin.assignable_roles():
            return gettext("chat.action.error.role_forbidden", roles=", ".join(admin.assignable_roles()))
        if role == target.role:
            return gettext("chat.action.error.same_role", role=str(ROLE_LABELS.get(role, role)))
    elif action == ACTION_BLOCK and target.role == Role.BLOCKED:
        return gettext("chat.action.error.already_blocked")
    elif action == ACTION_UNBLOCK and target.role != Role.BLOCKED:
        return gettext("chat.action.error.not_blocked")
    elif action == ACTION_EDIT:
        _, err = _check_edit(target, params.get("changes") or {})
        return err
    elif action not in ACTIONS:
        return f"неизвестное действие '{action}', доступно: " + ", ".join(ACTIONS)
    return None


def _describe(target: User, action: str, params: dict) -> tuple[str, list[str]]:
    """(краткое описание, строки-детали для карточки)."""
    name = target.username
    details: list[str] = []
    if action == ACTION_SET_ROLE:
        new = str(ROLE_LABELS.get(params["role"], params["role"]))
        details.append(gettext("chat.action.detail.role", old=str(ROLE_LABELS.get(target.role, target.role)), new=new))
        return gettext("chat.action.summary.set_role", name=name, new=new), details
    if action == ACTION_BLOCK:
        if target.role in (Role.ADMIN, Role.HEAD_ADMIN):
            details.append(gettext("chat.action.detail.role_lost", role=str(ROLE_LABELS[target.role])))
        return gettext("chat.action.summary.block", name=name), details
    if action == ACTION_UNBLOCK:
        details.append(gettext("chat.action.detail.unblock_role"))
        return gettext("chat.action.summary.unblock", name=name), details
    if action == ACTION_DELETE:
        details.append(gettext("chat.action.detail.delete"))
        return gettext("chat.action.summary.delete", name=name), details
    for key, value in (params.get("changes") or {}).items():
        old = getattr(target, key, "") or ""
        details.append(f"{_FIELD_LABELS[key]}: {old or '—'} → {value or '—'}")
    return gettext("chat.action.summary.edit", name=name), details


# ---------------------------------------------------------------------------
# Карточка под ответом
# ---------------------------------------------------------------------------


def action_card(a: ChatAction) -> dict:
    """Карточка заявки (рисует static/js/chat.js по kind == "action"). Содержит текущий статус;
    фронт при отображении перезапрашивает его по state_url, чтобы старые карточки не врали."""
    from . import categories  # отложенный импорт: categories сам импортирует отсюда хелперы

    if categories.is_category_action(a.action):
        return categories.action_card(a)
    target = db.session.get(User, a.target_id)
    params = a.params or {}
    if target is not None and a.status == ChatActionStatus.PENDING:
        try:
            _, details = _describe(target, a.action, params)
        except Exception:  # noqa: BLE001 — карточка не должна падать из-за устаревших параметров
            details = []
    else:
        details = list(params.get("details") or [])
    return {
        "kind": "action",
        "id": a.id,
        "action": a.action,
        "title": str(_TITLES.get(a.action, a.action)),
        "summary": a.summary,
        "details": details,
        "status": a.status,
        "result": a.result or "",
        "danger": a.action in (ACTION_DELETE, ACTION_BLOCK),
        "type_to_confirm": a.target_label if a.action == ACTION_DELETE else "",
        "target_url": url_for("panel.user_detail", pk=a.target_id) if target is not None else "",
        "state_url": url_for("chat.action_state", action_id=a.id),
        "confirm_url": url_for("chat.action_confirm", action_id=a.id),
        "cancel_url": url_for("chat.action_cancel", action_id=a.id),
        "expires": local_dt(a.expires_at, "chat") if a.expires_at else "",
    }


# ---------------------------------------------------------------------------
# Создание заявки (то, что вызывает модель)
# ---------------------------------------------------------------------------


def _expire_stale(admin_id: int) -> None:
    db.session.execute(
        update(ChatAction)
        .where(ChatAction.admin_id == admin_id, ChatAction.status == ChatActionStatus.PENDING,
               ChatAction.expires_at < utcnow())
        .values(status=ChatActionStatus.EXPIRED, resolved_at=utcnow(), result=gettext("chat.action.expired"))
    )


def manage_user(admin: User, raw_args, session_id: int | None = None) -> ToolResult:
    """Точка входа: аргументы от модели → заявка на подтверждение. Ничего не изменяет."""
    if not admin.is_head_admin:
        return _error("инструмент доступен только главному администратору")
    if not isinstance(raw_args, dict):
        return _error("args должен быть объектом")

    action = str(raw_args.get("action") or "").strip().lower()
    if action not in ACTIONS:
        return _error("нужен action: " + ", ".join(ACTIONS))

    target = find_target(raw_args.get("user"))
    params: dict = {}
    warnings: list[str] = []
    if action == ACTION_SET_ROLE:
        params["role"] = str(raw_args.get("role") or "").strip().lower()
    elif action == ACTION_EDIT:
        changes, warnings = _clean_changes(raw_args.get("changes"))
        if target is not None:
            changes, err = _check_edit(target, changes)
            if err:
                return _error(err, **({"warnings": warnings} if warnings else {}))
        params["changes"] = changes

    err = check_allowed(admin, target, action, params)
    if err:
        return _error(err)

    _expire_stale(admin.id)
    pending = db.session.scalars(
        select(ChatAction).where(ChatAction.admin_id == admin.id, ChatAction.status == ChatActionStatus.PENDING)
    ).all()
    same = next(
        (p for p in pending if p.action == action and p.target_id == target.id
         and {k: v for k, v in (p.params or {}).items() if k != "details"} == params),
        None,
    )
    if same is None and len(pending) >= conf("CHAT_ACTION_MAX_PENDING"):
        db.session.commit()
        return _error(
            f"уже есть {len(pending)} неподтверждённых заявок: подтвердите или отмените их в карточках выше, "
            "затем повторите"
        )

    summary, details = _describe(target, action, params)
    if same is None:
        same = ChatAction(
            admin_id=admin.id, session_id=session_id, action=action, target_id=target.id,
            target_label=target.username, params={**params, "details": details}, summary=summary[:400],
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
    if action == ACTION_DELETE:
        payload["note"] += " For delete, the admin must also type the username in the card."
    if warnings:
        payload["warnings"] = warnings
    return ToolResult(json.dumps(payload, ensure_ascii=False), references=[action_card(same)])


# ---------------------------------------------------------------------------
# Подтверждение / отмена (вызывает blueprint по нажатию кнопки, а не модель)
# ---------------------------------------------------------------------------


def _finish(a: ChatAction, status: str, result: str) -> None:
    a.status = status
    a.result = result[:2000]
    a.resolved_at = utcnow()
    db.session.commit()


def cancel_action(admin: User, a: ChatAction) -> tuple[bool, str]:
    if a.admin_id != admin.id:
        return False, gettext("chat.action.error.foreign")
    if a.status != ChatActionStatus.PENDING:
        return False, gettext("chat.action.error.processed")
    _finish(a, ChatActionStatus.CANCELLED, gettext("chat.action.cancelled"))
    return True, a.result


def confirm_action(admin: User, a: ChatAction, typed_text: str = "") -> tuple[bool, str]:
    """Выполняет заявку. Возвращает (успех, сообщение). Права и параметры проверяются заново."""
    from ..history import delete_user_account
    from . import categories

    if categories.is_category_action(a.action):  # заявки на категории выполняет их собственный модуль
        return categories.confirm_action(admin, a, typed_text)
    if a.admin_id != admin.id:
        return False, gettext("chat.action.error.foreign")
    if a.status == ChatActionStatus.PENDING and _aware(a.expires_at) < utcnow():
        _finish(a, ChatActionStatus.EXPIRED, gettext("chat.action.expired_retry"))
        return False, a.result

    target = db.session.get(User, a.target_id)
    params = {k: v for k, v in (a.params or {}).items() if k != "details"}
    if a.action == ACTION_DELETE and target is not None and (typed_text or "").strip() != target.username:
        return False, gettext("chat.action.error.type_login", username=target.username)

    # Атомарный захват: двойной клик или две вкладки не выполнят заявку дважды.
    claimed = db.session.execute(
        update(ChatAction)
        .where(ChatAction.id == a.id, ChatAction.status == ChatActionStatus.PENDING)
        .values(status=ChatActionStatus.RUNNING)
    ).rowcount
    db.session.commit()
    db.session.refresh(a)
    if not claimed:
        return False, gettext("chat.action.error.processed")

    err = check_allowed(admin, target, a.action, params)
    if err:
        _finish(a, ChatActionStatus.FAILED, gettext("chat.action.failed", error=err))
        return False, a.result

    try:
        message = _apply(target, a.action, params, delete_user_account)
    except Exception:  # noqa: BLE001
        db.session.rollback()
        current_app.logger.exception("chat action %s: сбой выполнения", a.id)
        _finish(a, ChatActionStatus.FAILED, gettext("chat.action.failed_internal"))
        return False, a.result

    current_app.logger.info(
        "chat action %s выполнено: админ %s (id %s) → %s над пользователем %s (id %s)",
        a.id, admin.username, admin.id, a.action, a.target_label, a.target_id,
    )
    _finish(a, ChatActionStatus.DONE, message)
    return True, message


def _apply(target: User, action: str, params: dict, delete_user_account) -> str:
    name = target.username
    if action == ACTION_SET_ROLE:
        old = target.role
        target.role = params["role"]
        db.session.commit()
        return gettext("chat.action.done.role", name=name, old=str(ROLE_LABELS.get(old, old)), new=str(ROLE_LABELS[params["role"]]))
    if action == ACTION_BLOCK:
        target.role = Role.BLOCKED
        db.session.commit()
        return gettext("flash.panel.blocked", username=name)
    if action == ACTION_UNBLOCK:
        target.role = Role.USER
        db.session.commit()
        return gettext("flash.panel.unblocked", username=name)
    if action == ACTION_DELETE:
        delete_user_account(target)
        return gettext("chat.action.done.delete", name=name)
    changes, err = _check_edit(target, params.get("changes") or {})
    if err:
        raise ValueError(err)
    for key, value in changes.items():
        setattr(target, key, value)
    db.session.commit()
    return gettext("chat.action.done.edit", name=target.username, fields=", ".join(str(_FIELD_LABELS[k]) for k in changes))
