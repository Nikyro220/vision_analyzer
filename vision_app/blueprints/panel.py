"""Панель управления: обзор, пользователи, роли/блокировки, все анализы."""

from urllib.parse import urlparse

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user
from sqlalchemy import func, select
from sqlalchemy.orm import joinedload

from ..decorators import head_admin_required, staff_required
from ..extensions import db
from ..history import delete_finished
from ..settings_store import (
    RUNTIME_SETTINGS,
    get_runtime_setting,
    is_runtime_setting_overridden,
    reset_runtime_setting,
    set_runtime_setting,
)
from ..models import ROLE_CHOICES, ROLE_LABELS, AnalysisResult, Role, Status, User
from ..services import VisionApiError, check_health
from ..utils import paginate, plural

bp = Blueprint("panel", __name__, url_prefix="/panel")


def _safe_referrer(default: str) -> str:
    """Редирект назад по Referer, но только в пределах этого же сайта."""
    ref = request.referrer
    if ref:
        parsed = urlparse(ref)
        if parsed.netloc == request.host:
            return ref
    return default


@bp.route("/")
@head_admin_required
def stats():
    total_users = db.session.scalar(select(func.count(User.id)))
    counts = dict(db.session.execute(select(User.role, func.count(User.id)).group_by(User.role)).all())
    role_counts = [(role, label, counts.get(role, 0)) for role, label in ROLE_CHOICES]

    done = AnalysisResult.status == Status.DONE
    total_analyses = db.session.scalar(select(func.count(AnalysisResult.id)).where(done))
    high_risk = db.session.scalar(
        select(func.count(AnalysisResult.id)).where(done, AnalysisResult.risk_level == "high")
    )
    needs_review = db.session.scalar(
        select(func.count(AnalysisResult.id)).where(done, AnalysisResult.needs_human_review.is_(True))
    )

    try:
        backend_status = check_health()
        backend_error = None
    except VisionApiError as exc:
        backend_status = None
        backend_error = str(exc)

    return render_template(
        "panel/stats.html",
        total_users=total_users,
        role_counts=role_counts,
        total_analyses=total_analyses,
        high_risk=high_risk,
        needs_review=needs_review,
        backend_status=backend_status,
        backend_error=backend_error,
    )


@bp.route("/users/")
@staff_required
def users_list():
    query = request.args.get("q", "").strip()
    role_filter = request.args.get("role", "").strip()

    stmt = select(User).order_by(User.created_at.desc(), User.id.desc())
    if query:
        stmt = stmt.where(User.username.icontains(query, autoescape=True))
    if role_filter:
        stmt = stmt.where(User.role == role_filter)

    page = paginate(stmt, per_page=20)
    return render_template(
        "panel/users.html",
        page=page,
        query=query,
        role_filter=role_filter,
        roles=ROLE_CHOICES,
    )


@bp.route("/users/<int:pk>/")
@staff_required
def user_detail(pk: int):
    target = db.get_or_404(User, pk)
    can_manage = current_user.can_manage(target)
    assignable = current_user.assignable_roles() if can_manage else []
    finished = (AnalysisResult.user_id == target.id, AnalysisResult.status == Status.DONE)
    analyses = db.session.scalars(
        select(AnalysisResult)
        .where(*finished)
        .order_by(AnalysisResult.created_at.desc(), AnalysisResult.id.desc())
        .limit(10)
    ).all()
    history_count = db.session.scalar(select(func.count(AnalysisResult.id)).where(*finished))

    return render_template(
        "panel/user_detail.html",
        target=target,
        assignable_roles=assignable,
        can_manage=can_manage,
        analyses=analyses,
        history_count=history_count,
        all_roles=ROLE_CHOICES,
    )


@bp.route("/users/<int:pk>/set-role/", methods=["POST"])
@staff_required
def user_set_role(pk: int):
    target = db.get_or_404(User, pk)
    new_role = request.form.get("role", "")

    if not current_user.can_manage(target):
        flash("У вас нет прав на изменение этого пользователя.", "error")
        return redirect(url_for("panel.user_detail", pk=target.id))

    if new_role not in current_user.assignable_roles():
        flash("Вы не можете назначить эту роль.", "error")
        return redirect(url_for("panel.user_detail", pk=target.id))

    old_role = target.role
    target.role = new_role
    db.session.commit()

    if old_role != new_role:
        flash(
            f"Роль пользователя «{target.username}» изменена: "
            f"{ROLE_LABELS[old_role]} → {ROLE_LABELS[new_role]}.",
            "success",
        )
    return redirect(url_for("panel.user_detail", pk=target.id))


@bp.route("/users/<int:pk>/toggle-block/", methods=["POST"])
@staff_required
def user_toggle_block(pk: int):
    """Быстрая кнопка блокировки/разблокировки."""
    target = db.get_or_404(User, pk)

    if not current_user.can_manage(target):
        flash("У вас нет прав на блокировку этого пользователя.", "error")
        return redirect(url_for("panel.users_list"))

    if target.role == Role.BLOCKED:
        target.role = Role.USER
        flash(f"Пользователь «{target.username}» разблокирован.", "success")
    else:
        target.role = Role.BLOCKED
        flash(f"Пользователь «{target.username}» заблокирован.", "success")

    db.session.commit()
    return redirect(_safe_referrer(url_for("panel.users_list")))


def _analysis_filters() -> tuple[list, str, bool]:
    """Условия фильтра для «Все анализы» (только завершённые) + значения для формы."""
    risk_filter = request.args.get("risk", request.form.get("risk", "")).strip()
    review_only = (request.args.get("review") or request.form.get("review")) == "1"

    conditions = [AnalysisResult.status == Status.DONE]
    if risk_filter:
        conditions.append(AnalysisResult.risk_level == risk_filter)
    if review_only:
        conditions.append(AnalysisResult.needs_human_review.is_(True))
    return conditions, risk_filter, review_only


@bp.route("/analyses/")
@staff_required
def analyses_list():
    """Все завершённые анализы в системе — для модерации."""
    conditions, risk_filter, review_only = _analysis_filters()
    stmt = (
        select(AnalysisResult)
        .options(joinedload(AnalysisResult.user))
        .where(*conditions)
        .order_by(AnalysisResult.created_at.desc(), AnalysisResult.id.desc())
    )
    page = paginate(stmt, per_page=20)
    return render_template(
        "panel/analyses.html", page=page, risk_filter=risk_filter, review_only=review_only
    )


@bp.route("/analyses/delete-filtered/", methods=["POST"])
@staff_required
def analyses_delete_filtered():
    """Удалить ВСЕ завершённые анализы, подходящие под текущий фильтр (не только с этой страницы)."""
    conditions, risk_filter, review_only = _analysis_filters()
    count = delete_finished(*conditions[1:])  # первый пункт (status=done) delete_finished добавляет сам
    flash(f"Удалено записей: {count}." if count else "Нечего удалять.", "success" if count else "info")
    return redirect(url_for("panel.analyses_list", risk=risk_filter or None, review="1" if review_only else None))


@bp.route("/users/<int:pk>/clear-history/", methods=["POST"])
@staff_required
def user_clear_history(pk: int):
    """Удалить всю завершённую историю пользователя (записи и изображения)."""
    target = db.get_or_404(User, pk)
    count = delete_finished(AnalysisResult.user_id == target.id)
    if count:
        flash(f"История пользователя «{target.username}» очищена: удалено {plural(count, ('запись', 'записи', 'записей'))}.", "success")
    else:
        flash(f"У пользователя «{target.username}» нет завершённых анализов.", "info")
    return redirect(url_for("panel.user_detail", pk=target.id))


@bp.route("/settings/", methods=["GET", "POST"])
@head_admin_required
def settings():
    """Настройки очереди и клиента vision-сервера — меняются на ходу, без перезапуска.

    Хранятся в БД (см. settings_store.py); значение по умолчанию берётся из .env/Config,
    пока главный админ явно его не переопределит здесь.
    """
    if request.method == "POST":
        errors = []
        for spec in RUNTIME_SETTINGS:
            if request.form.get(f"reset_{spec.key}"):
                reset_runtime_setting(spec.key)
                continue
            # Невыбранный чекбокс браузер вообще не отправляет — это и есть "выключено",
            # поэтому дефолт при отсутствии ключа в форме — пустая строка, а не "1".
            raw = request.form.get(spec.key, "")
            try:
                set_runtime_setting(spec.key, raw)
            except ValueError as exc:
                errors.append(str(exc))

        if errors:
            db.session.rollback()
            for message in errors:
                flash(message, "error")
        else:
            db.session.commit()
            flash("Настройки сохранены.", "success")
        return redirect(url_for("panel.settings"))

    rows = [
        {
            "spec": spec,
            "value": get_runtime_setting(spec.key),
            "overridden": is_runtime_setting_overridden(spec.key),
        }
        for spec in RUNTIME_SETTINGS
    ]
    return render_template("panel/settings.html", rows=rows)