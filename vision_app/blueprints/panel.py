"""Панель управления: обзор, пользователи, роли/блокировки, все анализы."""

from urllib.parse import urlparse

from flask import Blueprint, flash, redirect, render_template, request, url_for
from flask_login import current_user
from sqlalchemy import func, select
from sqlalchemy.orm import joinedload

from ..decorators import head_admin_required, staff_required
from ..extensions import db
from ..models import ROLE_CHOICES, ROLE_LABELS, AnalysisResult, Role, User
from ..services import VisionApiError, check_health
from ..utils import paginate

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

    total_analyses = db.session.scalar(select(func.count(AnalysisResult.id)))
    high_risk = db.session.scalar(
        select(func.count(AnalysisResult.id)).where(AnalysisResult.risk_level == "high")
    )
    needs_review = db.session.scalar(
        select(func.count(AnalysisResult.id)).where(AnalysisResult.needs_human_review.is_(True))
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
    analyses = db.session.scalars(
        select(AnalysisResult)
        .where(AnalysisResult.user_id == target.id)
        .order_by(AnalysisResult.created_at.desc(), AnalysisResult.id.desc())
        .limit(10)
    ).all()

    return render_template(
        "panel/user_detail.html",
        target=target,
        assignable_roles=assignable,
        can_manage=can_manage,
        analyses=analyses,
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


@bp.route("/analyses/")
@staff_required
def analyses_list():
    """Все результаты анализа в системе — для модерации."""
    risk_filter = request.args.get("risk", "").strip()
    review_only = request.args.get("review") == "1"

    stmt = (
        select(AnalysisResult)
        .options(joinedload(AnalysisResult.user))
        .order_by(AnalysisResult.created_at.desc(), AnalysisResult.id.desc())
    )
    if risk_filter:
        stmt = stmt.where(AnalysisResult.risk_level == risk_filter)
    if review_only:
        stmt = stmt.where(AnalysisResult.needs_human_review.is_(True))

    page = paginate(stmt, per_page=20)
    return render_template(
        "panel/analyses.html", page=page, risk_filter=risk_filter, review_only=review_only
    )
