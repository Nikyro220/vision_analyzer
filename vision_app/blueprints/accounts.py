"""Регистрация, вход/выход, страница блокировки, профиль."""

from flask import Blueprint, flash, redirect, render_template, request, session, url_for
from flask_login import current_user, login_required, login_user, logout_user
from sqlalchemy import func, select

from ..extensions import db
from ..forms import AccountForm, DeleteAccountForm, LoginForm, RegisterForm
from ..history import delete_user_account
from ..models import AnalysisResult, Role, Status, User
from ..utils import is_safe_next

bp = Blueprint("accounts", __name__, url_prefix="/accounts")


@bp.route("/register/", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("analyzer.dashboard"))

    form = RegisterForm()
    if form.validate_on_submit():
        user = User(
            username=form.username.data.strip(),
            email=(form.email.data or "").strip(),
        )
        user.set_password(form.password1.data)

        # Первый зарегистрированный в системе пользователь автоматически
        # становится главным админом — иначе панелью некому управлять.
        is_first = db.session.scalar(select(func.count(User.id))) == 0
        user.role = Role.HEAD_ADMIN if is_first else Role.USER

        db.session.add(user)
        db.session.commit()

        session.permanent = True
        login_user(user)
        if is_first:
            flash(
                "Регистрация выполнена. Вы первый пользователь системы — "
                "вам присвоена роль «Главный администратор».",
                "success",
            )
        else:
            flash("Регистрация прошла успешно. Добро пожаловать!", "success")
        return redirect(url_for("analyzer.dashboard"))

    return render_template("accounts/register.html", form=form)


@bp.route("/login/", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("analyzer.dashboard"))

    form = LoginForm()
    error = None
    if form.validate_on_submit():
        username = form.username.data.strip()
        user = db.session.scalars(
            select(User).where(func.lower(User.username) == username.lower())
        ).first()

        if user is None or not user.check_password(form.password.data):
            error = "Неверный логин или пароль."
        elif not user.is_active:
            error = "Этот аккаунт деактивирован."
        else:
            session.permanent = True
            login_user(user)
            if user.role == Role.BLOCKED:
                return redirect(url_for("accounts.blocked"))
            flash(f"С возвращением, {user.username}!", "success")
            target = request.args.get("next")
            return redirect(target if is_safe_next(target) else url_for("analyzer.dashboard"))

    return render_template("accounts/login.html", form=form, error=error)


@bp.route("/logout/", methods=["POST"])
def logout():
    logout_user()
    flash("Вы вышли из системы.", "info")
    return redirect(url_for("accounts.login"))


@bp.route("/blocked/")
@login_required
def blocked():
    if current_user.role != Role.BLOCKED:
        return redirect(url_for("analyzer.dashboard"))
    return render_template("accounts/blocked.html")


@bp.route("/profile/", methods=["GET", "POST"])
@login_required
def profile():
    form = AccountForm(obj=current_user, current_id=current_user.id)
    delete_form = DeleteAccountForm()

    if form.validate_on_submit():
        current_user.username = form.username.data.strip()
        current_user.email = (form.email.data or "").strip()
        current_user.first_name = (form.first_name.data or "").strip()
        current_user.last_name = (form.last_name.data or "").strip()
        db.session.commit()
        flash("Данные аккаунта обновлены.", "success")
        return redirect(url_for("accounts.profile"))

    finished = (AnalysisResult.user_id == current_user.id, AnalysisResult.status == Status.DONE)
    analyses = db.session.scalars(
        select(AnalysisResult)
        .where(*finished)
        .order_by(AnalysisResult.created_at.desc(), AnalysisResult.id.desc())
        .limit(10)
    ).all()
    history_count = db.session.scalar(select(func.count(AnalysisResult.id)).where(*finished))

    delete_sql = f"DELETE FROM users WHERE id = {current_user.id};"

    return render_template(
        "accounts/profile.html",
        form=form,
        delete_form=delete_form,
        delete_sql=delete_sql,
        analyses=analyses,
        history_count=history_count,
    )


@bp.route("/profile/delete/", methods=["POST"])
@login_required
def delete_own_account():
    form = DeleteAccountForm()
    delete_sql = f"DELETE FROM users WHERE id = {current_user.id};"

    if form.validate_on_submit() and (form.confirm_sql.data or "").strip() == delete_sql:
        username = current_user.username
        user = current_user._get_current_object()
        logout_user()
        delete_user_account(user)
        flash(f"Аккаунт «{username}» удалён.", "success")
        return redirect(url_for("accounts.login"))

    flash("Команда подтверждения введена неверно. Аккаунт не удалён.", "error")
    return redirect(url_for("accounts.profile"))
