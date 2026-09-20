"""Vision Triage — Flask-версия веб-интерфейса к vision_analyzer_server.py."""

from __future__ import annotations

import os
from pathlib import Path

import click
from flask import Flask, flash, redirect, render_template, request, url_for
from flask_login import current_user
from flask_wtf.csrf import CSRFError

from .config import Config
from .extensions import csrf, db, login_manager, migrate
from .models import ROLE_CHOICES, ROLE_LABELS, RISK_LABELS, Role, User
from .utils import local_dt, page_url, truncate_chars

# Эндпоинты, доступные заблокированному пользователю.
_BLOCKED_ALLOWED = {"accounts.blocked", "accounts.logout", "static"}


def create_app(config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    if config:
        app.config.update(config)

    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)

    # --- расширения ---
    db.init_app(app)
    migrate.init_app(app, db)
    csrf.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "accounts.login"
    login_manager.login_message = None  # как в Django-версии: без служебного сообщения

    @login_manager.user_loader
    def load_user(user_id: str):
        user = db.session.get(User, int(user_id))
        # Деактивированный пользователь автоматически «вылетает» из сессии.
        return user if user is not None and user.is_active else None

    # --- blueprints ---
    from .blueprints import accounts, analyzer, panel

    app.register_blueprint(accounts.bp)
    app.register_blueprint(analyzer.bp)
    app.register_blueprint(panel.bp)

    # --- Jinja ---
    app.jinja_env.filters["localdt"] = local_dt
    app.jinja_env.filters["trunc"] = truncate_chars

    @app.context_processor
    def inject_globals():
        return {
            "page_url": page_url,
            "Role": Role,
            "ROLE_LABELS": ROLE_LABELS,
            "ROLE_CHOICES": ROLE_CHOICES,
            "RISK_LABELS": RISK_LABELS,
        }

    # --- заблокированные пользователи видят только страницу блокировки ---
    @app.before_request
    def block_guard():
        if not current_user.is_authenticated or current_user.role != Role.BLOCKED:
            return None
        if request.endpoint is None or request.endpoint in _BLOCKED_ALLOWED:
            return None
        return redirect(url_for("accounts.blocked"))

    # --- обработчики ошибок ---
    @app.errorhandler(CSRFError)
    def handle_csrf_error(_exc):
        flash("Сессия истекла или форма устарела. Повторите действие.", "error")
        if current_user.is_authenticated:
            return redirect(url_for("analyzer.dashboard"))
        return redirect(url_for("accounts.login"))

    @app.errorhandler(413)
    def handle_too_large(_exc):
        limit_mb = app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024)
        flash(f"Файл слишком большой. Максимальный размер — {limit_mb} МБ.", "error")
        if current_user.is_authenticated:
            return redirect(url_for("analyzer.dashboard"))
        return redirect(url_for("accounts.login"))

    @app.errorhandler(404)
    def not_found(_exc):
        return render_template("error.html", code=404, title="Страница не найдена",
                               text="Такой страницы нет или у вас нет к ней доступа."), 404

    @app.errorhandler(405)
    def method_not_allowed(_exc):
        return render_template("error.html", code=405, title="Метод не разрешён",
                               text="Этот адрес не принимает такой тип запроса."), 405

    @app.errorhandler(500)
    def server_error(_exc):
        return render_template("error.html", code=500, title="Внутренняя ошибка",
                               text="Что-то пошло не так. Попробуйте ещё раз позже."), 500

    # --- БД и CLI ---
    if app.config.get("AUTO_CREATE_DB", True):
        with app.app_context():
            db.create_all()

    _register_cli(app)
    return app


def _register_cli(app: Flask) -> None:
    @app.cli.command("init-db")
    def init_db():
        """Создать таблицы в базе данных."""
        db.create_all()
        click.echo("Таблицы созданы.")

    @app.cli.command("set-role")
    @click.argument("username")
    @click.argument("role", type=click.Choice([r for r, _ in ROLE_CHOICES]))
    def set_role(username: str, role: str):
        """Назначить роль пользователю из консоли (например, вернуть себе head_admin)."""
        user = db.session.scalars(db.select(User).where(User.username == username)).first()
        if user is None:
            raise click.ClickException(f"Пользователь «{username}» не найден.")
        user.role = role
        db.session.commit()
        click.echo(f"«{username}» → {ROLE_LABELS[role]}")
