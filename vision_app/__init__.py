"""Vision Triage — Flask-версия веб-интерфейса к vision_analyzer_server.py."""

from __future__ import annotations

import os
from pathlib import Path

import click
from flask import Flask, Response, flash, redirect, render_template, request, url_for
from flask_login import current_user
from flask_wtf.csrf import CSRFError

from .categories_store import normalize_legacy_wrappers, seed_default_categories
from .config import Config, conf
from .extensions import csrf, db, login_manager, migrate
from .models import ROLE_CHOICES, ROLE_LABELS, RISK_LABELS, Role, User
from .queue_worker import ensure_worker, worker_enabled
from .schema import backfill_user_fields, ensure_schema, migrate_legacy_names
from .themes import theme_presets, themes_css, themes_version
from . import settings_store
from . import image_dedup, vector_search
from .utils import local_dt, page_url, plural, truncate_chars

# Эндпоинты, доступные заблокированному пользователю.
_BLOCKED_ALLOWED = {"accounts.blocked", "accounts.logout", "static", "themes_css"}


def create_app(config: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    if config:
        app.config.update(config)

    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)

    # SQLite: обработчик очереди пишет в БД из отдельного потока — даём ждать блокировку дольше 5 с (SQLITE_TIMEOUT).
    if app.config["SQLALCHEMY_DATABASE_URI"].startswith("sqlite") and "SQLALCHEMY_ENGINE_OPTIONS" not in app.config:
        app.config["SQLALCHEMY_ENGINE_OPTIONS"] = {"connect_args": {"timeout": app.config["SQLITE_TIMEOUT"]}}

    # Исходные значения ключей Flask (лимит запроса, сессия, cookie, CSRF): к ним возвращаемся,
    # когда главный админ сбрасывает переопределение в /panel/settings/.
    settings_store.install(app)

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
    from .blueprints import accounts, analyzer, chat, panel

    app.register_blueprint(accounts.bp)
    app.register_blueprint(analyzer.bp)
    app.register_blueprint(chat.bp, url_prefix="/chat")
    app.register_blueprint(panel.bp)

    @app.get("/themes.css", endpoint="themes_css")
    def themes_css_view():
        # CSS палитр тем, собранный из vision_app/themes/*.json. Доступен всем (в т.ч. гостям),
        # версия в ?v= меняется вместе с содержимым — можно кэшировать надолго.
        resp = Response(themes_css(), mimetype="text/css")
        resp.cache_control.public = True
        resp.cache_control.max_age = 31536000
        return resp

    # --- Jinja ---
    app.jinja_env.filters["localdt"] = local_dt
    app.jinja_env.filters["trunc"] = truncate_chars
    app.jinja_env.filters["plural"] = plural
    app.jinja_env.globals["theme_presets"] = theme_presets  # темы из vision_app/themes/*.json
    app.jinja_env.globals["themes_version"] = themes_version
    app.jinja_env.globals["conf"] = conf  # шаблоны читают настройки на ходу: {{ conf("CAPTION_MAX_CHARS") }}

    @app.context_processor
    def inject_globals():
        return {
            "page_url": page_url,
            "Role": Role,
            "ROLE_LABELS": ROLE_LABELS,
            "ROLE_CHOICES": ROLE_CHOICES,
            "RISK_LABELS": RISK_LABELS,
        }

    # --- настройки из /panel/settings/: ключи, которые читает сам Flask, переносим в app.config ---
    @app.before_request
    def apply_runtime_settings():
        settings_store.sync_flask_config(app)
        return None

    # --- очередь анализов: поток-обработчик стартует лениво, при первом запросе ---
    @app.before_request
    def start_queue_worker():
        if worker_enabled(app) and request.endpoint != "static":
            ensure_worker(app)
        return None

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
            ensure_schema(db.engine)  # добавляет новые колонки в уже существующие таблицы
            migrate_legacy_names(db.engine)  # разово: ФИО -> nickname, старые колонки удаляются
            backfill_user_fields()  # цвет, стиль имени и ссылка на аватар у уже существующих пользователей
            seed_default_categories()  # на пустой БД — стартовый набор категорий оценивания
            normalize_legacy_wrappers()  # разово приводит старые записи к новому формату полей

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

    @app.cli.command("reset-settings")
    @click.argument("keys", nargs=-1)
    def reset_settings(keys: tuple[str, ...]):
        """Сбросить настройки из /panel/settings/ к значениям config.py/.env: перечисленные ключи
        или все сразу, если ключи не заданы (например, если после смены настроек не войти в панель)."""
        if not keys:
            click.echo(f"Сброшено настроек: {settings_store.reset_all_runtime_settings()}.")
            return
        unknown = [k for k in keys if k not in settings_store.RUNTIME_SETTINGS_BY_KEY]
        if unknown:
            raise click.ClickException(f"Неизвестные настройки: {', '.join(unknown)}")
        for key in keys:
            settings_store.reset_runtime_setting(key)
        click.echo(f"Сброшено: {', '.join(keys)}.")

    @app.cli.command("purge-orphans")
    def purge_orphans_cmd():
        """Удалить анализы и чаты уже удалённых пользователей (остались от старого удаления аккаунта)."""
        from .history import purge_orphans

        result = purge_orphans()
        click.echo(f"Удалено анализов: {result['analyses']}, чатов: {result['chats']}.")

    @app.cli.command("seed-categories")
    def seed_categories():
        """Загрузить категории с сервера анализа (GET /categories), если таблица categories пуста."""
        if seed_default_categories():
            click.echo("Категории загружены с сервера анализа.")
        else:
            click.echo("Ничего не загружено: таблица не пуста либо сервер анализа недоступен (см. лог).")

    @app.cli.command("reindex-embeddings")
    @click.option("--all", "everything", is_flag=True,
                  help="Пересчитать векторы ВСЕХ анализов, а не только недостающие.")
    def reindex_embeddings(everything: bool):
        """Проиндексировать описания анализов для поиска по смыслу (нужен запущенный сервер анализа
        с готовой моделью эмбеддингов: GET /embeddings -> state=ready)."""
        result = vector_search.backfill(everything=everything)
        click.echo(f"Проиндексировано анализов: {result.indexed}")
        if result.reason:
            raise click.ClickException(f"Остановлено раньше времени: {result.reason}")

    @app.cli.command("semantic-search")
    @click.argument("queries", nargs=-1, required=True)
    @click.option("--limit", default=15, show_default=True, help="Сколько строк показать на запрос.")
    @click.option("--user", "username", default=None, help="Только анализы этого пользователя (по умолчанию — все).")
    def semantic_search_cmd(queries: tuple[str, ...], limit: int, username: str | None):
        """Сырые скоры поиска по смыслу БЕЗ участия LLM и БЕЗ отсечек — для подбора формулировок и порогов.

        Можно передать несколько запросов сразу, чтобы сравнить формулировки:
        flask semantic-search "человек в головном уборе" "кепка, шапка, шляпа, головной убор"
        """
        from .models import AnalysisResult, Status

        conditions = [AnalysisResult.status == Status.DONE]
        if username:
            user = db.session.scalars(db.select(User).where(User.username == username)).first()
            if user is None:
                raise click.ClickException(f"Пользователь «{username}» не найден.")
            conditions.append(AnalysisResult.user_id == user.id)

        for query in queries:
            try:
                qvec, model = vector_search.embed_query(query)
            except Exception as exc:  # noqa: BLE001 — для CLI важнее понятное сообщение
                raise click.ClickException(f"Не удалось получить эмбеддинг запроса: {exc}")
            result = vector_search.semantic_search(
                conditions, qvec, model, limit=limit, min_similarity=-1.0,
                scan_limit=int(conf("EMBEDDING_SCAN_LIMIT")),
            )
            rows = {
                r.id: r for r in db.session.scalars(
                    db.select(AnalysisResult).where(AnalysisResult.id.in_([i for i, _ in result.hits]))
                )
            }
            click.echo(f"\n=== {query!r}  (сравнено {result.scanned}, модель {model}) ===")
            for analysis_id, score in result.hits:
                desc = " ".join((rows[analysis_id].description or "").split())
                click.echo(f"  #{analysis_id:<5} {score:.3f}  {rows[analysis_id].risk_level:<8} {desc[:100]}")
            if result.hits:
                click.echo(f"  разброс среди показанных: {result.hits[0][1] - result.hits[-1][1]:.3f}")

    @app.cli.command("backfill-image-hashes")
    def backfill_image_hashes():
        """Посчитать хеши файлов у анализов, где их ещё нет (для склейки повторных загрузок одного файла)."""
        total = missing = 0
        while True:
            hashed, lost = image_dedup.backfill_hashes(limit=conf("DEDUP_CLI_BATCH"))
            if not hashed and not lost:
                break
            total += hashed
            missing += lost
        click.echo(f"Хешей посчитано: {total}, файл не найден: {missing}")
