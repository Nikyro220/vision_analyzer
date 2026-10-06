"""Панель управления: обзор, пользователи, роли/блокировки, все анализы."""

import json
from urllib.parse import urlparse

from flask import Blueprint, flash, jsonify, redirect, render_template, request, url_for
from flask_login import current_user
from sqlalchemy import func, select
from sqlalchemy.orm import joinedload

from .. import examples_codec
from ..config import conf
from ..decorators import head_admin_required, staff_required
from ..extensions import db
from ..chat_prompt import DEFAULT_CHAT_PROMPT, get_chat_prompt, is_customized, reset_chat_prompt, set_chat_prompt
from ..forms import AccountForm, CategoryForm, ChatPromptForm, DeleteAccountForm, DeleteCategoryForm
from ..categories_store import seed_default_categories
from ..history import delete_finished, delete_user_account
from ..settings_store import (
    SETTING_GROUPS,
    default_value,
    display_value,
    effective_bounds,
    is_runtime_setting_overridden,
    save_runtime_settings,
)
from ..models import ROLE_CHOICES, ROLE_LABELS, AnalysisResult, Category, Role, Status, User
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

    page = paginate(stmt, per_page=conf("PANEL_PER_PAGE"))
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
        .limit(conf("USER_RECENT_ANALYSES_LIMIT"))
    ).all()
    history_count = db.session.scalar(select(func.count(AnalysisResult.id)).where(*finished))

    edit_form = AccountForm(obj=target, current_id=target.id)
    delete_form = DeleteAccountForm()
    delete_sql = f"DELETE FROM users WHERE id = {target.id};"

    return render_template(
        "panel/user_detail.html",
        target=target,
        assignable_roles=assignable,
        can_manage=can_manage,
        analyses=analyses,
        history_count=history_count,
        all_roles=ROLE_CHOICES,
        edit_form=edit_form,
        delete_form=delete_form,
        delete_sql=delete_sql,
    )


@bp.route("/users/<int:pk>/history/")
@staff_required
def user_history(pk: int):
    """Вся история завершённых анализов одного пользователя (с пагинацией)."""
    target = db.get_or_404(User, pk)
    stmt = (
        select(AnalysisResult)
        .where(AnalysisResult.user_id == target.id, AnalysisResult.status == Status.DONE)
        .order_by(AnalysisResult.created_at.desc(), AnalysisResult.id.desc())
    )
    page = paginate(stmt, per_page=conf("PANEL_PER_PAGE"))
    return render_template("panel/user_history.html", target=target, page=page)


@bp.route("/users/<int:pk>/edit/", methods=["POST"])
@staff_required
def user_edit(pk: int):
    """Изменение данных аккаунта (логин/email/никнейм/стиль имени) со стороны админа."""
    target = db.get_or_404(User, pk)

    if not current_user.can_manage(target):
        flash("У вас нет прав на изменение этого пользователя.", "error")
        return redirect(url_for("panel.user_detail", pk=target.id))

    form = AccountForm(current_id=target.id)
    if form.validate_on_submit():
        target.username = form.username.data.strip()
        target.email = (form.email.data or "").strip()
        target.nickname = form.nickname.data or ""
        target.display_style = form.display_style.data
        db.session.commit()
        flash(f"Данные пользователя «{target.username}» обновлены.", "success")
    else:
        for field_errors in form.errors.values():
            for error in field_errors:
                flash(error, "error")

    return redirect(url_for("panel.user_detail", pk=target.id))


@bp.route("/users/<int:pk>/delete/", methods=["POST"])
@staff_required
def user_delete(pk: int):
    """Удаление аккаунта пользователя админом — с ручным подтверждением SQL-командой."""
    target = db.get_or_404(User, pk)

    if not current_user.can_manage(target):
        flash("У вас нет прав на удаление этого пользователя.", "error")
        return redirect(url_for("panel.user_detail", pk=target.id))

    delete_sql = f"DELETE FROM users WHERE id = {target.id};"
    form = DeleteAccountForm()

    if form.validate_on_submit() and (form.confirm_sql.data or "").strip() == delete_sql:
        username = target.username
        delete_user_account(target)
        flash(f"Аккаунт «{username}» удалён.", "success")
        return redirect(url_for("panel.users_list"))

    flash("Команда подтверждения введена неверно. Аккаунт не удалён.", "error")
    return redirect(url_for("panel.user_detail", pk=target.id))


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


def _analysis_filters() -> tuple[list, str, bool, str, str]:
    """Условия фильтра для «Все анализы» (только завершённые) + значения для формы.

    Поддерживает поиск по имени файла (``q``) и по имени пользователя (``user``).
    """
    risk_filter = request.args.get("risk", request.form.get("risk", "")).strip()
    review_only = (request.args.get("review") or request.form.get("review")) == "1"
    query = request.args.get("q", request.form.get("q", "")).strip()
    user_query = request.args.get("user", request.form.get("user", "")).strip()

    conditions = [AnalysisResult.status == Status.DONE]
    if risk_filter:
        conditions.append(AnalysisResult.risk_level == risk_filter)
    if review_only:
        conditions.append(AnalysisResult.needs_human_review.is_(True))
    if query:
        conditions.append(AnalysisResult.original_name.icontains(query, autoescape=True))
    if user_query:
        conditions.append(AnalysisResult.user.has(User.username.icontains(user_query, autoescape=True)))
    return conditions, risk_filter, review_only, query, user_query


@bp.route("/analyses/")
@staff_required
def analyses_list():
    """Все завершённые анализы в системе — для модерации."""
    conditions, risk_filter, review_only, query, user_query = _analysis_filters()
    stmt = (
        select(AnalysisResult)
        .options(joinedload(AnalysisResult.user))
        .where(*conditions)
        .order_by(AnalysisResult.created_at.desc(), AnalysisResult.id.desc())
    )
    page = paginate(stmt, per_page=conf("PANEL_PER_PAGE"))

    # Запрос из JS-фильтра: отдаём только фрагмент с таблицей/пагинацией.
    if request.headers.get("X-Requested-With") == "fetch":
        html = render_template("panel/_analyses_results.html", page=page)
        return jsonify(html=html, total=page.total, pages=page.pages, page_num=page.page)

    return render_template(
        "panel/analyses.html",
        page=page,
        risk_filter=risk_filter,
        review_only=review_only,
        query=query,
        user_query=user_query,
    )


@bp.route("/analyses/delete-filtered/", methods=["POST"])
@staff_required
def analyses_delete_filtered():
    """Удалить ВСЕ завершённые анализы, подходящие под текущий фильтр (не только с этой страницы)."""
    conditions, risk_filter, review_only, query, user_query = _analysis_filters()
    count = delete_finished(*conditions[1:])  # первый пункт (status=done) delete_finished добавляет сам
    flash(f"Удалено записей: {count}." if count else "Нечего удалять.", "success" if count else "info")
    return redirect(
        url_for(
            "panel.analyses_list",
            risk=risk_filter or None,
            review="1" if review_only else None,
            q=query or None,
            user=user_query or None,
        )
    )


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
    """Все настройки из config.py (кроме ключей, путей и логирования) — меняются на ходу, без перезапуска.

    Хранятся в БД (см. settings_store.py); пока главный админ явно не переопределит значение здесь,
    действует то, что задано в config.py/.env. Форма сохраняется целиком: при любой ошибке
    валидации не меняется ничего.
    """
    if request.method == "POST":
        errors, warnings = save_runtime_settings(request.form)
        if errors:
            for message in errors:
                flash(message, "error")
            flash("Настройки не сохранены — исправьте ошибки и повторите.", "error")
        else:
            flash("Настройки сохранены и уже действуют.", "success")
            for message in warnings:
                flash(message, "warning")
        return redirect(url_for("panel.settings"))

    groups = []
    for title, description, specs in SETTING_GROUPS:
        rows = []
        for spec in specs:
            lo, hi = effective_bounds(spec)
            if spec.kind == "int" and spec.scale > 1:
                lo = lo // spec.scale if spec.min_key else lo
                hi = hi // spec.scale if spec.max_key else hi
            rows.append(
                {
                    "spec": spec,
                    "value": display_value(spec, conf(spec.key)),
                    "overridden": is_runtime_setting_overridden(spec.key),
                    "default": display_value(spec, default_value(spec.key)),
                    "lo": lo if spec.kind in ("int", "float", "days", "opt_int") else None,
                    "hi": hi if spec.kind in ("int", "float", "days", "opt_int") else None,
                }
            )
        groups.append({"title": title, "description": description, "rows": rows})
    return render_template("panel/settings.html", groups=groups)


# ----------------------------------------------------------------------------
# Промпты: системный промпт чата + категории оценивания (страница /panel/categories/)
# ----------------------------------------------------------------------------
def _render_prompts(prompt_form: ChatPromptForm | None = None, status: int = 200):
    rows = db.session.scalars(select(Category).order_by(Category.position, Category.id)).all()
    if prompt_form is None:
        prompt_form = ChatPromptForm(formdata=None, prompt=get_chat_prompt())
    html = render_template(
        "panel/categories.html",
        rows=rows,
        delete_form=DeleteCategoryForm(),
        prompt_form=prompt_form,
        prompt_customized=is_customized(),
        prompt_default=DEFAULT_CHAT_PROMPT,
    )
    return html, status


@bp.route("/categories/")
@staff_required
def categories_list():
    """Раздел «Промпты». Сверху — системный промпт чата (хранится в БД, см. chat_prompt.py);
    ниже — список категорий оценивания, единственное место, где они хранятся
    (см. models.Category); при каждом анализе текущий набор целиком уходит
    на сервер анализа как разовый оверлей (см. services.analyze_image)."""
    return _render_prompts()


@bp.route("/chat-prompt/", methods=["POST"])
@staff_required
def chat_prompt_save():
    """Сохранить системный промпт чата. Действует со следующего сообщения в любом чате;
    промпт инструментов (поиск по анализам и т. п.) добавляется к нему автоматически."""
    form = ChatPromptForm()
    if not form.validate_on_submit():
        for errors in form.errors.values():
            for message in errors:
                flash(message, "error")
        return _render_prompts(prompt_form=form, status=400)  # введённый текст не теряем

    if set_chat_prompt(form.prompt.data or ""):
        flash("Системный промпт чата сохранён. Он применится со следующего сообщения.", "success")
    else:
        flash("Действует стандартный системный промпт чата.", "info")
    return redirect(url_for("panel.categories_list"))


@bp.route("/chat-prompt/reset/", methods=["POST"])
@staff_required
def chat_prompt_reset():
    form = DeleteCategoryForm()  # пустая форма с CSRF-токеном
    if not form.validate_on_submit():
        flash("Сессия устарела, попробуйте ещё раз.", "error")
    else:
        reset_chat_prompt()
        flash("Системный промпт чата сброшен к стандартному.", "success")
    return redirect(url_for("panel.categories_list"))


@bp.route("/categories/import/", methods=["POST"])
@staff_required
def categories_import():
    """Загрузить стартовый набор категорий с сервера анализа (GET /categories) — работает,
    только пока таблица пуста; существующие категории не трогает."""
    form = DeleteCategoryForm()  # пустая форма с CSRF-токеном
    if not form.validate_on_submit():
        flash("Сессия устарела, попробуйте ещё раз.", "error")
    elif db.session.scalar(select(func.count(Category.id))):
        flash("Категории уже есть — загрузка с сервера нужна только для пустого списка.", "warning")
    elif seed_default_categories():
        flash("Категории загружены с сервера анализа.", "success")
    else:
        flash("Не удалось получить категории с сервера анализа — проверьте, что он запущен.", "error")
    return redirect(url_for("panel.categories_list"))


def _examples_context(form, category) -> dict:
    """Начальные сцены для редактора примеров: при ошибке валидации — то, что
    пользователь только что отправил, иначе — разобранный текст из БД."""
    result = {}
    for lang, field in (("en", form.examples_en_data), ("ru", form.examples_ru_data)):
        scenes = None
        if field.data:
            try:
                loaded = json.loads(field.data)
                if isinstance(loaded, list):
                    scenes = loaded
            except ValueError:
                pass
        if scenes is None:
            text = getattr(category, f"example_{lang}", "") if category else ""
            scenes = examples_codec.parse_examples(text)
        result[lang] = scenes
    return result


@bp.route("/categories/new/", methods=["GET", "POST"])
@staff_required
def category_new():
    form = CategoryForm()
    if form.validate_on_submit():
        next_position = db.session.scalar(select(func.count(Category.id)))
        category = Category(
            name=form.name.data.strip(),
            title=(form.title.data or "").strip(),
            summary=form.summary.data.strip(),
            full=form.full.data.strip(),
            compact=form.compact.data.strip(),
            full_extra=(form.full_extra.data or "").strip(),
            compact_extra=(form.compact_extra.data or "").strip(),
            example_en=examples_codec.data_to_text(form.examples_en_data.data, form.name.data.strip()),
            example_ru=examples_codec.data_to_text(form.examples_ru_data.data, form.name.data.strip()),
            position=form.position.data if form.position.data is not None else next_position,
            is_active=form.is_active.data,
        )
        db.session.add(category)
        db.session.commit()
        flash(f"Категория «{category.name}» добавлена.", "success")
        return redirect(url_for("panel.categories_list"))

    return render_template(
        "panel/category_form.html",
        form=form,
        category=None,
        full_paragraphs=[""],
        compact_paragraphs=[""],
        examples_data=_examples_context(form, None),
    )


@bp.route("/categories/<int:pk>/", methods=["GET", "POST"])
@staff_required
def category_edit(pk: int):
    category = db.get_or_404(Category, pk)
    form = CategoryForm(obj=category, current_id=category.id)

    if form.validate_on_submit():
        category.name = form.name.data.strip()
        category.title = (form.title.data or "").strip()
        category.summary = form.summary.data.strip()
        category.full = form.full.data.strip()
        category.compact = form.compact.data.strip()
        category.full_extra = (form.full_extra.data or "").strip()
        category.compact_extra = (form.compact_extra.data or "").strip()
        category.example_en = examples_codec.data_to_text(form.examples_en_data.data, category.name)
        category.example_ru = examples_codec.data_to_text(form.examples_ru_data.data, category.name)
        category.position = form.position.data if form.position.data is not None else category.position
        category.is_active = form.is_active.data
        db.session.commit()
        flash(f"Категория «{category.name}» обновлена.", "success")
        return redirect(url_for("panel.categories_list"))

    return render_template(
        "panel/category_form.html",
        form=form,
        category=category,
        full_paragraphs=category.full_paragraphs(),
        compact_paragraphs=category.compact_paragraphs(),
        examples_data=_examples_context(form, category),
    )


@bp.route("/categories/<int:pk>/delete/", methods=["POST"])
@staff_required
def category_delete(pk: int):
    category = db.get_or_404(Category, pk)
    form = DeleteCategoryForm()
    if form.validate_on_submit():
        name = category.name
        db.session.delete(category)
        db.session.commit()
        flash(f"Категория «{name}» удалена.", "success")
    else:
        flash("Не удалось удалить категорию — сессия устарела, попробуйте ещё раз.", "error")
    return redirect(url_for("panel.categories_list"))


@bp.route("/categories/<int:pk>/toggle-active/", methods=["POST"])
@staff_required
def category_toggle_active(pk: int):
    """Быстрое включение/выключение без открытия формы редактирования."""
    category = db.get_or_404(Category, pk)
    category.is_active = not category.is_active
    db.session.commit()
    flash(
        f"Категория «{category.name}» {'включена' if category.is_active else 'выключена'}.",
        "success",
    )
    return redirect(_safe_referrer(url_for("panel.categories_list")))