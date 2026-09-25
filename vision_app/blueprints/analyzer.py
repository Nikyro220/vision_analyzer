"""Загрузка изображений (очередь), история, детальный результат, статус сервера анализа."""

import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from flask_login import current_user, login_required
from sqlalchemy import delete, func, select, update

from ..decorators import staff_required
from ..extensions import db
from ..forms import SAMPLING_FORMS, ImageUploadForm
from ..history import delete_finished, remove_image_files
from ..models import RISK_LABELS, STATUS_LABELS, AnalysisResult, Status
from ..queue_worker import wake_worker
from ..services import (
    BACKENDS,
    VisionApiError,
    check_health,
    get_models,
    get_sampling,
    set_sampling,
)
from ..settings_store import clear_analysis_target, get_analysis_target, set_analysis_target
from ..utils import is_safe_next, local_dt, paginate, plural

bp = Blueprint("analyzer", __name__)

_FILES = ("файл", "файла", "файлов")


def _own_results(q: str = ""):
    """История пользователя — только ЗАВЕРШЁННЫЕ анализы (очередь показывается отдельно).

    ``q`` — необязательный поиск по имени файла.
    """
    stmt = select(AnalysisResult).where(
        AnalysisResult.user_id == current_user.id, AnalysisResult.status == Status.DONE
    )
    if q:
        stmt = stmt.where(AnalysisResult.original_name.icontains(q, autoescape=True))
    return stmt.order_by(AnalysisResult.created_at.desc(), AnalysisResult.id.desc())


def _can_view(result: AnalysisResult) -> bool:
    return result.user_id == current_user.id or current_user.is_panel_staff


def _redirect_back(default_endpoint: str, **values):
    """Возврат на страницу, откуда пришёл запрос (?next=/hidden next), только внутри сайта."""
    target = request.form.get("next") or request.args.get("next")
    return redirect(target if is_safe_next(target) else url_for(default_endpoint, **values))


# ----------------------------------------------------------------------------
# Очередь
# ----------------------------------------------------------------------------
def _as_utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def queue_snapshot() -> dict:
    """Очередь текущего пользователя и его последние завершённые анализы (для страницы и JSON)."""
    # Очередь одна на всех (модель обрабатывает по одному изображению), поэтому позицию
    # считаем среди ВСЕХ ожидающих, а показываем пользователю только его записи.
    pending_ids = db.session.scalars(
        select(AnalysisResult.id).where(AnalysisResult.status != Status.DONE).order_by(AnalysisResult.id)
    ).all()
    position = {pid: idx for idx, pid in enumerate(pending_ids)}

    mine = db.session.scalars(
        select(AnalysisResult)
        .where(AnalysisResult.user_id == current_user.id, AnalysisResult.status != Status.DONE)
        .order_by(AnalysisResult.id)
    ).all()

    now = datetime.now(timezone.utc)
    pending = []
    for row in mine:
        since = row.started_at if row.status == Status.PROCESSING and row.started_at else row.created_at
        pending.append(
            {
                "id": row.id,
                "name": row.original_name or "без имени",
                "status": row.status,
                "status_label": STATUS_LABELS.get(row.status, row.status),
                "ahead": position.get(row.id, 0),
                "elapsed": max(int((now - _as_utc(since)).total_seconds()), 0),
                "cancel_url": url_for("analyzer.cancel_queued", pk=row.id) if row.status == Status.QUEUED else "",
            }
        )

    recent = [
        {
            "id": row.id,
            "name": row.original_name or "без имени",
            "risk_level": row.risk_level,
            "risk_label": RISK_LABELS.get(row.risk_level, row.risk_level),
            "error": row.is_error,
            "is_new": row.is_new,
            "date": local_dt(row.created_at, "%d.%m %H:%M"),
            "url": url_for("analyzer.result_detail", pk=row.id),
        }
        for row in db.session.scalars(_own_results().limit(6)).all()
    ]
    return {"pending": pending, "recent": recent, "queue_total": len(pending_ids)}


def _enqueue_uploads(form: ImageUploadForm):
    """Кладёт проверенные файлы в очередь и сразу возвращает пользователя на страницу."""
    from ..settings_store import get_runtime_setting

    limit = get_runtime_setting("QUEUE_MAX_PENDING_PER_USER") or 30
    pending_now = db.session.scalar(
        select(func.count(AnalysisResult.id)).where(
            AnalysisResult.user_id == current_user.id, AnalysisResult.status != Status.DONE 
        )
    )
    if pending_now + len(form.accepted) > limit:
        flash(
            f"В вашей очереди уже {pending_now}, максимум — {limit}. Дождитесь обработки и повторите.",
            "error",
        )
        return redirect(url_for("analyzer.dashboard"))

    now = datetime.now(timezone.utc)
    root = Path(current_app.config["UPLOAD_FOLDER"])
    written: list[Path] = []
    try:
        for item in form.accepted:
            # Файл на диске хранится под случайным именем, оригинальное имя — только в БД.
            rel_path = f"uploads/{now:%Y/%m/%d}/{uuid.uuid4().hex}{item.ext}"
            abs_path = root / rel_path
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_bytes(item.data)
            written.append(abs_path)
            db.session.add(
                AnalysisResult(
                    user_id=current_user.id,
                    image_path=rel_path,
                    original_name=item.filename[:255],
                    image_mime=item.mime,
                    caption=item.caption,
                    status=Status.QUEUED,
                )
            )
        db.session.commit()
    except Exception:
        db.session.rollback()
        for path in written:
            path.unlink(missing_ok=True)
        raise

    wake_worker(current_app)
    flash(f"Добавлено в очередь: {plural(len(form.accepted), _FILES)}.", "success")
    for name, reason in form.rejected[:5]:
        flash(f"Пропущен файл «{name}»: {reason}.", "warning")
    if len(form.rejected) > 5:
        flash(f"…и ещё пропущено: {len(form.rejected) - 5}.", "warning")
    return redirect(url_for("analyzer.dashboard"))


@bp.route("/", methods=["GET", "POST"])
@login_required
def dashboard():
    form = ImageUploadForm()
    if form.validate_on_submit():
        return _enqueue_uploads(form)

    snapshot = queue_snapshot()
    target_backend, target_model = get_analysis_target()
    return render_template(
        "analyzer/dashboard.html",
        form=form,
        pending=snapshot["pending"],
        recent=snapshot["recent"],
        queue_total=snapshot["queue_total"],
        target_backend=target_backend,
        target_model=target_model,
    )


@bp.route("/queue/status")
@login_required
def queue_status():
    """JSON для окна очереди: страница опрашивает его, пока есть незавершённые анализы."""
    response = jsonify(queue_snapshot())
    response.headers["Cache-Control"] = "no-store"
    return response


@bp.route("/queue/<int:pk>/cancel", methods=["POST"])
@login_required
def cancel_queued(pk: int):
    """Убрать запись из очереди. Только пока она не начала обрабатываться."""
    row = db.get_or_404(AnalysisResult, pk)
    if not _can_view(row):
        abort(404)
    image_path = row.image_path

    result = db.session.execute(
        delete(AnalysisResult)
        .where(AnalysisResult.id == pk, AnalysisResult.status == Status.QUEUED)
        .execution_options(synchronize_session=False)
    )
    db.session.commit()

    if result.rowcount:
        remove_image_files([image_path])
        flash("Убрано из очереди.", "success")
    else:
        flash("Анализ уже обрабатывается или завершён — отменить его нельзя.", "info")
    return _redirect_back("analyzer.dashboard")


# ----------------------------------------------------------------------------
# История
# ----------------------------------------------------------------------------
@bp.route("/history/")
@login_required
def history():
    q = request.args.get("q", "").strip()
    page = paginate(_own_results(q), per_page=12)

    # Запрос из JS-фильтра (см. static/js/list-filter.js): отдаём только фрагмент
    # с таблицей/пагинацией, без перерисовки всей страницы.
    if request.headers.get("X-Requested-With") == "fetch":
        html = render_template(
            "analyzer/_history_results.html",
            page=page,
            can_delete=current_user.is_panel_staff,
        )
        return jsonify(html=html, total=page.total, pages=page.pages, page_num=page.page)

    new_count = db.session.scalar(
        select(func.count(AnalysisResult.id)).where(
            AnalysisResult.user_id == current_user.id,
            AnalysisResult.status == Status.DONE,
            AnalysisResult.is_new.is_(True),
        )
    )
    return render_template("analyzer/history.html", page=page, new_count=new_count, q=q)

@bp.route("/history/mark-seen/", methods=["POST"])
@login_required
def mark_seen():
    """Сбросить метку «новое» у ВСЕХ своих анализов сразу, без просмотра каждого."""
    result = db.session.execute(
        update(AnalysisResult)
        .where(AnalysisResult.user_id == current_user.id, AnalysisResult.is_new.is_(True))
        .values(is_new=False)
        .execution_options(synchronize_session=False)
    )
    db.session.commit()
    if result.rowcount:
        flash(f"Метки «новое» сброшены: {plural(result.rowcount, ('запись', 'записи', 'записей'))}.", "success")
    else:
        flash("Новых меток нет.", "info")
    return _redirect_back("analyzer.history")

@bp.route("/history/delete/", methods=["POST"])
@staff_required
def delete_selected():
    """Удалить выбранные записи истории (только админы)."""
    ids = [int(x) for x in request.form.getlist("ids") if x.isdigit()][:1000]
    if not ids:
        flash("Ничего не выбрано.", "info")
    else:
        count = delete_finished(AnalysisResult.id.in_(ids))
        flash(f"Удалено записей: {count}." if count else "Нечего удалять.", "success" if count else "info")
    return _redirect_back("analyzer.history")


@bp.route("/result/<int:pk>/delete/", methods=["POST"])
@staff_required
def delete_result(pk: int):
    """Удалить один результат вместе с изображением (только админы)."""
    result = db.get_or_404(AnalysisResult, pk)
    if result.is_pending:
        flash("Анализ ещё не завершён — дождитесь результата или отмените его в очереди.", "error")
        return redirect(url_for("analyzer.result_detail", pk=pk))
    delete_finished(AnalysisResult.id == pk)
    flash("Результат удалён.", "success")
    return _redirect_back("analyzer.history")


def _normalize_signals(raw) -> list[dict]:
    """Приводим сигналы к списку словарей id/category/detail (модель может вернуть что угодно)."""
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if isinstance(item, dict):
            out.append(
                {
                    "id": item.get("id", ""),
                    "category": item.get("category", ""),
                    "detail": item.get("detail", ""),
                }
            )
        elif item not in (None, ""):
            out.append({"id": "", "category": "", "detail": str(item)})
    return out


@bp.route("/result/<int:pk>/")
@login_required
def result_detail(pk: int):
    result = db.get_or_404(AnalysisResult, pk)
    if not _can_view(result):
        abort(404)

    if result.is_new and result.user_id == current_user.id:
        result.is_new = False
        db.session.commit()

    signals, rationale, recommendation, text_on_image, context_text, raw_text = (
        [], "", "", "", "", ""
    )
    report = result.raw_report if isinstance(result.raw_report, dict) else {}
    if not result.is_error:
        raw_text = report.get("_raw", "") or ""
        if not raw_text:
            signals = _normalize_signals(report.get("signals"))
            rationale = report.get("rationale", "")
            recommendation = report.get("recommendation", "")
            text_on_image = report.get("text_on_image", "")
            context_text = report.get("context", "")

    return render_template(
        "analyzer/result_detail.html",
        result=result,
        signals=signals,
        rationale=rationale,
        recommendation=recommendation,
        text_on_image=text_on_image,
        context_text=context_text,
        raw_text=raw_text,
    )


@bp.route("/media/<path:filename>")
@login_required
def media(filename: str):
    """Отдача загруженных файлов. В отличие от Django-версии — только владельцу и админам."""
    result = db.session.scalars(
        select(AnalysisResult).where(AnalysisResult.image_path == filename)
    ).first()
    if result is None or not _can_view(result):
        abort(404)
    return send_from_directory(current_app.config["UPLOAD_FOLDER"], filename)


MAX_MODEL_LEN = 200


def _effective_backend(status: dict, target_backend: str) -> tuple[str, bool]:
    """Куда реально пойдёт анализ. Повторяет логику vision_analyzer_server.py.

    * Явный выбор (?backend=...) — сервер идёт именно туда и НЕ переключается на другой.
    * Без явного выбора — бэкенд по умолчанию; если к нему нельзя подключиться,
      сервер автоматически пробует второй бэкенд (фолбэк).

    Возвращает (бэкенд, это_фолбэк). Пустая строка — ни один бэкенд не доступен.
    """
    if target_backend:
        return target_backend, False

    backends = status.get("backends") or {}
    default = status.get("default_backend", "")
    default_ok = status.get("default_backend_ok")
    if default_ok is None:  # старые версии сервера не отдают default_backend_ok
        default_ok = bool((backends.get(default) or {}).get("ok"))
    if default_ok:
        return default, False

    other = "ollama" if default == "vllm" else "vllm"
    if (backends.get(other) or {}).get("ok"):
        return other, True
    return "", False


def _render_health(bound_forms: dict | None = None, status_code: int = 200):
    """Страница статуса. Настройки (модель, параметры) видны и доступны только админам."""
    bound_forms = bound_forms or {}
    can_configure = current_user.is_panel_staff

    try:
        status = check_health()
        error = None
    except VisionApiError as exc:
        status = None
        error = str(exc)

    cards = []
    sampling_error = None
    target_backend, target_model = get_analysis_target()
    effective_backend, via_fallback = _effective_backend(status, target_backend) if status else ("", False)
    target_down = bool(
        status
        and target_backend
        and not ((status.get("backends") or {}).get(target_backend) or {}).get("ok")
    )

    if status is not None:
        sampling = {}
        if can_configure:
            try:
                sampling = get_sampling()
            except VisionApiError as exc:
                sampling_error = str(exc)

        for name, info in (status.get("backends") or {}).items():
            info = info if isinstance(info, dict) else {}
            available = bool(info.get("ok"))
            card = {
                "name": name,
                "info": info,
                "available": available,
                "configurable": can_configure and available and name in BACKENDS,
                "is_active": name == effective_backend,
                "via_fallback": via_fallback and name == effective_backend,
                "active_model": target_model if target_backend == name else "",
                "models": [],
                "models_error": None,
                "form": None,
            }
            if card["configurable"]:
                try:
                    card["models"] = get_models(name)
                except VisionApiError as exc:
                    card["models_error"] = str(exc)
                # сохранённой модели может уже не быть в списке — не теряем её из виду
                if card["active_model"] and card["models"] and card["active_model"] not in card["models"]:
                    card["models"] = [card["active_model"], *card["models"]]

                form = bound_forms.get(name)
                if form is None:
                    initial = {k: v for k, v in sampling.items() if v is not None and k in SAMPLING_FORMS[name].SPEC}
                    if sampling.get("think") is not None:
                        initial["think"] = str(sampling["think"]).lower()   # True→"true", "high"→"high"
                    form = SAMPLING_FORMS[name](formdata=None, prefix=name, data=initial)
                card["form"] = form
            cards.append(card)

    html = render_template(
        "analyzer/health.html",
        status=status,
        error=error,
        cards=cards,
        can_configure=can_configure,
        sampling_error=sampling_error,
        target_backend=target_backend,
        target_model=target_model,
        effective_backend=effective_backend,
        via_fallback=via_fallback,
        target_down=target_down,
    )
    return html, status_code


@bp.route("/health/")
@login_required
def health():
    return _render_health()


def _require_backend(name: str) -> None:
    if name not in BACKENDS:
        abort(404)


def _backend_is_available(name: str) -> bool:
    try:
        info = (check_health().get("backends") or {}).get(name) or {}
    except VisionApiError as exc:
        flash(str(exc), "error")
        return False
    if not info.get("ok"):
        flash(f"Бэкенд «{name}» сейчас недоступен.", "error")
        return False
    return True


@bp.route("/health/backend/<name>/model", methods=["POST"])
@staff_required
def save_model(name: str):
    """Выбрать бэкенд и модель, на которых будут выполняться анализы. Пустая модель = авто."""
    _require_backend(name)
    model = request.form.get("model", "").strip()

    if len(model) > MAX_MODEL_LEN:
        flash("Слишком длинное название модели.", "error")
        return redirect(url_for("analyzer.health"))
    if not _backend_is_available(name):
        return redirect(url_for("analyzer.health"))

    if model:
        try:
            known = get_models(name)
        except VisionApiError:
            known = []  # список недоступен — не блокируем, сервер сам отклонит неизвестную модель
        if known and model not in known:
            flash(f"Модель «{model}» не найдена у бэкенда «{name}».", "error")
            return redirect(url_for("analyzer.health"))

    set_analysis_target(name, model)
    flash(
        f"Анализы будут выполняться на «{name}», модель: {model or 'авто (определяет сервер)'}.",
        "success",
    )
    return redirect(url_for("analyzer.health"))


@bp.route("/health/reset-target", methods=["POST"])
@staff_required
def reset_target():
    """Вернуться к выбору сервера: бэкенд по умолчанию и автоопределение модели."""
    clear_analysis_target()
    flash("Выбор сброшен: используется бэкенд и модель по умолчанию на сервере.", "success")
    return redirect(url_for("analyzer.health"))


@bp.route("/health/backend/<name>/sampling", methods=["POST"])
@staff_required
def save_sampling(name: str):
    """Изменить параметры генерации. На сервере они общие (POST /sampling)."""
    _require_backend(name)
    form = SAMPLING_FORMS[name](prefix=name)

    if not form.validate_on_submit():
        flash("Параметры не сохранены — исправьте ошибки в форме.", "error")
        return _render_health(bound_forms={name: form}, status_code=400)

    if not form.values:
        flash("Нечего сохранять: все поля пустые (пустое поле означает «не менять»).", "info")
        return redirect(url_for("analyzer.health"))

    try:
        set_sampling(form.values)
    except VisionApiError as exc:
        flash(str(exc), "error")
        return redirect(url_for("analyzer.health"))

    changed = ", ".join(f"{k}={v}" for k, v in form.values.items())
    flash(f"Параметры генерации обновлены на сервере: {changed}.", "success")
    return redirect(url_for("analyzer.health"))