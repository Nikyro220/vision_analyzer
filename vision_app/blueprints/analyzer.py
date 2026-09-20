"""Загрузка изображений, история, детальный результат, статус сервера анализа."""

import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import (
    Blueprint,
    abort,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from flask_login import current_user, login_required
from sqlalchemy import select

from ..decorators import staff_required
from ..extensions import db
from ..forms import SAMPLING_FORMS, ImageUploadForm
from ..models import AnalysisResult
from ..services import (
    BACKENDS,
    VisionApiError,
    analyze_image,
    check_health,
    get_models,
    get_sampling,
    set_sampling,
)
from ..settings_store import clear_analysis_target, get_analysis_target, set_analysis_target
from ..utils import paginate

bp = Blueprint("analyzer", __name__)


def _own_results():
    return (
        select(AnalysisResult)
        .where(AnalysisResult.user_id == current_user.id)
        .order_by(AnalysisResult.created_at.desc(), AnalysisResult.id.desc())
    )


def _can_view(result: AnalysisResult) -> bool:
    return result.user_id == current_user.id or current_user.is_panel_staff


@bp.route("/", methods=["GET", "POST"])
@login_required
def dashboard():
    form = ImageUploadForm()

    if form.validate_on_submit():
        upload = form.image.data
        now = datetime.now(timezone.utc)

        # Файл на диске хранится под случайным именем, оригинальное имя — только в БД.
        rel_path = f"uploads/{now:%Y/%m/%d}/{uuid.uuid4().hex}{form.image_ext}"
        abs_path = Path(current_app.config["UPLOAD_FOLDER"]) / rel_path
        abs_path.parent.mkdir(parents=True, exist_ok=True)
        abs_path.write_bytes(form.image_bytes)

        result = AnalysisResult(
            user_id=current_user.id,
            image_path=rel_path,
            original_name=(upload.filename or "")[:255],
        )
        db.session.add(result)
        db.session.commit()

        try:
            target_backend, target_model = get_analysis_target()
            outcome = analyze_image(
                form.image_bytes,
                form.image_mime,
                lang="ru",
                backend=target_backend,
                model=target_model,
            )
            result.backend = outcome.backend
            result.risk_level = outcome.risk_level
            result.needs_human_review = outcome.needs_human_review
            result.description = outcome.description
            result.raw_report = outcome.raw_report
            db.session.commit()
            flash("Изображение проанализировано.", "success")
        except VisionApiError as exc:
            result.error = str(exc)
            db.session.commit()
            flash(str(exc), "error")

        return redirect(url_for("analyzer.result_detail", pk=result.id))

    recent = db.session.scalars(_own_results().limit(6)).all()
    target_backend, target_model = get_analysis_target()
    return render_template(
        "analyzer/dashboard.html",
        form=form,
        recent=recent,
        target_backend=target_backend,
        target_model=target_model,
    )


@bp.route("/history/")
@login_required
def history():
    page = paginate(_own_results(), per_page=12)
    return render_template("analyzer/history.html", page=page)


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