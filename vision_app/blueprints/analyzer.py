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
    send_file,
    send_from_directory,
    url_for,
)
from flask_babel import gettext, ngettext
from flask_login import current_user, login_required
from sqlalchemy import delete, func, or_, select, update

from ..config import Config, conf
from ..decorators import staff_required
from .. import credentials, image_dedup
from ..extensions import db
from ..forms import ImageUploadForm, provider_settings_form_class, settings_initial
from ..history import delete_finished, remove_image_files
from ..models import RISK_LABELS, STATUS_LABELS, AnalysisResult, Status
from ..queue_worker import wake_worker
from ..services import (
    ProviderInfo,
    VisionApiError,
    check_health,
    invalidate_health,
    get_models,
    get_providers,
    invalidate_models_cache,
    invalidate_providers,
    models_cached_at,
    reset_provider_settings,
    set_provider_settings,
)
from ..settings_store import (
    clear_analysis_target, clear_chat_target, get_analysis_target, get_chat_target, has_chat_target,
    set_analysis_target, set_chat_target,
)
from ..thumbs import ensure_thumb
from ..utils import is_safe_next, local_dt, paginate, query_to_id

bp = Blueprint("analyzer", __name__)


def _own_results(q: str = ""):
    """История пользователя — только ЗАВЕРШЁННЫЕ анализы (очередь показывается отдельно).

    ``q`` — необязательный поиск по имени файла или по номеру анализа («42», «#42»).
    """
    stmt = select(AnalysisResult).where(
        AnalysisResult.user_id == current_user.id, AnalysisResult.status == Status.DONE
    )
    if q:
        match = AnalysisResult.original_name.icontains(q, autoescape=True)
        number = query_to_id(q)
        if number is not None:
            match = or_(match, AnalysisResult.id == number)
        stmt = stmt.where(match)
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
                "name": row.original_name or gettext("account.history.unnamed"),
                "status": row.status,
                "status_label": str(STATUS_LABELS.get(row.status, row.status)),
                "ahead": position.get(row.id, 0),
                "elapsed": max(int((now - _as_utc(since)).total_seconds()), 0),
                "cancel_url": url_for("analyzer.cancel_queued", pk=row.id) if row.status == Status.QUEUED else "",
            }
        )

    recent = [
        {
            "id": row.id,
            "name": row.original_name or gettext("account.history.unnamed"),
            "risk_level": row.risk_level,
            "risk_label": str(RISK_LABELS.get(row.risk_level, row.risk_level)),
            "error": row.is_error,
            "is_new": row.is_new,
            "date": local_dt(row.created_at, "short"),
            "url": url_for("analyzer.result_detail", pk=row.id),
            "thumb_url": url_for("analyzer.thumb", filename=row.image_path) if row.image_path else "",
        }
        for row in db.session.scalars(_own_results().limit(conf("DASHBOARD_RECENT_LIMIT"))).all()
    ]
    return {"pending": pending, "recent": recent, "queue_total": len(pending_ids)}


def _enqueue_uploads(form: ImageUploadForm):
    """Кладёт проверенные файлы в очередь и сразу возвращает пользователя на страницу."""
    from ..queue_worker import queue_limit_hit

    hit = queue_limit_hit(current_user.id, len(form.accepted) + len(form.link_urls))
    if hit:
        scope, now_count, limit = hit
        if scope == "user":
            msg = gettext("flash.queue.user_limit", count=now_count, limit=limit)
        else:
            msg = gettext("flash.queue.total_limit", count=now_count, limit=limit)
        flash(msg, "error")
        return redirect(url_for("analyzer.dashboard"))

    now = datetime.now(timezone.utc)
    root = Path(current_app.config["UPLOAD_FOLDER"])
    written: list[Path] = []
    added: list[AnalysisResult] = []
    try:
        for item in form.accepted:
            # Файл на диске хранится под случайным именем, оригинальное имя — только в БД.
            rel_path = f"{Config.UPLOADS_DIR}/{now:%Y/%m/%d}/{uuid.uuid4().hex}{item.ext}"
            abs_path = root / rel_path
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            abs_path.write_bytes(item.data)
            written.append(abs_path)
            row = AnalysisResult(
                user_id=current_user.id,
                image_path=rel_path,
                original_name=item.filename[:255],
                image_mime=item.mime,
                image_hash=image_dedup.sha256_bytes(item.data),
                caption=item.caption,
                status=Status.QUEUED,
            )
            db.session.add(row)
            added.append(row)
        # Ссылки на пост: картинку скачивает сервер анализа при обработке (queue_worker), поэтому
        # пока image_path пуст. image_hash="" — чтобы фоновый подсчёт хешей не принял запись за потерянный файл.
        link_rows: list[AnalysisResult] = []
        for link in form.link_urls:
            row = AnalysisResult(
                user_id=current_user.id,
                image_path="",
                original_name=link[:255],
                image_mime="",
                image_hash="",
                source_url=link,
                status=Status.QUEUED,
            )
            db.session.add(row)
            link_rows.append(row)
        db.session.commit()
    except Exception:
        db.session.rollback()
        for path in written:
            path.unlink(missing_ok=True)
        raise

    wake_worker(current_app)
    numbers = ", ".join(gettext("common.id_tag", id=row.id) for row in [*added, *link_rows])  # id доступны после commit
    parts = []
    if added:
        parts.append(ngettext("plural.files.one", "plural.files.other", len(added), n=len(added)))
    if link_rows:
        parts.append(ngettext("plural.links.one", "plural.links.other", len(link_rows), n=len(link_rows)))
    flash(gettext("flash.queue.added", items=gettext("flash.queue.and").join(parts), numbers=numbers), "success")
    shown = conf("REJECTED_FILES_SHOWN")
    for name, reason in form.rejected[:shown]:
        flash(gettext("flash.upload.skipped_file", name=name, reason=reason), "warning")
    if len(form.rejected) > shown:
        flash(gettext("flash.upload.skipped_more", count=len(form.rejected) - shown), "warning")
    for link, reason in form.rejected_links[:shown]:
        flash(gettext("flash.upload.skipped_link", link=link, reason=reason), "warning")
    if len(form.rejected_links) > shown:
        flash(gettext("flash.upload.skipped_more_links", count=len(form.rejected_links) - shown), "warning")
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
        flash(gettext("flash.queue.removed"), "success")
    else:
        flash(gettext("flash.queue.cannot_cancel"), "info")
    return _redirect_back("analyzer.dashboard")


# ----------------------------------------------------------------------------
# История
# ----------------------------------------------------------------------------
@bp.route("/history/")
@login_required
def history():
    q = request.args.get("q", "").strip()
    page = paginate(_own_results(q), per_page=conf("HISTORY_PER_PAGE"))

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
        flash(gettext("flash.history.new_cleared", items=ngettext("plural.records.one", "plural.records.other", result.rowcount, n=result.rowcount)), "success")
    else:
        flash(gettext("flash.history.no_new"), "info")
    return _redirect_back("analyzer.history")

@bp.route("/history/delete/", methods=["POST"])
@staff_required
def delete_selected():
    """Удалить выбранные записи истории (только админы)."""
    ids = [int(x) for x in request.form.getlist("ids") if x.isdigit()][: conf("BULK_DELETE_MAX_IDS")]
    if not ids:
        flash(gettext("flash.history.none_selected"), "info")
    else:
        count = delete_finished(AnalysisResult.id.in_(ids))
        flash(gettext("flash.history.deleted", count=count) if count else gettext("flash.history.nothing_to_delete"), "success" if count else "info")
    return _redirect_back("analyzer.history")


@bp.route("/result/<int:pk>/delete/", methods=["POST"])
@staff_required
def delete_result(pk: int):
    """Удалить один результат вместе с изображением (только админы)."""
    result = db.get_or_404(AnalysisResult, pk)
    if result.is_pending:
        flash(gettext("flash.result.pending"), "error")
        return redirect(url_for("analyzer.result_detail", pk=pk))
    delete_finished(AnalysisResult.id == pk)
    flash(gettext("flash.result.deleted"), "success")
    return _redirect_back("analyzer.history")


RENAME_MAX_LEN = 255  # длина колонки AnalysisResult.original_name


@bp.route("/result/<int:pk>/rename/", methods=["POST"])
@login_required
def rename_result(pk: int):
    """Изменить название записи (original_name). Автор — свои, админы — любые."""
    result = db.get_or_404(AnalysisResult, pk)
    if not _can_view(result):
        abort(404)  # чужую запись обычному пользователю не раскрываем
    if not result.can_rename(current_user):
        abort(403)

    # Схлопываем переводы строк/табы в пробелы: имя показывается в одну строку.
    name = " ".join(request.form.get("name", "").split())
    if not name:
        flash(gettext("flash.rename.empty"), "error")
    elif len(name) > RENAME_MAX_LEN:
        flash(gettext("flash.rename.too_long", max=RENAME_MAX_LEN), "error")
    elif name == result.original_name:
        flash(gettext("flash.rename.unchanged"), "info")
    else:
        result.original_name = name
        db.session.commit()
        flash(gettext("flash.rename.done"), "success")
    return redirect(url_for("analyzer.result_detail", pk=pk))


def _normalize_signals(raw) -> list[dict]:
    """Приводим сигналы к списку словарей category/detail (модель может вернуть что угодно)."""
    if not isinstance(raw, list):
        return []
    out = []
    for item in raw:
        if isinstance(item, dict):
            out.append(
                {
                    "category": item.get("category", ""),
                    "detail": item.get("detail", ""),
                }
            )
        elif item not in (None, ""):
            out.append({"category": "", "detail": str(item)})
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


def _require_file_access(filename: str) -> None:
    """404, если файла нет в БД или он чужой (доступ только владельцу и админам)."""
    result = db.session.scalars(
        select(AnalysisResult).where(AnalysisResult.image_path == filename)
    ).first()
    if result is None or not _can_view(result):
        abort(404)


@bp.route("/media/<path:filename>")
@login_required
def media(filename: str):
    """Отдача загруженных файлов. В отличие от Django-версии — только владельцу и админам."""
    _require_file_access(filename)
    return send_from_directory(current_app.config["UPLOAD_FOLDER"], filename)


@bp.route("/thumb/<path:filename>")
@login_required
def thumb(filename: str):
    """Миниатюра загруженного файла (JPEG ~96x96, кэшируется на диске). Права те же, что у /media/."""
    _require_file_access(filename)
    path = ensure_thumb(Path(current_app.config["UPLOAD_FOLDER"]), filename)
    if path is None:
        abort(404)
    resp = send_file(path, mimetype="image/jpeg", max_age=conf("THUMB_CACHE_SECONDS"), conditional=True)
    resp.cache_control.public = False  # картинки приватные: только в браузере пользователя
    resp.cache_control.private = True
    return resp



def _effective_backend(status: dict, target_backend: str, fallbacks: dict[str, str | None]) -> tuple[str, bool]:
    """Куда реально пойдёт анализ. Повторяет логику сервера анализа.

    * Явный выбор (?backend=...) — сервер идёт именно туда и НЕ переключается на другой.
    * Без явного выбора — бэкенд по умолчанию; если к нему нельзя подключиться, сервер
      автоматически пробует его фолбэк (Provider.fallback; у облачных провайдеров его нет).
      ``fallbacks`` — {провайдер: его фолбэк}, из GET /providers.

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

    other = fallbacks.get(default)
    if other and (backends.get(other) or {}).get("ok"):
        return other, True
    return "", False


_TABS = ("analysis", "chat")


def _tab() -> str:
    """Вкладка страницы статуса: analysis (по умолчанию) или chat. Берётся из ?tab=."""
    tab = request.values.get("tab", "")
    return tab if tab in _TABS else "analysis"


def _back():
    return redirect(url_for("analyzer.health", tab=_tab()))


def _render_health(bound_forms: dict | None = None, status_code: int = 200):
    """Страница статуса. Настройки (модель, параметры, ключ API) видны и доступны только админам.

    Список карточек строится по GET /providers, а не по зашитому перечню: провайдер,
    добавленный на сервере анализа, появляется здесь сам. Не настроенные провайдеры
    (например, Gemini без ключа) показываются тоже — иначе ключ было бы негде ввести."""
    bound_forms = bound_forms or {}
    can_configure = current_user.is_panel_staff

    try:
        status = check_health(force=request.values.get("refresh") == "1")
        error = None
    except VisionApiError as exc:
        status = None
        error = str(exc)

    providers: list[ProviderInfo] = []
    if status is not None:
        try:
            providers = get_providers()
        except VisionApiError as exc:
            status, error = None, str(exc)

    tab = _tab()  # только какая вкладка открыта первой; переключаются они в браузере, без запросов
    health_backends = (status or {}).get("backends") or {}
    fallbacks = {p.name: p.fallback for p in providers}

    # Всё, что зависит от вкладки: выбранный бэкенд/модель, куда реально уйдёт запрос, варианты селекта.
    # У чата своего выбора может не быть — тогда он идёт туда же, куда анализы (get_chat_target это делает).
    tabs = {}
    for name_, is_chat in (("analysis", False), ("chat", True)):
        backend, model = get_chat_target() if is_chat else get_analysis_target()
        follows_analysis = is_chat and not has_chat_target()
        own_backend = "" if follows_analysis else backend
        effective, via_fb = _effective_backend(status, backend, fallbacks) if status else ("", False)
        tabs[name_] = {
            "backend": backend,
            "model": model,
            "follows_analysis": follows_analysis,
            "effective_backend": effective,
            "via_fallback": via_fb,
            "target_down": bool(status and backend and not (health_backends.get(backend) or {}).get("ok")),
            "options": [
                {
                    "name": p.name,
                    "title": p.title,
                    "configured": p.configured,
                    "available": bool((health_backends.get(p.name) or {}).get("ok")),
                    "selected": p.name == own_backend,
                }
                for p in providers
            ],
        }

    cards = []

    if status is not None:
        for provider in providers:
            name = provider.name
            info = health_backends.get(name)
            if not isinstance(info, dict):
                # /health скрывает не настроенных провайдеров — показываем их как «не настроен»
                info = {"ok": False, "endpoint": provider.endpoint}
            available = bool(info.get("ok"))
            card = {
                "name": name,
                "title": provider.title,
                "info": info,
                "available": available,
                "configured": provider.configured,
                # Ключ API: форма только для админов; значение наружу не отдаётся — лишь «задан / последние 4».
                "credential": provider.credential if can_configure else None,
                "credential_status": credentials.status(name) if can_configure and provider.credential else None,
                "credentials_available": credentials.is_available(),
                "configurable": can_configure and available,
                "by_tab": {
                    t: {
                        "is_active": name == ctx["effective_backend"],
                        "via_fallback": ctx["via_fallback"] and name == ctx["effective_backend"],
                        "active_model": ctx["model"] if ctx["backend"] == name else "",
                    }
                    for t, ctx in tabs.items()
                },
                "models": [],
                "models_error": None,
                "models_cached_at": None,
                "form": None,
            }
            if card["configurable"]:
                # Страница не должна ждать провайдера: берём список из кэша БД (для облачных
                # провайдеров), а обновляется он кнопкой «Обновить список» (refresh_models).
                try:
                    card["models"] = get_models(name, cached_only=True)
                    card["models_cached_at"] = models_cached_at(name)
                except VisionApiError as exc:
                    card["models_error"] = str(exc)
                # сохранённой модели может уже не быть в списке — не теряем её из виду
                for t in card["by_tab"].values():
                    m = t["active_model"]
                    t["models"] = [m, *card["models"]] if m and card["models"] and m not in card["models"] else card["models"]

                if provider.settings:
                    # Форма — по описанию параметров ЭТОГО провайдера (у каждого свой набор и значения).
                    form = bound_forms.get(name)
                    if form is None:
                        form = provider_settings_form_class(provider.settings)(
                            formdata=None, prefix=name, data=settings_initial(provider.settings),
                        )
                    card["form"] = form
            cards.append(card)

    html = render_template(
        "analyzer/health.html",
        status=status,
        error=error,
        cards=cards,
        tab=tab,
        tabs=tabs,
        can_configure=can_configure,
    )
    return html, status_code


@bp.route("/health/")
@login_required
def health():
    return _render_health()


def _require_backend(name: str) -> ProviderInfo:
    """Провайдер по имени из URL; 404, если сервер анализа такого не знает (список — GET /providers)."""
    try:
        provider = next((p for p in get_providers() if p.name == name), None)
    except VisionApiError as exc:
        flash(str(exc), "error")
        abort(503)
    if provider is None:
        abort(404)
    return provider


def _backend_is_available(name: str) -> bool:
    try:
        info = (check_health(force=True).get("backends") or {}).get(name) or {}
    except VisionApiError as exc:
        flash(str(exc), "error")
        return False
    if not info.get("ok"):
        flash(gettext("flash.backend.unavailable", name=name), "error")
        return False
    return True


def _validated_model(name: str) -> str | None:
    """Модель из формы, проверенная для бэкенда name; None — отказ (flash уже показан).
    Пустая модель = авто."""
    _require_backend(name)
    model = request.form.get("model", "").strip()

    if len(model) > conf("MODEL_NAME_MAX_LEN"):
        flash(gettext("flash.model.too_long"), "error")
        return None
    if not _backend_is_available(name):
        return None

    if model:
        try:
            known = get_models(name, cached_only=True)  # из кэша — без обращения к провайдеру
            if known and model not in known:
                known = get_models(name, force=True)  # в кэше нет: перед отказом сверяемся с живым списком
        except VisionApiError:
            known = []  # список недоступен — не блокируем, сервер сам отклонит неизвестную модель
        if known and model not in known:
            flash(gettext("flash.model.not_found", model=model, name=name), "error")
            return None
    return model


@bp.route("/health/backend/<name>/model", methods=["POST"])
@staff_required
def save_model(name: str):
    """Выбрать бэкенд и модель для текущей вкладки (анализ или чат). Пустая модель = авто."""
    model = _validated_model(name)
    if model is None:
        return _back()

    shown = model or gettext("health.option.auto.lower")
    if _tab() == "chat":
        set_chat_target(name, model)
        flash(gettext("flash.backend.chat_set", name=name, model=shown), "success")
    else:
        set_analysis_target(name, model)
        flash(gettext("flash.backend.analysis_set", name=name, model=shown), "success")
    return _back()


@bp.route("/health/used-backend", methods=["POST"])
@staff_required
def save_used_backend():
    """Выбрать используемый бэкенд для текущей вкладки (анализ или чат).

    Пустое значение — сброс: у анализа бэкенд снова выбирает сервер, а чат идёт туда же, куда
    анализы. Модель сохраняется, если бэкенд тот же, и сбрасывается на «авто», если он сменился —
    конкретную модель выбирают на карточке бэкенда."""
    chat = _tab() == "chat"
    name = request.form.get("backend", "").strip().lower()

    if not name:
        if chat:
            clear_chat_target()
            flash(gettext("flash.backend.chat_cleared"), "success")
        else:
            clear_analysis_target()
            flash(gettext("flash.backend.analysis_cleared"), "success")
        return _back()

    provider = _require_backend(name)
    if not provider.configured:
        flash(gettext("flash.backend.not_configured", title=provider.title), "error")
        return _back()

    current_backend, current_model = (get_chat_target() if has_chat_target() else ("", "")) if chat else get_analysis_target()
    model = current_model if current_backend == name else ""
    (set_chat_target if chat else set_analysis_target)(name, model)
    flash((gettext("flash.backend.chat_uses", title=provider.title) if chat else gettext("flash.backend.analysis_uses", title=provider.title)), "success")

    try:
        ok = ((check_health().get("backends") or {}).get(name) or {}).get("ok")
    except VisionApiError:
        ok = True  # статус неизвестен — не пугаем
    if not ok:
        flash(gettext("flash.backend.down_warning", title=provider.title), "info")
    return _back()


@bp.route("/health/backend/<name>/sampling", methods=["POST"])
@staff_required
def save_sampling(name: str):
    """Изменить параметры генерации ОДНОГО провайдера. Другие провайдеры не затрагиваются."""
    provider = _require_backend(name)
    if not provider.settings:
        abort(404)  # у провайдера нет настраиваемых параметров
    form = provider_settings_form_class(provider.settings)(prefix=name)

    if not form.validate_on_submit():
        flash(gettext("flash.sampling.invalid"), "error")
        return _render_health(bound_forms={name: form}, status_code=400)

    if not form.values:
        flash(gettext("flash.sampling.empty"), "info")
        return _back()

    try:
        set_provider_settings(name, form.values)
    except VisionApiError as exc:
        flash(str(exc), "error")
        return _back()

    changed = ", ".join(f"{k}={'auto' if v is None else v}" for k, v in form.values.items())
    flash(gettext("flash.sampling.saved", title=provider.title, changed=changed), "success")
    return _back()


@bp.route("/health/backend/<name>/sampling/reset", methods=["POST"])
@staff_required
def reset_sampling(name: str):
    """Вернуть параметры генерации ОДНОГО провайдера к значениям по умолчанию."""
    provider = _require_backend(name)
    if not provider.settings:
        abort(404)
    try:
        reset_provider_settings(name)
    except VisionApiError as exc:
        flash(str(exc), "error")
        return _back()
    flash(gettext("flash.sampling.reset", title=provider.title), "success")
    return _back()


@bp.route("/health/backend/<name>/settings", methods=["POST"])
@staff_required
def save_provider_settings(name: str):
    """Сохранить или удалить API-ключ провайдера (например, Gemini).

    Ключ шифруется и кладётся в БД приложения (credentials.py); сервер анализа его не хранит и
    получает в каждом запросе. Значение нигде не показывается и не пишется во флеш/лог: пустое
    поле = «не менять», удалить можно только явным чекбоксом.
    """
    provider = _require_backend(name)
    if provider.credential is None:
        abort(404)  # у этого провайдера нет ключа

    if request.form.get("clear_credential"):
        removed = credentials.clear_key(name)
        invalidate_models_cache(name)
        invalidate_providers()
        invalidate_health()
        flash(gettext("flash.key.erased", title=provider.title) if removed else gettext("flash.key.already_unset"), "success" if removed else "info")
        if removed:
            try:
                is_default = check_health().get("default_backend") == name
            except VisionApiError:
                is_default = False
            if is_default:
                flash(gettext("flash.key.default_lost", title=provider.title), "error")
        return _back()

    raw = request.form.get("credential", "")
    if not raw.strip():
        flash(gettext("flash.key.empty"), "info")
        return _back()

    try:
        credentials.set_key(name, raw, user_id=current_user.id)
    except credentials.CredentialsError as exc:
        flash(str(exc), "error")
        return _back()
    finally:
        raw = ""  # не держим значение дольше нужного

    invalidate_models_cache(name)  # кэш относился к прежнему ключу
    invalidate_providers()
    invalidate_health()

    # Проверяем ключ реальным запросом (/health теперь шлёт его в заголовке).
    try:
        probe = (check_health(force=True).get("backends") or {}).get(name) or {}
    except VisionApiError as exc:
        flash(gettext("flash.key.saved_unchecked", title=provider.title, error=exc), "error")
        return _back()

    if probe.get("ok"):
        flash(gettext("flash.key.saved_ok", title=provider.title), "success")
    else:
        reason = probe.get("error") or gettext("flash.key.no_answer")
        flash(gettext("flash.key.saved_down", title=provider.title, reason=reason), "error")
    return _back()


@bp.route("/health/backend/<name>/models/refresh", methods=["POST"])
@staff_required
def refresh_models(name: str):
    """Обновить кэш списка моделей провайдера (живой запрос — может занять время)."""
    provider = _require_backend(name)
    if not _backend_is_available(name):
        return _back()
    try:
        models = get_models(name, force=True)
    except VisionApiError as exc:
        flash(gettext("flash.models.fetch_failed", title=provider.title, error=exc), "error")
        return _back()
    flash(gettext("flash.models.refreshed", title=provider.title, count=len(models)), "success")
    return _back()
