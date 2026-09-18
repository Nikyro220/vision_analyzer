from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.http import Http404
from django.shortcuts import get_object_or_404, redirect, render

from accounts.models import Role

from .forms import ImageUploadForm
from .models import AnalysisResult
from .services import VisionApiError, analyze_image, check_health


def _guard_blocked(request):
    """Возвращает redirect на страницу блокировки, если пользователь заблокирован."""
    if request.user.role == Role.BLOCKED:
        return redirect("accounts:blocked")
    return None


@login_required(login_url="accounts:login")
def dashboard_view(request):
    guard = _guard_blocked(request)
    if guard:
        return guard

    if request.method == "POST":
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded = form.cleaned_data["image"]
            result = AnalysisResult.objects.create(
                user=request.user,
                image=uploaded,
                original_name=uploaded.name,
            )
            uploaded.seek(0)
            image_bytes = uploaded.read()
            mime_type = getattr(uploaded, "content_type", "") or "image/jpeg"

            try:
                outcome = analyze_image(image_bytes, mime_type, lang="ru")
                result.backend = outcome.backend
                result.risk_level = outcome.risk_level
                result.needs_human_review = outcome.needs_human_review
                result.description = outcome.description
                result.raw_report = outcome.raw_report
                result.save()
                messages.success(request, "Изображение проанализировано.")
            except VisionApiError as exc:
                result.error = str(exc)
                result.save()
                messages.error(request, str(exc))

            return redirect("analyzer:result_detail", pk=result.pk)
    else:
        form = ImageUploadForm()

    recent = AnalysisResult.objects.filter(user=request.user)[:6]
    return render(
        request,
        "analyzer/dashboard.html",
        {"form": form, "recent": recent},
    )


@login_required(login_url="accounts:login")
def history_view(request):
    guard = _guard_blocked(request)
    if guard:
        return guard

    queryset = AnalysisResult.objects.filter(user=request.user)
    paginator = Paginator(queryset, 12)
    page = paginator.get_page(request.GET.get("page"))
    return render(request, "analyzer/history.html", {"page": page})


@login_required(login_url="accounts:login")
def result_detail_view(request, pk: int):
    guard = _guard_blocked(request)
    if guard:
        return guard

    result = get_object_or_404(AnalysisResult, pk=pk)
    if result.user_id != request.user.id and not request.user.is_panel_staff:
        raise Http404("Результат не найден.")

    signals = []
    rationale = ""
    recommendation = ""
    text_on_image = ""
    context = ""
    if not result.is_error and not result.raw_report.get("_raw"):
        signals = result.raw_report.get("signals", [])
        rationale = result.raw_report.get("rationale", "")
        recommendation = result.raw_report.get("recommendation", "")
        text_on_image = result.raw_report.get("text_on_image", "")
        context = result.raw_report.get("context", "")

    return render(
        request,
        "analyzer/result_detail.html",
        {
            "result": result,
            "signals": signals,
            "rationale": rationale,
            "recommendation": recommendation,
            "text_on_image": text_on_image,
            "context": context,
        },
    )


@login_required(login_url="accounts:login")
def health_view(request):
    guard = _guard_blocked(request)
    if guard:
        return guard

    try:
        status = check_health()
        error = None
    except VisionApiError as exc:
        status = None
        error = str(exc)

    return render(request, "analyzer/health.html", {"status": status, "error": error})
