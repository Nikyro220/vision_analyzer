from django.contrib import messages
from django.contrib.auth import get_user_model
from django.core.paginator import Paginator
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from accounts.decorators import head_admin_required, staff_required
from accounts.models import Role
from analyzer.models import AnalysisResult
from analyzer.services import VisionApiError, check_health

User = get_user_model()


@staff_required
def users_list_view(request):
    query = request.GET.get("q", "").strip()
    role_filter = request.GET.get("role", "").strip()

    queryset = User.objects.all().order_by("-created_at")
    if query:
        queryset = queryset.filter(username__icontains=query)
    if role_filter:
        queryset = queryset.filter(role=role_filter)

    paginator = Paginator(queryset, 20)
    page = paginator.get_page(request.GET.get("page"))

    return render(
        request,
        "panel/users.html",
        {
            "page": page,
            "query": query,
            "role_filter": role_filter,
            "roles": Role.choices,
        },
    )


@staff_required
def user_detail_view(request, pk: int):
    target = get_object_or_404(User, pk=pk)
    assignable = request.user.assignable_roles() if request.user.can_manage(target) else []
    analyses = AnalysisResult.objects.filter(user=target)[:10]

    return render(
        request,
        "panel/user_detail.html",
        {
            "target": target,
            "assignable_roles": assignable,
            "can_manage": request.user.can_manage(target),
            "analyses": analyses,
            "all_roles": Role.choices,
        },
    )


@staff_required
@require_POST
def user_set_role_view(request, pk: int):
    target = get_object_or_404(User, pk=pk)
    new_role = request.POST.get("role", "")

    if not request.user.can_manage(target):
        messages.error(request, "У вас нет прав на изменение этого пользователя.")
        return redirect("panel:user_detail", pk=target.pk)

    if new_role not in request.user.assignable_roles():
        messages.error(request, "Вы не можете назначить эту роль.")
        return redirect("panel:user_detail", pk=target.pk)

    old_role = target.role
    target.role = new_role
    target.save(update_fields=["role"])

    if old_role != new_role:
        messages.success(
            request,
            f"Роль пользователя «{target.username}» изменена: "
            f"{Role(old_role).label} → {Role(new_role).label}.",
        )
    return redirect("panel:user_detail", pk=target.pk)


@staff_required
@require_POST
def user_toggle_block_view(request, pk: int):
    """Быстрая кнопка блокировки/разблокировки из списка пользователей."""
    target = get_object_or_404(User, pk=pk)

    if not request.user.can_manage(target):
        messages.error(request, "У вас нет прав на блокировку этого пользователя.")
        return redirect("panel:users_list")

    if target.role == Role.BLOCKED:
        target.role = Role.USER
        messages.success(request, f"Пользователь «{target.username}» разблокирован.")
    else:
        if target.role in request.user.assignable_roles() or Role.BLOCKED in request.user.assignable_roles():
            target.role = Role.BLOCKED
            messages.success(request, f"Пользователь «{target.username}» заблокирован.")
        else:
            messages.error(request, "У вас нет прав на блокировку этого пользователя.")
            return redirect("panel:users_list")

    target.save(update_fields=["role"])
    return redirect(request.META.get("HTTP_REFERER") or "panel:users_list")


@head_admin_required
def stats_view(request):
    total_users = User.objects.count()
    role_counts = [
        (role, label, User.objects.filter(role=role).count()) for role, label in Role.choices
    ]
    total_analyses = AnalysisResult.objects.count()
    high_risk = AnalysisResult.objects.filter(risk_level="high").count()
    needs_review = AnalysisResult.objects.filter(needs_human_review=True).count()

    try:
        backend_status = check_health()
        backend_error = None
    except VisionApiError as exc:
        backend_status = None
        backend_error = str(exc)

    return render(
        request,
        "panel/stats.html",
        {
            "total_users": total_users,
            "role_counts": role_counts,
            "total_analyses": total_analyses,
            "high_risk": high_risk,
            "needs_review": needs_review,
            "backend_status": backend_status,
            "backend_error": backend_error,
        },
    )


@staff_required
def analyses_list_view(request):
    """Все результаты анализа в системе — для модерации."""
    risk_filter = request.GET.get("risk", "").strip()
    review_only = request.GET.get("review") == "1"

    queryset = AnalysisResult.objects.select_related("user").all()
    if risk_filter:
        queryset = queryset.filter(risk_level=risk_filter)
    if review_only:
        queryset = queryset.filter(needs_human_review=True)

    paginator = Paginator(queryset, 20)
    page = paginator.get_page(request.GET.get("page"))

    return render(
        request,
        "panel/analyses.html",
        {"page": page, "risk_filter": risk_filter, "review_only": review_only},
    )
