from functools import wraps

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect


def role_required(*allowed_roles):
    """Пускает только пользователей с одной из указанных ролей.

    Заблокированных пользователей всегда отправляет на страницу блокировки,
    даже если 'blocked' случайно оказался в allowed_roles.
    """

    def decorator(view_func):
        @wraps(view_func)
        @login_required(login_url="accounts:login")
        def _wrapped(request, *args, **kwargs):
            user = request.user
            if user.role == "blocked":
                return redirect("accounts:blocked")
            if user.role not in allowed_roles:
                messages.error(request, "Недостаточно прав для доступа к этой странице.")
                return redirect("analyzer:dashboard")
            return view_func(request, *args, **kwargs)

        return _wrapped

    return decorator


def staff_required(view_func):
    """Доступ для admin и head_admin (панель управления)."""
    return role_required("admin", "head_admin")(view_func)


def head_admin_required(view_func):
    """Доступ только для head_admin."""
    return role_required("head_admin")(view_func)
