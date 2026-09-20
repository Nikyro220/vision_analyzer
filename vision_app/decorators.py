from functools import wraps

from flask import current_app, flash, redirect, url_for
from flask_login import current_user

from .models import Role


def role_required(*allowed_roles):
    """Пускает только пользователей с одной из указанных ролей.

    Неавторизованных отправляет на логин. Заблокированных всегда отправляет на
    страницу блокировки, даже если 'blocked' случайно оказался в allowed_roles.
    """

    def decorator(view_func):
        @wraps(view_func)
        def wrapped(*args, **kwargs):
            if not current_user.is_authenticated:
                return current_app.login_manager.unauthorized()
            if current_user.role == Role.BLOCKED:
                return redirect(url_for("accounts.blocked"))
            if current_user.role not in allowed_roles:
                flash("Недостаточно прав для доступа к этой странице.", "error")
                return redirect(url_for("analyzer.dashboard"))
            return view_func(*args, **kwargs)

        return wrapped

    return decorator


def staff_required(view_func):
    """Доступ для admin и head_admin (панель управления)."""
    return role_required(Role.ADMIN, Role.HEAD_ADMIN)(view_func)


def head_admin_required(view_func):
    """Доступ только для head_admin."""
    return role_required(Role.HEAD_ADMIN)(view_func)
