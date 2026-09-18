from django.contrib import messages
from django.contrib.auth import login as auth_login
from django.contrib.auth import logout as auth_logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.views import LoginView
from django.shortcuts import redirect, render

from .forms import LoginForm, ProfileForm, RegisterForm
from .models import Role, User


def register_view(request):
    if request.user.is_authenticated:
        return redirect("analyzer:dashboard")

    if request.method == "POST":
        form = RegisterForm(request.POST)
        if form.is_valid():
            user = form.save(commit=False)
            # Первый зарегистрированный в системе пользователь автоматически
            # становится главным админом — иначе панелью некому управлять.
            if not User.objects.exists():
                user.role = Role.HEAD_ADMIN
            else:
                user.role = Role.USER
            user.save()
            auth_login(request, user)
            if user.role == Role.HEAD_ADMIN:
                messages.success(
                    request,
                    "Регистрация выполнена. Вы первый пользователь системы — "
                    "вам присвоена роль «Главный администратор».",
                )
            else:
                messages.success(request, "Регистрация прошла успешно. Добро пожаловать!")
            return redirect("analyzer:dashboard")
    else:
        form = RegisterForm()

    return render(request, "accounts/register.html", {"form": form})


class VisionLoginView(LoginView):
    template_name = "accounts/login.html"
    authentication_form = LoginForm
    redirect_authenticated_user = True

    def form_valid(self, form):
        user = form.get_user()
        if user.role == Role.BLOCKED:
            auth_login(self.request, user)
            return redirect("accounts:blocked")
        response = super().form_valid(form)
        messages.success(self.request, f"С возвращением, {user.username}!")
        return response

    def get_success_url(self):
        return "/"


def logout_view(request):
    auth_logout(request)
    messages.info(request, "Вы вышли из системы.")
    return redirect("accounts:login")


@login_required(login_url="accounts:login")
def blocked_view(request):
    if request.user.role != Role.BLOCKED:
        return redirect("analyzer:dashboard")
    return render(request, "accounts/blocked.html")


@login_required(login_url="accounts:login")
def profile_view(request):
    if request.user.role == Role.BLOCKED:
        return redirect("accounts:blocked")

    if request.method == "POST":
        form = ProfileForm(request.POST, instance=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, "Профиль обновлён.")
            return redirect("accounts:profile")
    else:
        form = ProfileForm(instance=request.user)

    return render(request, "accounts/profile.html", {"form": form})
