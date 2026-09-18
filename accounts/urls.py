from django.urls import path

from . import views

app_name = "accounts"

urlpatterns = [
    path("register/", views.register_view, name="register"),
    path("login/", views.VisionLoginView.as_view(), name="login"),
    path("logout/", views.logout_view, name="logout"),
    path("blocked/", views.blocked_view, name="blocked"),
    path("profile/", views.profile_view, name="profile"),
]
