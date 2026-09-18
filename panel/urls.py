from django.urls import path

from . import views

app_name = "panel"

urlpatterns = [
    path("", views.stats_view, name="stats"),
    path("users/", views.users_list_view, name="users_list"),
    path("users/<int:pk>/", views.user_detail_view, name="user_detail"),
    path("users/<int:pk>/set-role/", views.user_set_role_view, name="user_set_role"),
    path("users/<int:pk>/toggle-block/", views.user_toggle_block_view, name="user_toggle_block"),
    path("analyses/", views.analyses_list_view, name="analyses_list"),
]
