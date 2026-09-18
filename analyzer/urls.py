from django.urls import path

from . import views

app_name = "analyzer"

urlpatterns = [
    path("", views.dashboard_view, name="dashboard"),
    path("history/", views.history_view, name="history"),
    path("result/<int:pk>/", views.result_detail_view, name="result_detail"),
    path("health/", views.health_view, name="health"),
]
