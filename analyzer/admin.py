from django.contrib import admin

from .models import AnalysisResult


@admin.register(AnalysisResult)
class AnalysisResultAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "original_name", "risk_level", "needs_human_review", "created_at")
    list_filter = ("risk_level", "needs_human_review", "backend")
    search_fields = ("original_name", "user__username")

