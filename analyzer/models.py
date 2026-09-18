from django.conf import settings
from django.db import models


class RiskLevel(models.TextChoices):
    LOW = "low", "Низкий"
    MEDIUM = "medium", "Средний"
    HIGH = "high", "Высокий"
    UNKNOWN = "unknown", "Не определён"


class AnalysisResult(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="analyses",
        verbose_name="Пользователь",
    )
    image = models.ImageField(upload_to="uploads/%Y/%m/%d/", verbose_name="Изображение")
    original_name = models.CharField(max_length=255, blank=True, verbose_name="Имя файла")

    backend = models.CharField(max_length=32, blank=True, verbose_name="Бэкенд")
    risk_level = models.CharField(
        max_length=16, choices=RiskLevel.choices, default=RiskLevel.UNKNOWN, verbose_name="Уровень риска"
    )
    needs_human_review = models.BooleanField(default=False, verbose_name="Нужна проверка человеком")
    description = models.TextField(blank=True, verbose_name="Описание")
    raw_report = models.JSONField(default=dict, verbose_name="Полный отчёт (JSON)")

    error = models.TextField(blank=True, verbose_name="Ошибка")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Дата анализа")

    class Meta:
        verbose_name = "Результат анализа"
        verbose_name_plural = "Результаты анализа"
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.original_name or self.pk} — {self.get_risk_level_display()}"

    @property
    def is_error(self) -> bool:
        return bool(self.error)
