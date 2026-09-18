from django.contrib.auth.models import AbstractUser
from django.db import models


class Role(models.TextChoices):
    BLOCKED = "blocked", "Заблокирован"
    USER = "user", "Пользователь"
    ADMIN = "admin", "Администратор"
    HEAD_ADMIN = "head_admin", "Главный администратор"


# Иерархия ролей: чем больше число — тем больше прав.
ROLE_RANK = {
    Role.BLOCKED: 0,
    Role.USER: 1,
    Role.ADMIN: 2,
    Role.HEAD_ADMIN: 3,
}


class User(AbstractUser):
    role = models.CharField(
        max_length=20,
        choices=Role.choices,
        default=Role.USER,
        verbose_name="Роль",
    )
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Дата регистрации")

    @property
    def rank(self) -> int:
        return ROLE_RANK.get(self.role, 0)

    @property
    def is_blocked(self) -> bool:
        return self.role == Role.BLOCKED

    @property
    def is_panel_staff(self) -> bool:
        """Может ли пользователь заходить в админ-панель."""
        return self.role in (Role.ADMIN, Role.HEAD_ADMIN)

    @property
    def is_head_admin(self) -> bool:
        return self.role == Role.HEAD_ADMIN

    def role_display_ru(self) -> str:
        return Role(self.role).label

    def can_manage(self, target: "User") -> bool:
        """Может ли self управлять ролью/блокировкой пользователя target."""
        if self.pk == target.pk:
            return False
        if self.role == Role.HEAD_ADMIN:
            return True
        if self.role == Role.ADMIN:
            # Обычный админ может блокировать/разблокировать только
            # рядовых пользователей — не трогает других админов и глав. админов.
            return target.role in (Role.USER, Role.BLOCKED)
        return False

    def assignable_roles(self) -> list[str]:
        """Какие роли self может назначать другим (в рамках can_manage)."""
        if self.role == Role.HEAD_ADMIN:
            return [Role.BLOCKED, Role.USER, Role.ADMIN, Role.HEAD_ADMIN]
        if self.role == Role.ADMIN:
            return [Role.BLOCKED, Role.USER]
        return []

    def __str__(self) -> str:
        return f"{self.username} ({self.role_display_ru()})"
