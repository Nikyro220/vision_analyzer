"""Модели: User (с ролями) и AnalysisResult."""

from __future__ import annotations

from datetime import datetime, timezone

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from .extensions import db


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ----------------------------------------------------------------------------
# Роли
# ----------------------------------------------------------------------------
class Role:
    BLOCKED = "blocked"
    USER = "user"
    ADMIN = "admin"
    HEAD_ADMIN = "head_admin"


ROLE_LABELS = {
    Role.BLOCKED: "Заблокирован",
    Role.USER: "Пользователь",
    Role.ADMIN: "Администратор",
    Role.HEAD_ADMIN: "Главный администратор",
}
ROLE_CHOICES = list(ROLE_LABELS.items())

# Иерархия ролей: чем больше число — тем больше прав.
ROLE_RANK = {
    Role.BLOCKED: 0,
    Role.USER: 1,
    Role.ADMIN: 2,
    Role.HEAD_ADMIN: 3,
}


class RiskLevel:
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNKNOWN = "unknown"


class Status:
    """Жизненный цикл анализа: очередь -> обработка -> готово (после этого запись попадает в историю)."""

    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"


STATUS_LABELS = {
    Status.QUEUED: "В очереди",
    Status.PROCESSING: "Обрабатывается",
    Status.DONE: "Готово",
}


RISK_LABELS = {
    RiskLevel.LOW: "Низкий",
    RiskLevel.MEDIUM: "Средний",
    RiskLevel.HIGH: "Высокий",
    RiskLevel.UNKNOWN: "Не определён",
}


# ----------------------------------------------------------------------------
# Пользователь
# ----------------------------------------------------------------------------
class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False, index=True)
    email = db.Column(db.String(254), nullable=False, default="")
    first_name = db.Column(db.String(150), nullable=False, default="")
    last_name = db.Column(db.String(150), nullable=False, default="")
    password_hash = db.Column(db.String(256), nullable=False)
    active = db.Column(db.Boolean, nullable=False, default=True)
    role = db.Column(db.String(20), nullable=False, default=Role.USER)
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)

    # --- пароль ---
    def set_password(self, raw_password: str) -> None:
        self.password_hash = generate_password_hash(raw_password)

    def check_password(self, raw_password: str) -> bool:
        return check_password_hash(self.password_hash, raw_password)

    # --- Flask-Login ---
    @property
    def is_active(self) -> bool:  # type: ignore[override]
        return bool(self.active)

    # --- роли ---
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

    @property
    def role_display_ru(self) -> str:
        return ROLE_LABELS.get(self.role, self.role)

    def can_manage(self, target: "User") -> bool:
        """Может ли self управлять ролью/блокировкой пользователя target."""
        if self.id == target.id:
            return False
        if self.role == Role.HEAD_ADMIN:
            return True
        if self.role == Role.ADMIN:
            # Обычный админ работает только с рядовыми пользователями —
            # не трогает других админов и главных админов.
            return target.role in (Role.USER, Role.BLOCKED)
        return False

    def assignable_roles(self) -> list[str]:
        """Какие роли self может назначать другим (в рамках can_manage).

        «Заблокирован» сюда не входит — для блокировки есть отдельная
        кнопка (toggle-block), а не выпадающий список ролей.
        """
        if self.role == Role.HEAD_ADMIN:
            return [Role.USER, Role.ADMIN, Role.HEAD_ADMIN]
        if self.role == Role.ADMIN:
            return [Role.USER]
        return []

    def __repr__(self) -> str:
        return f"<User {self.username} ({self.role})>"


# ----------------------------------------------------------------------------
# Результат анализа
# ----------------------------------------------------------------------------
class AnalysisResult(db.Model):
    __tablename__ = "analysis_results"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user = db.relationship(
        "User", backref=db.backref("analyses", cascade="all, delete-orphan", passive_deletes=True)
    )

    # Путь относительно UPLOAD_FOLDER, например uploads/2026/09/20/<uuid>.png
    image_path = db.Column(db.String(500), nullable=False)
    original_name = db.Column(db.String(255), nullable=False, default="")

    backend = db.Column(db.String(32), nullable=False, default="")
    risk_level = db.Column(db.String(16), nullable=False, default=RiskLevel.UNKNOWN, index=True)
    needs_human_review = db.Column(db.Boolean, nullable=False, default=False, index=True)
    description = db.Column(db.Text, nullable=False, default="")
    raw_report = db.Column(db.JSON, nullable=False, default=dict)

    error = db.Column(db.Text, nullable=False, default="")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)

    # --- очередь ---
    # server_default нужен для уже существующих записей (их анализ давно выполнен): при
    # автоматическом добавлении колонок они получают status='done'.
    status = db.Column(
        db.String(16), nullable=False, default=Status.QUEUED, server_default=Status.DONE
    )
    image_mime = db.Column(db.String(64), nullable=False, default="", server_default="")
    started_at = db.Column(db.DateTime, nullable=True)
    finished_at = db.Column(db.DateTime, nullable=True)
    caption = db.Column(db.Text, nullable=False, default="", server_default="")
    is_new = db.Column(db.Boolean, nullable=False, default=False, server_default="0")

    @property
    def is_error(self) -> bool:
        return bool(self.error)

    @property
    def is_pending(self) -> bool:
        """В очереди или обрабатывается — в историю такая запись ещё не попадает."""
        return self.status != Status.DONE

    @property
    def status_display(self) -> str:
        return STATUS_LABELS.get(self.status, self.status)

    @property
    def risk_level_display(self) -> str:
        return RISK_LABELS.get(self.risk_level, self.risk_level)

    def __repr__(self) -> str:
        return f"<AnalysisResult {self.id} {self.risk_level}>"


# ----------------------------------------------------------------------------
# Настройки приложения (ключ-значение)
# ----------------------------------------------------------------------------
class Setting(db.Model):
    """Простое хранилище настроек приложения, общее для всех пользователей.

    Сейчас используется для выбора бэкенда и модели, на которых выполняются анализы.
    """

    __tablename__ = "settings"

    key = db.Column(db.String(64), primary_key=True)
    value = db.Column(db.Text, nullable=False, default="")
