"""Модели: User (с ролями) и AnalysisResult."""

from __future__ import annotations

import re
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
    # SHA-256 содержимого файла (image_dedup.py): по нему повторные загрузки одного и того же
    # файла склеиваются в один снимок. NULL — ещё не посчитан, "" — файл на диске потерян.
    image_hash = db.Column(db.String(64), nullable=True, index=True)

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


class AnalysisEmbedding(db.Model):
    """Вектор описания анализа для поиска «по смыслу» (см. vector_search.py).

    Отдельная таблица, а не колонка в analysis_results: BLOB на несколько КБ не
    должен тянуться в каждый select(AnalysisResult) (история, панель), а вектор
    можно пересчитать (смена модели, правка описания), не трогая сам анализ.
    Один вектор на анализ; `model` — имя модели, которой он посчитан: векторы
    разных моделей несовместимы, поиск использует только векторы «текущей» модели.

    FK с ondelete=CASCADE в SQLite сам по себе НЕ срабатывает (PRAGMA foreign_keys
    в приложении не включён), поэтому удаление вручную дублируется в history.py,
    а поиск идёт через JOIN с analysis_results — «сирота» в результат не попадёт.
    """

    __tablename__ = "analysis_embeddings"

    analysis_id = db.Column(
        db.Integer, db.ForeignKey("analysis_results.id", ondelete="CASCADE"), primary_key=True
    )
    model = db.Column(db.String(200), nullable=False, index=True)
    dim = db.Column(db.Integer, nullable=False)
    # float32 little-endian, L2-нормализован: cosine-сходство == скалярное произведение.
    vector = db.Column(db.LargeBinary, nullable=False)
    text_hash = db.Column(db.String(40), nullable=False, default="")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)

    def __repr__(self) -> str:
        return f"<AnalysisEmbedding {self.analysis_id} {self.model} dim={self.dim}>"


# ----------------------------------------------------------------------------
# Чат с моделью — сессии и сообщения
# ----------------------------------------------------------------------------
class ChatRole:
    USER = "user"
    ASSISTANT = "assistant"


class ChatSession(db.Model):
    """Одна ветка переписки пользователя с моделью (аналог «чата» в ChatGPT).

    Заголовок изначально пуст — заполняется первым сообщением пользователя
    (см. blueprints/chat.py), поэтому display_title ниже подставляет
    заглушку, пока сообщений ещё не было.
    """

    __tablename__ = "chat_sessions"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    user = db.relationship(
        "User", backref=db.backref("chat_sessions", cascade="all, delete-orphan", passive_deletes=True)
    )

    title = db.Column(db.String(200), nullable=False, default="")
    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    # Обновляется при каждом новом сообщении — по этому полю сортируется список чатов.
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow, index=True)

    messages = db.relationship(
        "ChatMessage",
        backref="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.id",
    )

    @property
    def display_title(self) -> str:
        return self.title or "Новый чат"

    def __repr__(self) -> str:
        return f"<ChatSession {self.id} {self.title!r}>"


class ChatMessage(db.Model):
    __tablename__ = "chat_messages"

    id = db.Column(db.Integer, primary_key=True)
    session_id = db.Column(
        db.Integer, db.ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )

    role = db.Column(db.String(16), nullable=False)  # ChatRole.USER / ChatRole.ASSISTANT
    content = db.Column(db.Text, nullable=False, default="")

    # Ссылки на анализы, которые инструмент search_analyses (chat_tools/analyses.py) вернул под этот
    # ответ ассистента — список словарей {id, url, thumb_url, label, risk_level,
    # risk_label, date}. Формируются из тех же строк БД, что и текстовая сводка
    # в system-промпте, поэтому не могут "поплыть"/сгаллюцинироваться в отличие
    # от того, если бы модель сама писала ссылки текстом. Пусто для role=user
    # и для ответов без сработавшего ретрива.
    refs = db.Column(db.JSON, nullable=False, default=list, server_default="[]")

    # Только для role=assistant — чем/на чём был получен этот ответ (для отображения).
    backend = db.Column(db.String(32), nullable=False, default="")
    model = db.Column(db.String(120), nullable=False, default="")

    created_at = db.Column(db.DateTime, nullable=False, default=utcnow, index=True)

    def __repr__(self) -> str:
        return f"<ChatMessage {self.id} {self.role}>"


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


# ----------------------------------------------------------------------------
# Категории оценивания (сигналов), которые распознаёт нейросеть на изображении
# ----------------------------------------------------------------------------

# Имя категории передаётся серверу анализа как есть и там же используется как
# имя файла на диске (см. inference/categories.py) — поэтому те же ограничения:
# только латинские буквы, цифры и «_».
CATEGORY_NAME_RE = re.compile(r"^[A-Za-z0-9_]+$")

# Раньше full/compact хранились ЦЕЛИКОМ как <signal_category name="...">...</signal_category>
# (так их ждёт сервер анализа — см. inference/categories.py: full_signals_block/
# compact_signals_block просто склеивают эти строки). Начиная с админ-панели эта
# обёртка больше не часть того, что редактирует человек: хранится и правится только
# «тело» правил, а тег с именем категории достраивается автоматически в
# to_overlay_dict(). Изредка после закрывающего тега шёл ещё один вспомогательный
# блок (например <known_phrase_reference>) — такой «хвост» не относится к телу
# правил и сохраняется отдельно в full_extra/compact_extra, чтобы не потерять его
# при переносе в новый формат (см. _split_wrapper) и не завернуть по ошибке внутрь
# <signal_category>.
_OPEN_TAG_RE = re.compile(r'^\s*<signal_category\s+name="[^"]*"\s*>\s*\n?')
_CLOSE_TAG = "</signal_category>"


def _split_wrapper(text: str) -> tuple[str, str]:
    """Если text — это (или начинается с) готовый <signal_category>...</signal_category>,
    возвращает (тело, всё-что-после-закрывающего-тега). Иначе — (text как есть, "")."""
    text = (text or "").strip()
    match = _OPEN_TAG_RE.match(text)
    if not match:
        return text, ""
    rest = text[match.end():]
    idx = rest.find(_CLOSE_TAG)
    if idx == -1:
        return text, ""  # тег открыт, но не закрыт — не похоже на валидную обёртку, не трогаем
    body = rest[:idx].strip()
    extra = rest[idx + len(_CLOSE_TAG):].strip()
    return body, extra


def _split_paragraphs(text: str) -> list[str]:
    """Разбивает текст правил на абзацы (по пустой строке) — единица
    редактирования в UI («+ Добавить абзац/правило»). Всегда возвращает
    хотя бы один (возможно пустой) элемент, чтобы в форме было куда писать."""
    parts = [p.strip() for p in re.split(r"\n\s*\n", (text or "").strip()) if p.strip()]
    return parts or [""]


class Category(db.Model):
    """Одна категория оценивания (сигнала), которую сервер анализа ищет на
    изображении. Раньше жили файлами в inference/categories/*.json — теперь
    единственное место хранения и редактирования — эта таблица в vision_app;
    при каждом вызове /analyze текущий набор целиком уходит на сервер анализа
    как разовый оверлей (см. services.build_categories_payload и
    inference/categories.py: build_overlay).
    """

    __tablename__ = "categories"

    id = db.Column(db.Integer, primary_key=True)

    # Техническое имя (== имя категории на сервере анализа): только латиница/цифры/«_».
    name = db.Column(db.String(64), unique=True, nullable=False, index=True)
    # Человекочитаемое название для интерфейса — на имя не влияет.
    title = db.Column(db.String(150), nullable=False, default="")

    # summary — короткое описание для первого (классифицирующего) прохода.
    summary = db.Column(db.Text, nullable=False, default="")
    # full/compact — ТЕЛО правил категории для второго (полного) прохода, БЕЗ
    # обёртки <signal_category>...</signal_category> (см. _split_wrapper выше и
    # to_overlay_dict ниже). compact — сокращённая версия, используется как fallback.
    full = db.Column(db.Text, nullable=False, default="")
    compact = db.Column(db.Text, nullable=False, default="")
    # Редкий «хвост» после закрывающего </signal_category> (см. _split_wrapper) —
    # например вспомогательный <known_phrase_reference>. Почти всегда пусто;
    # в форме это отдельное, свёрнутое по умолчанию поле "доп. блоки".
    # server_default="" (а не только default="") нужен, чтобы schema.py смог
    # добавить эти колонки в уже существующую таблицу через ALTER TABLE — без
    # него ensure_schema не знает, каким значением заполнить старые строки, и
    # просто пропускает NOT NULL колонку с предупреждением (см. schema.py).
    full_extra = db.Column(db.Text, nullable=False, default="", server_default="")
    compact_extra = db.Column(db.Text, nullable=False, default="", server_default="")
    # Необязательные разобранные примеры сцены на каждом языке.
    example_en = db.Column(db.Text, nullable=False, default="")
    example_ru = db.Column(db.Text, nullable=False, default="")

    # Порядок появления в промпте (по возрастанию), затем — по id.
    position = db.Column(db.Integer, nullable=False, default=0, index=True)
    # Выключенная категория хранится, но не отправляется на сервер анализа.
    is_active = db.Column(db.Boolean, nullable=False, default=True)

    created_at = db.Column(db.DateTime, nullable=False, default=utcnow)
    updated_at = db.Column(db.DateTime, nullable=False, default=utcnow, onupdate=utcnow)

    def full_paragraphs(self) -> list[str]:
        """Тело full, разбитое на абзацы — то, что рисует форма как список
        отдельных правил с кнопками добавить/удалить."""
        return _split_paragraphs(self.full)

    def compact_paragraphs(self) -> list[str]:
        return _split_paragraphs(self.compact)

    def _wrap(self, body: str, extra: str = "") -> str:
        """Оборачивает тело правил в <signal_category name="...">...</signal_category>
        и, если есть, дописывает «хвост» (extra) после закрывающего тега. Если в
        body всё же оказался вставленный вручную готовый блок с тегом — сначала
        разворачивает его, чтобы не получить двойную обёртку."""
        body = (body or "").strip()
        extra = (extra or "").strip()
        inner_body, inner_extra = _split_wrapper(body)
        if inner_body != body:
            body = inner_body
            extra = f"{inner_extra}\n\n{extra}" if extra else inner_extra
        wrapped = f'<signal_category name="{self.name}">\n{body}\n</signal_category>'
        return f"{wrapped}\n\n{extra}" if extra else wrapped

    def to_overlay_dict(self) -> dict:
        """Формат одного элемента списка "categories" в теле /analyze (см.
        inference/categories.py: build_overlay / _merge_category_payload).
        full/compact оборачиваются в <signal_category name="...">...</signal_category>
        здесь — это единственное место, где тег вообще появляется."""
        item: dict = {
            "name": self.name,
            "summary": self.summary,
            "full": self._wrap(self.full, self.full_extra),
            "compact": self._wrap(self.compact, self.compact_extra),
        }
        examples = {}
        if self.example_en.strip():
            examples["en"] = self.example_en
        if self.example_ru.strip():
            examples["ru"] = self.example_ru
        if examples:
            item["examples"] = examples
        return item

    def __repr__(self) -> str:
        return f"<Category {self.name}>"