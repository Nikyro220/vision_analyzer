"""Формы (Flask-WTF). Все сообщения — на русском."""

from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from urllib.parse import urlsplit

from flask import current_app, request
from flask_wtf import FlaskForm
from flask_wtf.file import FileRequired, MultipleFileField
from PIL import Image
from sqlalchemy import func, select
from wtforms import BooleanField, HiddenField, IntegerField, PasswordField, SelectField, StringField, TextAreaField
from wtforms.fields import EmailField
from wtforms.widgets import ColorInput
from wtforms.validators import (
    DataRequired,
    Email,
    EqualTo,
    Length,
    NumberRange,
    Optional,
    Regexp,
    ValidationError,
)

from . import examples_codec
from .config import conf
from .extensions import db
from .models import CATEGORY_NAME_RE, COLOR_RE, DISPLAY_STYLE_CHOICES, Category, User

# Небольшой встроенный список самых частых паролей (аналог CommonPasswordValidator).
COMMON_PASSWORDS = {
    "password", "password1", "password123", "12345678", "123456789", "1234567890",
    "qwerty123", "qwertyuiop", "qwerty12", "11111111", "1q2w3e4r", "1q2w3e4r5t",
    "iloveyou", "admin123", "administrator", "letmein1", "welcome1", "abc12345",
    "zaq12wsx", "passw0rd", "p@ssw0rd", "football", "monkey123", "dragon12",
    "00000000", "123123123", "987654321", "qazwsxedc", "asdfghjkl", "qwertyui",
    "йцукенгшщзх", "пароль123", "привет123",
}


def validate_password_strength(password: str, username: str = "") -> None:
    """Аналог AUTH_PASSWORD_VALIDATORS из Django-версии. Бросает ValidationError."""
    min_length = conf("PASSWORD_MIN_LENGTH")
    if len(password) < min_length:
        raise ValidationError(f"Пароль слишком короткий. Минимум {min_length} символов.")
    if password.isdigit():
        raise ValidationError("Пароль не может состоять только из цифр.")
    if password.lower() in COMMON_PASSWORDS:
        raise ValidationError("Этот пароль слишком широко распространён.")
    if username:
        u, p = username.lower(), password.lower()
        similar = SequenceMatcher(a=p, b=u).quick_ratio() >= conf("PASSWORD_USERNAME_SIMILARITY")
        if len(u) >= conf("PASSWORD_USERNAME_MIN_LEN") and (u in p or similar):
            raise ValidationError("Пароль слишком похож на логин.")


class RegisterForm(FlaskForm):
    username = StringField(
        "Имя пользователя",
        validators=[
            DataRequired("Введите логин."),
            Length(max=150, message="Не более 150 символов."),
            Regexp(r"^[\w.@+-]+$", message="Допустимы только буквы, цифры и символы @/./+/-/_"),
        ],
        render_kw={"placeholder": "Придумайте логин", "autocomplete": "username", "autofocus": True},
    )
    email = EmailField(
        "Email",
        validators=[Optional(), Email("Введите корректный адрес электронной почты."), Length(max=254)],
        render_kw={"placeholder": "you@example.com", "autocomplete": "email"},
    )
    password1 = PasswordField(
        "Пароль",
        validators=[DataRequired("Введите пароль.")],
        render_kw={"placeholder": "Придумайте пароль", "autocomplete": "new-password"},
    )
    password2 = PasswordField(
        "Подтверждение пароля",
        validators=[
            DataRequired("Повторите пароль."),
            EqualTo("password1", message="Введённые пароли не совпадают."),
        ],
        render_kw={"placeholder": "Повторите пароль", "autocomplete": "new-password"},
    )

    def validate_username(self, field):
        exists = db.session.scalar(
            select(func.count(User.id)).where(func.lower(User.username) == field.data.strip().lower())
        )
        if exists:
            raise ValidationError("Пользователь с таким логином уже существует.")

    def validate_password1(self, field):
        validate_password_strength(field.data, self.username.data or "")


class LoginForm(FlaskForm):
    username = StringField(
        "Имя пользователя",
        validators=[DataRequired("Введите логин.")],
        render_kw={"placeholder": "Логин", "autocomplete": "username", "autofocus": True},
    )
    password = PasswordField(
        "Пароль",
        validators=[DataRequired("Введите пароль.")],
        render_kw={"placeholder": "Пароль", "autocomplete": "current-password"},
    )


def _squash_spaces(value):
    return " ".join((value or "").split())


class AccountForm(FlaskForm):
    """Редактирование данных аккаунта: свой профиль или (для админа) карточка пользователя."""

    username = StringField(
        "Логин",
        validators=[
            DataRequired("Введите логин."),
            Length(max=150, message="Не более 150 символов."),
            Regexp(r"^[\w.@+-]+$", message="Допустимы только буквы, цифры и символы @/./+/-/_"),
        ],
        render_kw={"placeholder": "Логин", "autocomplete": "username"},
    )
    email = EmailField(
        "Email",
        validators=[Optional(), Email("Введите корректный адрес электронной почты."), Length(max=254)],
        render_kw={"placeholder": "you@example.com", "autocomplete": "email"},
    )
    family_name = StringField(
        "Фамилия",
        validators=[Length(max=150, message="Не более 150 символов.")],
        filters=[_squash_spaces],
        render_kw={"placeholder": "Необязательно", "autocomplete": "family-name"},
    )
    given_name = StringField(
        "Имя",
        validators=[Length(max=150, message="Не более 150 символов.")],
        filters=[_squash_spaces],
        render_kw={"placeholder": "Необязательно", "autocomplete": "given-name"},
    )
    middle_name = StringField(
        "Отчество",
        validators=[Length(max=150, message="Не более 150 символов.")],
        filters=[_squash_spaces],
        render_kw={"placeholder": "Необязательно", "autocomplete": "additional-name"},
    )
    nickname = StringField(
        "Название аккаунта",
        validators=[Length(max=150, message="Не более 150 символов.")],
        filters=[_squash_spaces],  # лишние пробелы и переводы строк -> один пробел
        render_kw={"placeholder": "Необязательно: как называется ваш аккаунт", "autocomplete": "nickname"},
    )
    display_style = SelectField(
        "Как показывать моё имя",
        choices=DISPLAY_STYLE_CHOICES,
        default=DISPLAY_STYLE_CHOICES[0][0],
    )
    color = StringField(
        "Цвет пользователя",
        validators=[Optional(), Regexp(COLOR_RE, message="Цвет задаётся как #rrggbb.")],  # пусто — цвет не меняем
        widget=ColorInput(),
        render_kw={"title": "Красит аватар без картинки и ваши сообщения в чате"},
    )

    def __init__(self, *args, current_id: int | None = None, **kwargs):
        """current_id — id редактируемого пользователя, чтобы не спотыкаться о его же логин."""
        self._current_id = current_id
        super().__init__(*args, **kwargs)

    def validate_username(self, field):
        stmt = select(func.count(User.id)).where(func.lower(User.username) == field.data.strip().lower())
        if self._current_id is not None:
            stmt = stmt.where(User.id != self._current_id)
        if db.session.scalar(stmt):
            raise ValidationError("Пользователь с таким логином уже существует.")


# Обратная совместимость на случай, если что-то ещё импортирует старое имя.
ProfileForm = AccountForm


class DeleteAccountForm(FlaskForm):
    """Подтверждение удаления аккаунта — тупо по приколу просим вручную ввести SQL-запрос с логином."""

    confirm_sql = StringField(
        "Подтверждение",
        validators=[DataRequired("Введите команду подтверждения.")],
        render_kw={"placeholder": "DELETE FROM users WHERE username = '...';", "autocomplete": "off"},
    )


# ----------------------------------------------------------------------------
# Категории оценивания (/panel/categories/)
# ----------------------------------------------------------------------------

# Единственный плейсхолдер, который сервер анализа подставляет ВНУТРИ текста
# правил Full/Compact (см. inference/prompt.py: get_system_prompt — финальный
# .replace("__OUTPUT_LANGUAGE__", ...) применяется уже после того, как правила
# категории вставлены в промпт). Пишется ровно так — регистр и число подчёркиваний
# важны: опечатка не сломает сохранение, но и не заменится языком, а попадёт в
# промпт как мусорный текст. См. _placeholder_typo_check ниже.
OUTPUT_LANGUAGE_PLACEHOLDER = "__OUTPUT_LANGUAGE__"

_PLACEHOLDER_LOOKALIKE_RE = re.compile(r"_{0,3}\s*output[\s_]+language\s*_{0,3}", re.IGNORECASE)


def _placeholder_typo_check(field):
    """Ловит похожие-но-не-точные написания __OUTPUT_LANGUAGE__ (не то число
    подчёркиваний, пробел вместо "_", другой регистр) — они не упадут с ошибкой
    сами по себе, но и не сработают на сервере анализа, просто протекут в
    промпт как есть."""
    text = field.data or ""
    for match in _PLACEHOLDER_LOOKALIKE_RE.finditer(text):
        if match.group(0) != OUTPUT_LANGUAGE_PLACEHOLDER:
            raise ValidationError(
                f"Похоже на опечатку в служебном слове «{match.group(0).strip()}». "
                f"Нужно написать ровно «{OUTPUT_LANGUAGE_PLACEHOLDER}» (два подчёркивания "
                "с каждой стороны, заглавными) — иначе сервер анализа не подставит язык, "
                "и это слово останется в промпте как есть."
            )


class CategoryForm(FlaskForm):
    """Создание/редактирование одной категории оценивания.

    full/compact/summary — те же поля, что раньше жили в
    inference/categories/<имя>.json (см. models.Category)."""

    name = StringField(
        "Техническое имя",
        validators=[
            DataRequired("Введите техническое имя категории."),
            Length(max=64, message="Не более 64 символов."),
            Regexp(CATEGORY_NAME_RE, message="Допустимы только латинские буквы, цифры и «_»."),
        ],
        render_kw={"placeholder": "weapons_and_dangerous_objects", "autocomplete": "off"},
    )
    title = StringField(
        "Название для интерфейса",
        validators=[Length(max=150, message="Не более 150 символов.")],
        render_kw={"placeholder": "Оружие и опасные предметы"},
    )
    summary = TextAreaField(
        "Summary (для первого, классифицирующего прохода)",
        validators=[DataRequired("Заполните summary.")],
        render_kw={
            "rows": 3,
            "placeholder": "Короткое описание на английском: что должно быть видно на "
            "изображении, чтобы категория стала кандидатом.",
        },
    )
    full = TextAreaField(
        "Full (полные правила для второго прохода)",
        validators=[DataRequired("Заполните full.")],
        render_kw={"rows": 10, "placeholder": '<signal_category name="...">...</signal_category>'},
    )
    compact = TextAreaField(
        "Compact (сокращённая версия, fallback)",
        validators=[DataRequired("Заполните compact.")],
        render_kw={"rows": 6, "placeholder": '<signal_category name="...">...</signal_category>'},
    )
    full_extra = TextAreaField(
        "Доп. блоки к Full (редко нужно)", validators=[Optional()], render_kw={"rows": 4}
    )
    compact_extra = TextAreaField(
        "Доп. блоки к Compact (редко нужно)", validators=[Optional()], render_kw={"rows": 4}
    )
    # Примеры сцен редактируются структурно (JS в category_form.html) и приходят
    # сюда JSON-списком сцен; в текст для БД их превращает examples_codec.
    examples_en_data = HiddenField(validators=[Optional()])
    examples_ru_data = HiddenField(validators=[Optional()])
    position = IntegerField(
        "Порядок появления в промпте",
        validators=[Optional(), NumberRange(min=0, max=100000, message="От 0 до 100000.")],
        render_kw={"placeholder": "0"},
    )
    is_active = BooleanField("Включена (участвует в анализе)", default=True)

    def __init__(self, *args, current_id: int | None = None, **kwargs):
        """current_id — id редактируемой категории, чтобы не спотыкаться о её же имя."""
        self._current_id = current_id
        super().__init__(*args, **kwargs)

    def validate_name(self, field):
        stmt = select(func.count(Category.id)).where(Category.name == field.data.strip())
        if self._current_id is not None:
            stmt = stmt.where(Category.id != self._current_id)
        if db.session.scalar(stmt):
            raise ValidationError("Категория с таким именем уже существует.")

    def _validate_examples(self, field):
        try:
            examples_codec.data_to_text(field.data, "x")
        except (ValueError, TypeError):
            raise ValidationError("Не удалось разобрать примеры сцен — обновите страницу и повторите.")

    def validate_examples_en_data(self, field):
        self._validate_examples(field)

    def validate_examples_ru_data(self, field):
        self._validate_examples(field)

    def validate_full(self, field):
        _placeholder_typo_check(field)

    def validate_compact(self, field):
        _placeholder_typo_check(field)

    def validate_full_extra(self, field):
        _placeholder_typo_check(field)

    def validate_compact_extra(self, field):
        _placeholder_typo_check(field)


class DeleteCategoryForm(FlaskForm):
    """Пустая форма-обёртка ради CSRF-токена на кнопке «Удалить»."""


class ChatPromptForm(FlaskForm):
    """Системный промпт чата (раздел «Промпты»). Пустое значение = вернуть стандартный текст."""

    prompt = TextAreaField("Системный промпт чата", validators=[Optional()])

    def validate_prompt(self, field) -> None:
        limit = conf("CHAT_PROMPT_MAX_CHARS")
        if len((field.data or "").strip()) > limit:
            raise ValidationError(f"Слишком длинный промпт: не более {limit} символов.")


# Форматы Pillow -> (расширение файла, MIME-тип)
IMAGE_FORMATS = {
    "JPEG": (".jpg", "image/jpeg"),
    "PNG": (".png", "image/png"),
    "GIF": (".gif", "image/gif"),
    "WEBP": (".webp", "image/webp"),
    "BMP": (".bmp", "image/bmp"),
    "TIFF": (".tiff", "image/tiff"),
}


@dataclass
class AcceptedImage:
    filename: str
    data: bytes
    ext: str
    mime: str
    caption: str = ""


LINK_MAX_LEN = 2048
_BARE_LINK_RE = re.compile(r"^[\w-]+(\.[\w-]+)+(/\S*)?$", re.UNICODE)  # t.me/chan/1, instagram.com/p/x


def parse_links(raw: str) -> tuple[list[str], list[tuple[str, str]]]:
    """Текст из поля «Ссылки» -> (годные ссылки без повторов, [(ссылка, причина отказа)]).

    Ссылки разделяются пробелами/переводами строк. Адрес без схемы («t.me/chan/1») дополняется
    https://. Принимаются только http/https. Дальнейшую проверку (публичный ли адрес, есть ли
    там картинка) делает сервер анализа при обработке — см. inference/link_fetcher.py.
    """
    good: list[str] = []
    bad: list[tuple[str, str]] = []
    seen: set[str] = set()
    for token in (raw or "").split():
        link = token.strip().strip("<>\"'")
        if not link:
            continue
        if "://" not in link and _BARE_LINK_RE.match(link):
            link = "https://" + link
        if len(link) > LINK_MAX_LEN:
            bad.append((link[:60] + "…", "слишком длинная ссылка"))
            continue
        try:
            parts = urlsplit(link)
            host = parts.hostname
        except ValueError:
            bad.append((link, "некорректный адрес"))
            continue
        if parts.scheme not in ("http", "https") or not host:
            bad.append((link[:80], "нужна ссылка вида https://…"))
            continue
        if link in seen:
            continue
        seen.add(link)
        good.append(link)
    return good, bad


class ImageUploadForm(FlaskForm):
    """Файлы и/или ссылки на посты за раз. Годные файлы и ссылки уходят в очередь,
    негодные пропускаются с пояснением (form.rejected / form.rejected_links).

    Ссылку на пост скачивает и разбирает сервер анализа (yt-dlp / Open Graph): из поста берутся
    картинки и контекст (автор, текст), который попадает в подпись к снимку.
    """

    image = MultipleFileField("Изображения")
    links = TextAreaField("Ссылки на посты", render_kw={"rows": 3})

    # Заполняются в validate_image / validate_links
    accepted: list[AcceptedImage]
    rejected: list[tuple[str, str]]
    link_urls: list[str]
    rejected_links: list[tuple[str, str]]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accepted, self.rejected = [], []
        self.link_urls, self.rejected_links = [], []

    def validate_image(self, field):
        """Проверяем содержимое через Pillow, а не расширение файла.

        Подписи (caption) идут отдельным полем формы "captions", по одной на файл,
        в ТОМ ЖЕ порядке, что и файлы (это обеспечивает static/js/queue.js — рисует
        поле подписи сразу под каждым выбранным файлом). Если подписей меньше, чем
        файлов (JS не сработал, форма отправлена без него), недостающие — пустые.

        Пустой выбор — не ошибка: можно прислать только ссылки (см. validate()).
        """
        self.accepted, self.rejected = [], []

        files = [f for f in (field.data or []) if getattr(f, "filename", "")]
        from .settings_store import get_runtime_setting

        raw_captions = request.form.getlist("captions")
        caption_max = conf("CAPTION_MAX_CHARS")
        max_files = get_runtime_setting("QUEUE_MAX_FILES_PER_UPLOAD")
        if len(files) > max_files:
            raise ValidationError(f"За один раз можно загрузить не больше {max_files} файлов.")

        for idx, upload in enumerate(files):
            name = upload.filename
            data = upload.read()
            upload.stream.seek(0)
            caption = raw_captions[idx].strip()[:caption_max] if idx < len(raw_captions) else ""

            if not data:
                self.rejected.append((name, "файл пуст"))
                continue
            try:
                with Image.open(io.BytesIO(data)) as img:
                    fmt = img.format
                    img.verify()
            except Exception:  # noqa: BLE001
                self.rejected.append((name, "не является изображением или повреждён"))
                continue
            if fmt not in IMAGE_FORMATS:
                self.rejected.append((name, "формат не поддерживается"))
                continue

            ext, mime = IMAGE_FORMATS[fmt]
            self.accepted.append(AcceptedImage(name, data, ext, mime, caption))

    def validate_links(self, field):
        self.link_urls, self.rejected_links = parse_links(field.data or "")

    def validate(self, extra_validators=None):
        if not super().validate(extra_validators):
            return False

        from .settings_store import get_runtime_setting

        total = len(self.accepted) + len(self.link_urls)
        if not total:
            reasons = "; ".join(f"«{n}»: {why}" for n, why in self.rejected[: conf("REJECTED_FILES_SHOWN")])
            if self.rejected_links:
                reasons = "; ".join(
                    [reasons] * bool(reasons)
                    + [f"«{n}»: {why}" for n, why in self.rejected_links[: conf("REJECTED_FILES_SHOWN")]]
                )
            message = (
                "Загрузите правильное изображение или ссылку. " + reasons
                if reasons
                else "Выберите файл изображения или вставьте ссылку на пост."
            )
            self.image.errors = [*self.image.errors, message]
            return False

        max_files = get_runtime_setting("QUEUE_MAX_FILES_PER_UPLOAD")
        if total > max_files:
            self.links.errors = [*self.links.errors, f"За один раз можно добавить не больше {max_files} файлов и ссылок суммарно."]
            return False
        return True


# ----------------------------------------------------------------------------
# Параметры генерации (POST /sampling)
# ----------------------------------------------------------------------------
def _parse_number(raw: str, kind: str, lo, hi, lo_exclusive: bool = False):
    """Разбирает число из поля. Бросает ValueError с русским сообщением."""
    if kind == "int":
        if not re.fullmatch(r"[+-]?\d+", raw):
            raise ValueError("Введите целое число.")
        value = int(raw)
    else:
        try:
            value = float(raw)
        except ValueError:
            raise ValueError("Введите число (например, 0.7).") from None
        if not math.isfinite(value):
            raise ValueError("Введите обычное число.")

    too_low = value <= lo if lo_exclusive else value < lo
    if (lo is not None and too_low) or (hi is not None and value > hi):
        left = f"больше {lo}" if lo_exclusive else f"от {lo}"
        raise ValueError(f"Допустимое значение: {left} до {hi}.")
    return value


# Описание каждого параметра генерации: поле формы и ограничения. Какие из них показывать
# для конкретного провайдера, решает сам провайдер (sampling_keys в GET /providers), поэтому
# для нового бэкенда править этот файл не нужно — только если появится новый ПАРАМЕТР.
#   имя -> (тип, минимум, максимум, минимум не включается)
_SAMPLING_SPEC = {
    "temperature": ("float", 0, 2, False),
    "top_p": ("float", 0, 1, True),
    "top_k": ("int", -1, 100000, False),
    "seed": ("int", -(2**63), 2**63 - 1, False),
    "num_predict": ("int", 1, 1048576, False),
    "num_ctx": ("int", 128, 1048576, False),
}


def _text_field(name: str, placeholder: str, inputmode: str) -> StringField:
    return StringField(
        name,
        validators=[Optional(), Length(max=32)],
        render_kw={"placeholder": placeholder, "inputmode": inputmode, "autocomplete": "off"},
    )


def _think_field() -> SelectField:
    return SelectField(
        "think",
        choices=[
            ("false", "выкл"),
            ("true", "вкл"),
            ("low", "low (GPT-OSS)"),
            ("medium", "medium (GPT-OSS)"),
            ("high", "high (GPT-OSS)"),
        ],
        validators=[Optional()],
    )


# Фабрики полей; порядок здесь = порядок полей на странице.
_SAMPLING_FIELDS = {
    "temperature": lambda: _text_field("temperature", "0–2", "decimal"),
    "top_p": lambda: _text_field("top_p", "0–1", "decimal"),
    "top_k": lambda: _text_field("top_k", "целое", "numeric"),
    "seed": lambda: _text_field("seed", "целое", "numeric"),
    "num_predict": lambda: _text_field("num_predict", "1–1048576", "numeric"),
    "think": _think_field,
    "num_ctx": lambda: _text_field("num_ctx", "128–1048576", "numeric"),
}


class _SamplingForm(FlaskForm):
    """Базовая форма (без полей — их добавляет sampling_form_class по sampling_keys провайдера).
    Пустое поле = «не менять»: на сервер уходят только заполненные."""

    class Meta:
        # Формы с prefix называют поле токена «<prefix>-csrf_token», а глобальный
        # CSRFProtect ищет «csrf_token». Защита всё равно включена глобально:
        # шаблон кладёт обычное скрытое поле csrf_token.
        csrf = False

    SPEC: dict = {}
    values: dict

    def validate(self, extra_validators=None):
        ok = super().validate(extra_validators)
        self.values = {}
        for name, (kind, lo, hi, lo_excl) in self.SPEC.items():
            field = getattr(self, name)
            raw = (field.data or "").strip().replace(",", ".")
            if not raw:
                continue
            try:
                self.values[name] = _parse_number(raw, kind, lo, hi, lo_excl)
            except ValueError as exc:
                field.errors = [*field.errors, str(exc)]
                ok = False

        if "think" in self._fields:
            self.values["think"] = {"true": True, "false": False}.get(self.think.data, self.think.data)

        return ok


@lru_cache(maxsize=None)
def sampling_form_class(keys: tuple[str, ...]) -> type[_SamplingForm]:
    """Форма параметров генерации ровно с теми полями, которые провайдер использует
    (sampling_keys из GET /providers). Неизвестные ключи игнорируются; классы кэшируются —
    для одного набора ключей он создаётся один раз."""
    ordered = tuple(k for k in _SAMPLING_FIELDS if k in keys)
    attrs: dict = {k: _SAMPLING_FIELDS[k]() for k in ordered}
    attrs["SPEC"] = {k: _SAMPLING_SPEC[k] for k in ordered if k in _SAMPLING_SPEC}
    return type("SamplingForm", (_SamplingForm,), attrs)