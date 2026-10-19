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
from flask_babel import gettext, lazy_gettext
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
        raise ValidationError(gettext("forms.password.too_short", min=min_length))
    if password.isdigit():
        raise ValidationError(gettext("forms.password.digits_only"))
    if password.lower() in COMMON_PASSWORDS:
        raise ValidationError(gettext("forms.password.common"))
    if username:
        u, p = username.lower(), password.lower()
        similar = SequenceMatcher(a=p, b=u).quick_ratio() >= conf("PASSWORD_USERNAME_SIMILARITY")
        if len(u) >= conf("PASSWORD_USERNAME_MIN_LEN") and (u in p or similar):
            raise ValidationError(gettext("forms.password.similar"))


class RegisterForm(FlaskForm):
    username = StringField(
        lazy_gettext("forms.register.username"),
        validators=[
            DataRequired(lazy_gettext("forms.common.enter_login")),
            Length(max=150, message=lazy_gettext("forms.common.max150")),
            Regexp(r"^[\w.@+-]+$", message=lazy_gettext("forms.common.login_chars")),
        ],
        render_kw={"placeholder": lazy_gettext("forms.register.username_placeholder"), "autocomplete": "username", "autofocus": True},
    )
    email = EmailField(
        "Email",
        validators=[Optional(), Email(lazy_gettext("forms.common.email_invalid")), Length(max=254)],
        render_kw={"placeholder": "you@example.com", "autocomplete": "email"},
    )
    password1 = PasswordField(
        lazy_gettext("forms.common.password"),
        validators=[DataRequired(lazy_gettext("forms.common.enter_password"))],
        render_kw={"placeholder": lazy_gettext("forms.register.password_placeholder"), "autocomplete": "new-password"},
    )
    password2 = PasswordField(
        lazy_gettext("forms.register.password2"),
        validators=[
            DataRequired(lazy_gettext("forms.register.repeat_required")),
            EqualTo("password1", message=lazy_gettext("forms.register.mismatch")),
        ],
        render_kw={"placeholder": lazy_gettext("forms.register.password2_placeholder"), "autocomplete": "new-password"},
    )

    def validate_username(self, field):
        exists = db.session.scalar(
            select(func.count(User.id)).where(func.lower(User.username) == field.data.strip().lower())
        )
        if exists:
            raise ValidationError(lazy_gettext("forms.common.login_exists"))

    def validate_password1(self, field):
        validate_password_strength(field.data, self.username.data or "")


class LoginForm(FlaskForm):
    username = StringField(
        lazy_gettext("forms.register.username"),
        validators=[DataRequired(lazy_gettext("forms.common.enter_login"))],
        render_kw={"placeholder": lazy_gettext("account.field.login"), "autocomplete": "username", "autofocus": True},
    )
    password = PasswordField(
        lazy_gettext("forms.common.password"),
        validators=[DataRequired(lazy_gettext("forms.common.enter_password"))],
        render_kw={"placeholder": lazy_gettext("forms.common.password"), "autocomplete": "current-password"},
    )


def _squash_spaces(value):
    return " ".join((value or "").split())


class AccountForm(FlaskForm):
    """Редактирование данных аккаунта: свой профиль или (для админа) карточка пользователя."""

    username = StringField(
        lazy_gettext("account.field.login"),
        validators=[
            DataRequired(lazy_gettext("forms.common.enter_login")),
            Length(max=150, message=lazy_gettext("forms.common.max150")),
            Regexp(r"^[\w.@+-]+$", message=lazy_gettext("forms.common.login_chars")),
        ],
        render_kw={"placeholder": lazy_gettext("account.field.login"), "autocomplete": "username"},
    )
    email = EmailField(
        "Email",
        validators=[Optional(), Email(lazy_gettext("forms.common.email_invalid")), Length(max=254)],
        render_kw={"placeholder": "you@example.com", "autocomplete": "email"},
    )
    family_name = StringField(
        lazy_gettext("forms.account.family_name"),
        validators=[Length(max=150, message=lazy_gettext("forms.common.max150"))],
        filters=[_squash_spaces],
        render_kw={"placeholder": lazy_gettext("forms.account.optional"), "autocomplete": "family-name"},
    )
    given_name = StringField(
        lazy_gettext("forms.account.given_name"),
        validators=[Length(max=150, message=lazy_gettext("forms.common.max150"))],
        filters=[_squash_spaces],
        render_kw={"placeholder": lazy_gettext("forms.account.optional"), "autocomplete": "given-name"},
    )
    middle_name = StringField(
        lazy_gettext("forms.account.middle_name"),
        validators=[Length(max=150, message=lazy_gettext("forms.common.max150"))],
        filters=[_squash_spaces],
        render_kw={"placeholder": lazy_gettext("forms.account.optional"), "autocomplete": "additional-name"},
    )
    nickname = StringField(
        lazy_gettext("account.field.nickname_title"),
        validators=[Length(max=150, message=lazy_gettext("forms.common.max150"))],
        filters=[_squash_spaces],  # лишние пробелы и переводы строк -> один пробел
        render_kw={"placeholder": lazy_gettext("forms.account.nickname_placeholder"), "autocomplete": "nickname"},
    )
    display_style = SelectField(
        lazy_gettext("forms.account.display_style"),
        choices=DISPLAY_STYLE_CHOICES,
        default=DISPLAY_STYLE_CHOICES[0][0],
    )
    color = StringField(
        lazy_gettext("forms.account.color"),
        validators=[Optional(), Regexp(COLOR_RE, message=lazy_gettext("forms.account.color_format"))],  # пусто — цвет не меняем
        widget=ColorInput(),
        render_kw={"title": lazy_gettext("forms.account.color_hint")},
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
            raise ValidationError(lazy_gettext("forms.common.login_exists"))


# Обратная совместимость на случай, если что-то ещё импортирует старое имя.
ProfileForm = AccountForm


class DeleteAccountForm(FlaskForm):
    """Подтверждение удаления аккаунта — тупо по приколу просим вручную ввести SQL-запрос с логином."""

    confirm_sql = StringField(
        lazy_gettext("forms.delete.confirm_label"),
        validators=[DataRequired(lazy_gettext("forms.delete.confirm_required"))],
        render_kw={"placeholder": "DELETE FROM users WHERE username = '...';", "autocomplete": "off"},
    )


# ----------------------------------------------------------------------------
# Категории оценивания (/panel/categories/)
# ----------------------------------------------------------------------------

# Единственный плейсхолдер, который сервер анализа подставляет ВНУТРИ текста
# правил Full/Compact (см. inference/tools/vision_analyzer/prompt.py: get_system_prompt — финальный
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
            raise ValidationError(gettext(
                "forms.category.placeholder_typo",
                found=match.group(0).strip(),
                expected=OUTPUT_LANGUAGE_PLACEHOLDER,
            ))


class CategoryForm(FlaskForm):
    """Создание/редактирование одной категории оценивания.

    full/compact/summary — те же поля, что раньше жили в
    inference/tools/vision_analyzer/categories/<имя>.json (см. models.Category)."""

    name = StringField(
        lazy_gettext("forms.category.name"),
        validators=[
            DataRequired(lazy_gettext("forms.category.name_required")),
            Length(max=64, message=lazy_gettext("forms.category.name_max")),
            Regexp(CATEGORY_NAME_RE, message=lazy_gettext("forms.category.name_chars")),
        ],
        render_kw={"placeholder": "weapons_and_dangerous_objects", "autocomplete": "off"},
    )
    title = StringField(
        lazy_gettext("forms.category.title"),
        validators=[Length(max=150, message=lazy_gettext("forms.common.max150"))],
        render_kw={"placeholder": lazy_gettext("forms.category.title_placeholder")},
    )
    summary = TextAreaField(
        lazy_gettext("forms.category.summary"),
        validators=[DataRequired(lazy_gettext("forms.category.summary_required"))],
        render_kw={
            "rows": 3,
            "placeholder": lazy_gettext("forms.category.summary_placeholder"),
        },
    )
    full = TextAreaField(
        lazy_gettext("forms.category.full"),
        validators=[DataRequired(lazy_gettext("forms.category.full_required"))],
        render_kw={"rows": 10, "placeholder": '<signal_category name="...">...</signal_category>'},
    )
    compact = TextAreaField(
        lazy_gettext("forms.category.compact"),
        validators=[DataRequired(lazy_gettext("forms.category.compact_required"))],
        render_kw={"rows": 6, "placeholder": '<signal_category name="...">...</signal_category>'},
    )
    full_extra = TextAreaField(
        lazy_gettext("forms.category.full_extra"), validators=[Optional()], render_kw={"rows": 4}
    )
    compact_extra = TextAreaField(
        lazy_gettext("forms.category.compact_extra"), validators=[Optional()], render_kw={"rows": 4}
    )
    # Примеры сцен редактируются структурно (JS в category_form.html) и приходят
    # сюда JSON-списком сцен; в текст для БД их превращает examples_codec.
    examples_en_data = HiddenField(validators=[Optional()])
    examples_ru_data = HiddenField(validators=[Optional()])
    position = IntegerField(
        lazy_gettext("forms.category.position"),
        validators=[Optional(), NumberRange(min=0, max=100000, message=lazy_gettext("forms.category.position_range"))],
        render_kw={"placeholder": "0"},
    )
    is_active = BooleanField(lazy_gettext("forms.category.is_active"), default=True)

    def __init__(self, *args, current_id: int | None = None, **kwargs):
        """current_id — id редактируемой категории, чтобы не спотыкаться о её же имя."""
        self._current_id = current_id
        super().__init__(*args, **kwargs)

    def validate_name(self, field):
        stmt = select(func.count(Category.id)).where(Category.name == field.data.strip())
        if self._current_id is not None:
            stmt = stmt.where(Category.id != self._current_id)
        if db.session.scalar(stmt):
            raise ValidationError(lazy_gettext("forms.category.exists"))

    def _validate_examples(self, field):
        try:
            examples_codec.data_to_text(field.data, "x")
        except (ValueError, TypeError):
            raise ValidationError(lazy_gettext("forms.category.examples_bad"))

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

    prompt = TextAreaField(lazy_gettext("prompts.chat.title"), validators=[Optional()])

    def validate_prompt(self, field) -> None:
        limit = conf("CHAT_PROMPT_MAX_CHARS")
        if len((field.data or "").strip()) > limit:
            raise ValidationError(gettext("forms.prompt.too_long", limit=limit))


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
    там картинка) делает сервер анализа при обработке — см. inference/tools/vision_analyzer/link_fetcher.py.
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
            bad.append((link[:60] + "…", gettext("forms.link.too_long")))
            continue
        try:
            parts = urlsplit(link)
            host = parts.hostname
        except ValueError:
            bad.append((link, gettext("forms.link.invalid")))
            continue
        if parts.scheme not in ("http", "https") or not host:
            bad.append((link[:80], gettext("forms.link.need_url")))
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

    image = MultipleFileField(lazy_gettext("forms.upload.images"))
    links = TextAreaField(lazy_gettext("forms.upload.links"), render_kw={"rows": 3})

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
            raise ValidationError(gettext("forms.upload.too_many", max=max_files))

        for idx, upload in enumerate(files):
            name = upload.filename
            data = upload.read()
            upload.stream.seek(0)
            caption = raw_captions[idx].strip()[:caption_max] if idx < len(raw_captions) else ""

            if not data:
                self.rejected.append((name, gettext("forms.upload.empty")))
                continue
            try:
                with Image.open(io.BytesIO(data)) as img:
                    fmt = img.format
                    img.verify()
            except Exception:  # noqa: BLE001
                self.rejected.append((name, gettext("forms.upload.not_image")))
                continue
            if fmt not in IMAGE_FORMATS:
                self.rejected.append((name, gettext("forms.upload.unsupported")))
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
            reasons = "; ".join(gettext("forms.upload.reason", name=n, why=why) for n, why in self.rejected[: conf("REJECTED_FILES_SHOWN")])
            if self.rejected_links:
                reasons = "; ".join(
                    [reasons] * bool(reasons)
                    + [gettext("forms.upload.reason", name=n, why=why) for n, why in self.rejected_links[: conf("REJECTED_FILES_SHOWN")]]
                )
            message = (
                gettext("forms.upload.invalid", reasons=reasons)
                if reasons
                else gettext("forms.upload.nothing")
            )
            self.image.errors = [*self.image.errors, message]
            return False

        max_files = get_runtime_setting("QUEUE_MAX_FILES_PER_UPLOAD")
        if total > max_files:
            self.links.errors = [*self.links.errors, gettext("forms.upload.too_many_total", max=max_files)]
            return False
        return True


# ----------------------------------------------------------------------------
# Параметры провайдера (POST /providers/<имя>/settings)
# ----------------------------------------------------------------------------
# У каждого провайдера свой набор параметров, свои значения по умолчанию и свои допустимые
# значения. Панель ничего о них не знает: сервер описывает каждый параметр (тип, диапазон,
# варианты) в GET /providers → settings, и форма строится ровно по этому описанию. Новый
# провайдер или новый параметр не требуют правок в этом файле.

_RESET_WORDS = ("auto", "none", "null", "default")


def _parse_number(raw: str, kind: str, lo, hi, lo_exclusive: bool = False):
    """Разбирает число из поля. Бросает ValueError с русским сообщением."""
    if kind == "int":
        if not re.fullmatch(r"[+-]?\d+", raw):
            raise ValueError(gettext("forms.number.int"))
        value = int(raw)
    else:
        try:
            value = float(raw)
        except ValueError:
            raise ValueError(gettext("forms.number.float")) from None
        if not math.isfinite(value):
            raise ValueError(gettext("forms.number.finite"))

    too_low = lo is not None and (value <= lo if lo_exclusive else value < lo)
    if too_low or (hi is not None and value > hi):
        raise ValueError(gettext("forms.number.allowed", range=_range_text(lo, hi, lo_exclusive)))
    return value


def _fmt(x) -> str:
    return str(int(x)) if float(x).is_integer() else str(x)


def _range_text(lo, hi, lo_exclusive: bool = False) -> str:
    if lo is not None and hi is not None:
        return (gettext("forms.range.gt_to", lo=_fmt(lo), hi=_fmt(hi)) if lo_exclusive
                else gettext("forms.range.from_to", lo=_fmt(lo), hi=_fmt(hi)))
    if lo is not None:
        return (gettext("forms.range.gt", lo=_fmt(lo)) if lo_exclusive else gettext("forms.range.ge", lo=_fmt(lo)))
    if hi is not None:
        return gettext("forms.range.le", hi=_fmt(hi))
    return gettext("forms.range.any")


def _display(value) -> str:
    """Текущее значение параметра → строка для поля формы (None — «не задано» — пустое поле)."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def settings_initial(settings) -> dict:
    """Начальные значения полей формы из текущих значений провайдера на сервере."""
    return {st.key: _display(st.value) for st in settings}


def _setting_field(spec: tuple):
    """Поле формы по описанию параметра (SettingInfo.spec)."""
    key, kind, label, hint, default, lo, hi, lo_excl, choices, nullable = spec
    description = " ".join(filter(None, [
        hint,
        gettext("forms.setting.default", value=_display(default) or "auto"),
    ]))
    if kind == "choice":
        return SelectField(
            label or key, choices=list(choices), validators=[Optional()], description=description,
        )
    if lo is not None or hi is not None:
        placeholder = f"{_fmt(lo)}–{_fmt(hi)}" if lo is not None and hi is not None else _range_text(lo, hi, lo_excl)
    else:
        placeholder = gettext("forms.number.placeholder_int") if kind == "int" else gettext("forms.number.placeholder_number")
    if nullable:
        placeholder += gettext("forms.number.or_auto")
    return StringField(
        label or key,
        validators=[Optional(), Length(max=32)],
        description=description,
        render_kw={
            "placeholder": placeholder,
            "inputmode": "numeric" if kind == "int" else "decimal",
            "autocomplete": "off",
        },
    )


class _SettingsForm(FlaskForm):
    """Базовая форма (поля добавляет provider_settings_form_class по описанию параметров
    провайдера). Пустое числовое поле = «не менять»: на сервер уходят только заполненные."""

    class Meta:
        # Формы с prefix называют поле токена «<prefix>-csrf_token», а глобальный
        # CSRFProtect ищет «csrf_token». Защита всё равно включена глобально:
        # шаблон кладёт обычное скрытое поле csrf_token.
        csrf = False

    SPEC: dict = {}      # ключ -> SettingInfo.spec
    values: dict

    def validate(self, extra_validators=None):
        ok = super().validate(extra_validators)
        self.values = {}
        for key, (_, kind, _, _, _, lo, hi, lo_excl, _, nullable) in self.SPEC.items():
            field = getattr(self, key)
            raw = (field.data or "").strip().replace(",", ".")
            if not raw:
                continue
            if kind == "choice":
                self.values[key] = {"true": True, "false": False}.get(raw, raw)
            elif nullable and raw.lower() in _RESET_WORDS:
                self.values[key] = None
            else:
                try:
                    self.values[key] = _parse_number(raw, kind, lo, hi, lo_excl)
                except ValueError as exc:
                    field.errors = [*field.errors, str(exc)]
                    ok = False
        return ok


@lru_cache(maxsize=None)
def _settings_form_class(specs: tuple) -> type[_SettingsForm]:
    attrs: dict = {spec[0]: _setting_field(spec) for spec in specs}
    attrs["SPEC"] = {spec[0]: spec for spec in specs}
    return type("ProviderSettingsForm", (_SettingsForm,), attrs)


def provider_settings_form_class(settings) -> type[_SettingsForm]:
    """Форма параметров провайдера ровно с теми полями, которые он описал (GET /providers →
    settings). Классы кэшируются по описанию — для одного набора параметров он создаётся один раз."""
    return _settings_form_class(tuple(st.spec for st in settings))
