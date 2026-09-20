"""Формы (Flask-WTF). Все сообщения — на русском."""

from __future__ import annotations

import io
import math
import re
from difflib import SequenceMatcher

from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired
from PIL import Image
from sqlalchemy import func, select
from wtforms import PasswordField, StringField
from wtforms.fields import EmailField
from wtforms.validators import (
    DataRequired,
    Email,
    EqualTo,
    Length,
    Optional,
    Regexp,
    ValidationError,
)

from .extensions import db
from .models import User

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
    if len(password) < 8:
        raise ValidationError("Пароль слишком короткий. Минимум 8 символов.")
    if password.isdigit():
        raise ValidationError("Пароль не может состоять только из цифр.")
    if password.lower() in COMMON_PASSWORDS:
        raise ValidationError("Этот пароль слишком широко распространён.")
    if username:
        u, p = username.lower(), password.lower()
        if len(u) >= 3 and (u in p or SequenceMatcher(a=p, b=u).quick_ratio() >= 0.7):
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


class ProfileForm(FlaskForm):
    email = EmailField(
        "Email",
        validators=[Optional(), Email("Введите корректный адрес электронной почты."), Length(max=254)],
        render_kw={"placeholder": "you@example.com"},
    )
    first_name = StringField(
        "Имя", validators=[Length(max=150)], render_kw={"placeholder": "Имя"}
    )
    last_name = StringField(
        "Фамилия", validators=[Length(max=150)], render_kw={"placeholder": "Фамилия"}
    )


# Форматы Pillow -> (расширение файла, MIME-тип)
IMAGE_FORMATS = {
    "JPEG": (".jpg", "image/jpeg"),
    "PNG": (".png", "image/png"),
    "GIF": (".gif", "image/gif"),
    "WEBP": (".webp", "image/webp"),
    "BMP": (".bmp", "image/bmp"),
    "TIFF": (".tiff", "image/tiff"),
}


class ImageUploadForm(FlaskForm):
    image = FileField(
        "Изображение",
        validators=[FileRequired("Выберите файл изображения.")],
    )

    # Заполняются в validate_image
    image_bytes: bytes = b""
    image_ext: str = ".jpg"
    image_mime: str = "image/jpeg"

    def validate_image(self, field):
        """Проверяем, что загружено настоящее изображение (через Pillow),
        а не просто файл с картинкой в расширении."""
        upload = field.data
        data = upload.read()
        upload.stream.seek(0)

        if not data:
            raise ValidationError("Файл пуст.")

        try:
            with Image.open(io.BytesIO(data)) as img:
                fmt = img.format
                img.verify()
        except Exception as exc:  # noqa: BLE001
            raise ValidationError(
                "Загрузите правильное изображение. Файл, который вы загрузили, "
                "поврежден или не является изображением."
            ) from exc

        if fmt not in IMAGE_FORMATS:
            raise ValidationError("Этот формат изображения не поддерживается.")

        self.image_bytes = data
        self.image_ext, self.image_mime = IMAGE_FORMATS[fmt]


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


class _SamplingForm(FlaskForm):
    """Базовая форма. Пустое поле = «не менять»: на сервер уходят только заполненные."""

    class Meta:
        # Формы с prefix называют поле токена «<prefix>-csrf_token», а глобальный
        # CSRFProtect ищет «csrf_token». Защита всё равно включена глобально:
        # шаблон кладёт обычное скрытое поле csrf_token.
        csrf = False

    # имя поля -> (тип, минимум, максимум, минимум не включается)
    SPEC: dict = {}
    values: dict

    temperature = StringField(
        "temperature",
        validators=[Optional(), Length(max=32)],
        render_kw={"placeholder": "0–2", "inputmode": "decimal", "autocomplete": "off"},
    )
    top_p = StringField(
        "top_p",
        validators=[Optional(), Length(max=32)],
        render_kw={"placeholder": "0–1", "inputmode": "decimal", "autocomplete": "off"},
    )
    top_k = StringField(
        "top_k",
        validators=[Optional(), Length(max=32)],
        render_kw={"placeholder": "целое", "inputmode": "numeric", "autocomplete": "off"},
    )
    seed = StringField(
        "seed",
        validators=[Optional(), Length(max=32)],
        render_kw={"placeholder": "целое", "inputmode": "numeric", "autocomplete": "off"},
    )

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
        return ok


_COMMON_SPEC = {
    "temperature": ("float", 0, 2, False),
    "top_p": ("float", 0, 1, True),
    "top_k": ("int", -1, 100000, False),
    "seed": ("int", -(2**63), 2**63 - 1, False),
}


class VllmSamplingForm(_SamplingForm):
    """vLLM: num_ctx здесь нет — размер контекста в vLLM задаётся при запуске сервера."""

    SPEC = dict(_COMMON_SPEC)


class OllamaSamplingForm(_SamplingForm):
    SPEC = {**_COMMON_SPEC, "num_ctx": ("int", 128, 1048576, False)}

    num_ctx = StringField(
        "num_ctx",
        validators=[Optional(), Length(max=32)],
        render_kw={"placeholder": "128–1048576", "inputmode": "numeric", "autocomplete": "off"},
    )


SAMPLING_FORMS = {"vllm": VllmSamplingForm, "ollama": OllamaSamplingForm}
