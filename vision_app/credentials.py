"""API-ключи провайдеров: хранение в БД в зашифрованном виде.

Почему шифрование, а не хеш. Пароль хранят хешем, потому что системе достаточно проверить
совпадение. Ключ облачного провайдера нужно отправлять ему в каждом запросе, то есть вернуть в исходном
виде, а из хеша это невозможно. Поэтому ключ шифруется обратимо (Fernet = AES-128-CBC +
HMAC-SHA256), а мастер-ключ шифрования хранится ВНЕ БД — в переменной окружения
VISION_CREDENTIALS_KEY (обычно записывается в файл .env в корне проекта — см. config.py:
_load_dotenv; реальная переменная окружения приоритетнее файла). Дамп БД без неё ничего
не раскрывает.

Мастер-ключ отдельный, а не FLASK_SECRET_KEY: иначе смена секрета Flask (например, чтобы
разлогинить всех) молча «убила» бы все сохранённые ключи.

    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

Ротация: можно указать несколько ключей через запятую — «новый,старый». Шифруется первым,
расшифровывается любым; ключ в БД перешифровывается при следующем сохранении.

Хеш тоже используется, но по другому назначению: fingerprint() — HMAC-SHA256 от ключа для
ключа кэша списка моделей (и чтобы не расшифровывать ради сравнения). Сам по себе fingerprint
ключ не раскрывает.

Сервер анализа (inference) ключей не хранит: приложение присылает ключ в КАЖДОМ запросе
заголовком X-Api-Key-<Провайдер> (см. services.py: _auth_headers).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
from datetime import timezone

from cryptography.fernet import Fernet, InvalidToken, MultiFernet

from .extensions import db
from .models import ProviderCredential, utcnow

log = logging.getLogger("vision_app.credentials")

ENV_MASTER_KEY = "VISION_CREDENTIALS_KEY"
HEADER_PREFIX = "X-Api-Key-"  # то же, что providers/base.py: CREDENTIAL_HEADER_PREFIX на сервере анализа
MAX_KEY_LEN = 512

SETUP_HINT = (
    f"Не задана переменная {ENV_MASTER_KEY} — без неё ключи API нельзя хранить. "
    'Сгенерируйте значение командой: python -c "from cryptography.fernet import Fernet; '
    f'print(Fernet.generate_key().decode())" — впишите его в файл .env в корне проекта '
    f"({ENV_MASTER_KEY}=...) и перезапустите приложение."
)


class CredentialsError(Exception):
    """Ключ нельзя сохранить/прочитать (нет мастер-ключа, неверный формат и т. п.)."""


def header_name(provider: str) -> str:
    """Заголовок, в котором ключ уходит серверу анализа (X-Api-Key-Gemini)."""
    return HEADER_PREFIX + provider.capitalize()


# ---- шифрование -------------------------------------------------------------
def _master_keys() -> list[str]:
    return [k.strip() for k in os.environ.get(ENV_MASTER_KEY, "").split(",") if k.strip()]


def is_available() -> bool:
    """Задан ли корректный мастер-ключ (иначе сохранять ключи нельзя)."""
    try:
        _fernet()
        return True
    except CredentialsError:
        return False


def _fernet() -> MultiFernet:
    keys = _master_keys()
    if not keys:
        raise CredentialsError(SETUP_HINT)
    try:
        return MultiFernet([Fernet(k.encode("ascii")) for k in keys])
    except (ValueError, TypeError, UnicodeEncodeError) as exc:
        raise CredentialsError(
            f"{ENV_MASTER_KEY} имеет неверный формат (нужен ключ Fernet: 32 байта в urlsafe-base64)."
        ) from exc


def encrypt(plain: str) -> str:
    return _fernet().encrypt(plain.encode("utf-8")).decode("ascii")


def decrypt(token: str) -> str | None:
    """None — не удалось расшифровать (сменили мастер-ключ или запись повреждена)."""
    try:
        return _fernet().decrypt(token.encode("ascii")).decode("utf-8")
    except (InvalidToken, UnicodeError):
        return None
    except CredentialsError:
        return None


def fingerprint(plain: str) -> str:
    """HMAC-SHA256 ключа (по первому мастер-ключу), 32 hex-символа. Нужен как ключ кэша."""
    keys = _master_keys()
    secret = (keys[0] if keys else "no-master-key").encode("utf-8")
    return hmac.new(secret, plain.encode("utf-8"), hashlib.sha256).hexdigest()[:32]


# ---- хранилище (БД) ----------------------------------------------------------
def validate_key(value: str) -> str:
    """Проверка введённого ключа: непустой, без пробелов/переводов строк (частая ошибка при
    копировании), в разумной длине. Возвращает очищенное значение или бросает CredentialsError."""
    value = (value or "").strip()
    if not value:
        raise CredentialsError("Ключ пустой.")
    if len(value) > MAX_KEY_LEN:
        raise CredentialsError("Слишком длинное значение ключа.")
    if any(ch.isspace() for ch in value) or not value.isascii():
        raise CredentialsError("В ключе не должно быть пробелов, переводов строк и не-ASCII символов.")
    return value


def set_key(provider: str, key: str, user_id: int | None = None) -> None:
    """Шифрует и сохраняет ключ провайдера (перезаписывает прежний)."""
    key = validate_key(key)
    token = encrypt(key)  # бросит CredentialsError без мастер-ключа — до любых записей в БД
    row = db.session.get(ProviderCredential, provider)
    if row is None:
        db.session.add(ProviderCredential(provider=provider, ciphertext=token, updated_by=user_id))
    else:
        row.ciphertext, row.updated_at, row.updated_by = token, utcnow(), user_id
    db.session.commit()


def clear_key(provider: str) -> bool:
    row = db.session.get(ProviderCredential, provider)
    if row is None:
        return False
    db.session.delete(row)
    db.session.commit()
    return True


def get_key(provider: str) -> str:
    """Расшифрованный ключ или '' (нет записи / не расшифровывается)."""
    row = db.session.get(ProviderCredential, provider)
    if row is None:
        return ""
    plain = decrypt(row.ciphertext)
    if plain is None:
        log.warning(
            "Ключ провайдера %r не расшифровывается — вероятно, изменился %s. Введите ключ заново.",
            provider, ENV_MASTER_KEY,
        )
        return ""
    return plain


def get_all_keys() -> dict[str, str]:
    """{провайдер: расшифрованный ключ} — всё, что удалось расшифровать."""
    result = {}
    for row in db.session.scalars(db.select(ProviderCredential)).all():
        plain = decrypt(row.ciphertext)
        if plain:
            result[row.provider] = plain
        else:
            log.warning("Ключ провайдера %r не расшифровывается (изменился %s?).", row.provider, ENV_MASTER_KEY)
    return result


def status(provider: str) -> dict:
    """Для UI: {'stored': bool, 'ok': bool (расшифровывается), 'hint': '…abcd', 'updated_at': datetime|None}.
    Показываем только последние 4 символа — чтобы отличать ключи, не раскрывая их."""
    row = db.session.get(ProviderCredential, provider)
    if row is None:
        return {"stored": False, "ok": False, "hint": "", "updated_at": None}
    plain = decrypt(row.ciphertext)
    updated = row.updated_at
    if updated is not None and updated.tzinfo is None:
        updated = updated.replace(tzinfo=timezone.utc)
    return {
        "stored": True,
        "ok": bool(plain),
        "hint": f"…{plain[-4:]}" if plain and len(plain) >= 12 else "",
        "updated_at": updated,
    }
