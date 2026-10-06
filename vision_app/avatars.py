"""Аватары пользователей: проверка присланной картинки, перекодирование и хранение на диске.

Статичные аватары редактируются в браузере (static/js/avatar-editor.js), а сервер получает уже
готовую картинку. Ей всё равно не доверяем: открываем через Pillow, проверяем формат и размеры и
ПЕРЕКОДИРУЕМ в квадратный WebP — метаданные и любой посторонний «хвост» в файле пропадают, на диск
попадает только то, что мы сами сохранили.

Анимированные аватары (GIF, анимированный WebP, APNG) браузер на canvas отрисовать целиком не может
(получился бы один кадр), поэтому он присылает исходный файл и параметры редактора (сдвиг, масштаб,
поворот, цвет...), а сервер применяет их к КАЖДОМУ кадру и собирает анимированный WebP.
Результат в обоих случаях лежит в одном месте: avatars/<id>.webp.
"""

from __future__ import annotations

import io
import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path

from flask import current_app
from PIL import Image, ImageOps, ImageSequence, UnidentifiedImageError

from .config import Config

log = logging.getLogger("vision_app.avatars")

ALLOWED_FORMATS = {"PNG", "JPEG", "WEBP"}
ANIMATED_FORMATS = ALLOWED_FORMATS | {"GIF"}
MAX_RENDERED_SIDE = 4096  # px; потолок промежуточного кадра при сильном увеличении
MAX_RENDERED_PIXELS = 12_000_000  # и по площади: иначе 4096 x 4096 x 4 байта на каждый кадр


class AvatarError(ValueError):
    """Ошибка для пользователя: текст можно показывать как есть."""


def avatar_file(user_id: int) -> Path:
    return Path(current_app.config["UPLOAD_FOLDER"]) / Config.AVATARS_DIR / f"{int(user_id)}.webp"


def avatar_link(user_id: int, version: str) -> str:
    """Адрес, по которому отдаётся аватар (blueprint accounts, маршрут avatar). Это значение
    хранится в users.avatar_url; ?v= меняется при каждой загрузке и сбрасывает кэш браузера."""
    return f"/accounts/avatar/{int(user_id)}/?v={version}"


# ----------------------------------------------------------------------------
# Параметры редактора (те же, что в avatar-editor.js)
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class EditParams:
    zoom: float = 1.0
    sx: float = 1.0
    sy: float = 1.0
    rot: float = 0.0
    hue: float = 0.0
    sat: float = 1.0
    bri: float = 1.0
    con: float = 1.0
    dx: float = 0.0
    dy: float = 0.0
    flip: int = 1


# допустимые диапазоны — как у ползунков редактора; снаружи принимаем только их
_RANGES = {
    "zoom": (0.2, 5), "sx": (0.2, 3), "sy": (0.2, 3), "rot": (-180, 180), "hue": (-180, 180),
    "sat": (0, 2), "bri": (0, 2), "con": (0, 2), "dx": (-256, 256), "dy": (-256, 256),
}


def parse_params(raw: dict) -> EditParams:
    """Проверяет присланные из браузера параметры (числа в допустимых диапазонах)."""
    if not isinstance(raw, dict):
        raise AvatarError("Некорректные параметры редактора.")
    values: dict = {}
    for key, (lo, hi) in _RANGES.items():
        if key not in raw:
            continue
        try:
            number = float(raw[key])
        except (TypeError, ValueError):
            raise AvatarError("Некорректные параметры редактора.") from None
        if not math.isfinite(number):
            raise AvatarError("Некорректные параметры редактора.")
        values[key] = min(hi, max(lo, number))
    values["flip"] = -1 if raw.get("flip") in (-1, "-1") else 1
    return EditParams(**values)


# ----------------------------------------------------------------------------
# Статичный аватар: браузер уже нарисовал итоговый квадрат
# ----------------------------------------------------------------------------
def _check_size(data: bytes, limit: int) -> None:
    if not data:
        raise AvatarError("Файл пустой.")
    if len(data) > limit:
        raise AvatarError(f"Файл слишком большой (максимум {limit // 1024} КБ).")


def process_avatar(data: bytes) -> bytes:
    """Картинка (PNG/JPEG/WebP) -> квадратный WebP AVATAR_SIZE x AVATAR_SIZE."""
    _check_size(data, current_app.config.get("AVATAR_MAX_UPLOAD_BYTES", Config.AVATAR_MAX_UPLOAD_BYTES))

    max_side = current_app.config.get("AVATAR_MAX_SIDE", Config.AVATAR_MAX_SIDE)
    size = current_app.config.get("AVATAR_SIZE", Config.AVATAR_SIZE)
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.format not in ALLOWED_FORMATS:
                raise AvatarError("Поддерживаются только PNG, JPEG и WebP.")
            if max(img.size) > max_side:  # размер известен до распаковки пикселей
                raise AvatarError(f"Изображение слишком большое (максимум {max_side}x{max_side}).")
            img.load()
            img = ImageOps.exif_transpose(img).convert("RGBA")
    except AvatarError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError):
        raise AvatarError("Не удалось прочитать изображение.") from None

    # Редактор уже отдаёт квадрат; если прислали другое — вырезаем центр, без искажений.
    img = ImageOps.fit(img, (size, size), Image.Resampling.LANCZOS)
    out = io.BytesIO()
    img.save(out, "WEBP", quality=90, method=4)
    return out.getvalue()


# ----------------------------------------------------------------------------
# Анимированный аватар: параметры редактора применяются к каждому кадру на сервере
# ----------------------------------------------------------------------------
def _color_matrix(p: EditParams) -> tuple[float, ...] | None:
    """Оттенок -> насыщенность -> яркость -> контраст одной матрицей 3x4 для Image.convert("RGB", m).
    Те же формулы, что в avatar-editor.js (матрицы из спецификации CSS-фильтров)."""
    if p.hue == 0 and p.sat == 1 and p.bri == 1 and p.con == 1:
        return None
    a = math.radians(p.hue)
    c, s = math.cos(a), math.sin(a)
    h = (
        (0.213 + c * 0.787 - s * 0.213, 0.715 - c * 0.715 - s * 0.715, 0.072 - c * 0.072 + s * 0.928),
        (0.213 - c * 0.213 + s * 0.143, 0.715 + c * 0.285 + s * 0.140, 0.072 - c * 0.072 - s * 0.283),
        (0.213 - c * 0.213 - s * 0.787, 0.715 - c * 0.715 + s * 0.715, 0.072 + c * 0.928 + s * 0.072),
    )
    sat = p.sat
    sm = (
        (0.213 + 0.787 * sat, 0.715 - 0.715 * sat, 0.072 - 0.072 * sat),
        (0.213 - 0.213 * sat, 0.715 + 0.285 * sat, 0.072 - 0.072 * sat),
        (0.213 - 0.213 * sat, 0.715 - 0.715 * sat, 0.072 + 0.928 * sat),
    )
    gain = p.bri * p.con
    offset = 128 * (1 - p.con)  # ((x * bri) - 128) * con + 128 = x * bri * con + 128 * (1 - con)
    out: list[float] = []
    for r in range(3):
        row = [sum(sm[r][k] * h[k][q] for k in range(3)) * gain for q in range(3)]
        out.extend([*row, offset])
    return tuple(out)


def render_frame(frame: Image.Image, p: EditParams, size: int, matrix: tuple[float, ...] | None) -> Image.Image:
    """Один кадр -> size x size RGBA. Порядок преобразований тот же, что на canvas в редакторе:
    масштаб (в осях картинки, с отражением) -> поворот -> сдвиг, затем цвет."""
    w, h = frame.size
    k = size / min(w, h) * p.zoom  # «cover»: при масштабе 100% кадр заполнен
    tw, th = max(1, round(w * k * p.sx)), max(1, round(h * k * p.sy))
    if max(tw, th) > MAX_RENDERED_SIDE or tw * th > MAX_RENDERED_PIXELS:
        raise AvatarError("Слишком сильное увеличение для такой картинки. Уменьшите масштаб или растяжение.")

    scaled = frame.resize((tw, th), Image.Resampling.LANCZOS)  # RGBA ресайзится с учётом прозрачности
    if p.flip < 0:
        scaled = scaled.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    # forward: out = R(rot) * (in - c) + t; Pillow ждёт обратное отображение out -> in.
    cos, sin = math.cos(math.radians(p.rot)), math.sin(math.radians(p.rot))
    cx, cy = tw / 2, th / 2
    tx, ty = size / 2 + p.dx, size / 2 + p.dy
    coeffs = (
        cos, sin, -cos * tx - sin * ty + cx,
        -sin, cos, sin * tx - cos * ty + cy,
    )
    out = scaled.transform((size, size), Image.Transform.AFFINE, coeffs, resample=Image.Resampling.BICUBIC)

    if matrix is not None:
        alpha = out.getchannel("A")
        out = out.convert("RGB").convert("RGB", matrix)
        out.putalpha(alpha)
    return out


def process_animated(data: bytes, params: EditParams) -> tuple[bytes, bool]:
    """Исходный файл + параметры редактора -> (WebP size x size, анимирован ли он).

    Работает и с неподвижными картинками (один кадр -> обычный WebP), так что клиент может
    слать на этот путь всё, что не смог отрисовать сам.
    """
    cfg = current_app.config
    _check_size(data, cfg.get("AVATAR_MAX_ANIMATED_BYTES", Config.AVATAR_MAX_ANIMATED_BYTES))
    size = cfg.get("AVATAR_SIZE", Config.AVATAR_SIZE)
    max_side = cfg.get("AVATAR_MAX_SIDE", Config.AVATAR_MAX_SIDE)
    max_frames = cfg.get("AVATAR_MAX_FRAMES", Config.AVATAR_MAX_FRAMES)
    max_pixels = cfg.get("AVATAR_MAX_ANIMATED_PIXELS", Config.AVATAR_MAX_ANIMATED_PIXELS)
    max_out = cfg.get("AVATAR_MAX_OUTPUT_BYTES", Config.AVATAR_MAX_OUTPUT_BYTES)
    matrix = _color_matrix(params)

    frames: list[Image.Image] = []
    durations: list[int] = []
    try:
        with Image.open(io.BytesIO(data)) as img:
            if img.format not in ANIMATED_FORMATS:
                raise AvatarError("Поддерживаются PNG, JPEG, WebP и GIF.")
            if max(img.size) > max_side:
                raise AvatarError(f"Изображение слишком большое (максимум {max_side}x{max_side}).")
            count = getattr(img, "n_frames", 1)
            if count > max_frames:
                raise AvatarError(f"Слишком много кадров ({count}, максимум {max_frames}).")
            if img.size[0] * img.size[1] * count > max_pixels:
                raise AvatarError("Анимация слишком тяжёлая: уменьшите размер или число кадров.")
            animated = count > 1
            for frame in ImageSequence.Iterator(img):
                duration = int(frame.info.get("duration") or 0)
                rgba = frame.convert("RGBA")
                if not animated:
                    rgba = ImageOps.exif_transpose(rgba)
                frames.append(render_frame(rgba, params, size, matrix))
                durations.append(duration if duration >= 20 else 100)  # 0 / 10 мс браузеры всё равно растягивают до 100
    except AvatarError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, ValueError, EOFError):
        raise AvatarError("Не удалось прочитать изображение.") from None

    if not frames:
        raise AvatarError("В файле нет кадров.")

    if len(frames) == 1:
        out = io.BytesIO()
        frames[0].save(out, "WEBP", quality=90, method=4)
        return out.getvalue(), False

    for quality in (80, 60, 40):  # анимация тяжёлая: если не влезаем в лимит, снижаем качество
        out = io.BytesIO()
        frames[0].save(
            out, "WEBP", save_all=True, append_images=frames[1:], duration=durations,
            loop=0, quality=quality, method=3,
        )
        if out.tell() <= max_out:
            return out.getvalue(), True
    raise AvatarError(f"Анимация получилась слишком тяжёлой (больше {max_out // 1024} КБ). Попробуйте файл покороче.")


def save_avatar(user_id: int, data: bytes) -> None:
    """Атомарно записывает уже обработанный аватар."""
    target = avatar_file(user_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, target)


def remove_avatar(user_id: int) -> None:
    try:
        avatar_file(user_id).unlink(missing_ok=True)
    except OSError as exc:
        log.warning("Не удалось удалить аватар пользователя %s: %s", user_id, exc)
