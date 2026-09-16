"""
image_upscaler.py — опциональный апскейлинг маленьких изображений в SCALE раз
перед отправкой в vision-модель, на весах RealESRGAN_x4plus.pth.

Подключается к vision_analyzer_server.py тем же паттерном, что и
locales.py / vision_analyzer_prompt.py: если torch/numpy не установлены
или файл весов не найден — апскейлинг тихо отключается, сервер продолжает
работать и просто отправляет изображения как есть.

Внешних Real-ESRGAN-пакетов не требует: архитектура сети (RRDBNet) —
это ~80 строк чистого torch, приведённых ниже, той же структуры, что и
у весов x4plus (num_feat=64, num_block=23, num_grow_ch=32, scale=4).
Чекпоинт грузится напрямую (.pth), без обёрток и конвертации.

Единственные новые зависимости — обычные pip-пакеты (без git, без
компиляции C-расширений):
    pip install torch numpy
(на Windows/без GPU это поставит CPU-сборку torch; для CUDA нужен свой
 индекс пакетов — см. https://pytorch.org/get-started/locally/)
"""

import asyncio
import io
import logging
import os

from PIL import Image

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

# по площади/меньшей стороне
MIN_SIDE_PX = 1200

# Коэффициент апскейла и параметры сети — должны соответствовать весам
# в MODEL_PATH. RealESRGAN_x4plus.pth обучен именно с этими значениями,
# менять их отдельно от файла весов нельзя.
SCALE = 4
NUM_FEAT = 64
NUM_BLOCK = 23
NUM_GROW_CH = 32

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "upscaler", "RealESRGAN_x4plus.pth"
)

# Формат, в котором пересохраняется апскейленное изображение перед base64.
OUTPUT_FORMAT = "PNG"

# Апскейл прогоняется одним куском, без тайлинга — это ок, т.к. в эту
# функцию попадают только изображения меньше MIN_SIDE_PX по обеим сторонам
# (т.е. после апскейла — не больше ~MIN_SIDE_PX * SCALE по стороне).


# ---------------------------------------------------------------------------
# Архитектура сети (RRDBNet), инференс-часть.
# Совместима по структуре весов с официальным Real-ESRGAN x4plus.
# ---------------------------------------------------------------------------

def _build_net():
    """Импортирует torch и собирает класс RRDBNet.

    Вынесено в функцию, а не в module-level код, чтобы torch не был
    обязательной зависимостью для импорта этого модуля — только для
    реального использования апскейла.
    """
    import torch
    from torch import nn
    from torch.nn import functional as F

    class ResidualDenseBlock(nn.Module):
        def __init__(self, num_feat=64, num_grow_ch=32):
            super().__init__()
            self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
            self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
            self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, num_feat, num_grow_ch=32):
            super().__init__()
            self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

        def forward(self, x):
            out = self.rdb1(x)
            out = self.rdb2(out)
            out = self.rdb3(out)
            return out * 0.2 + x

    class RRDBNet(nn.Module):
        def __init__(self, num_in_ch=3, num_out_ch=3, scale=4,
                     num_feat=64, num_block=23, num_grow_ch=32):
            super().__init__()
            self.scale = scale
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
            self.body = nn.Sequential(*[
                RRDB(num_feat, num_grow_ch) for _ in range(num_block)
            ])
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            feat = self.conv_first(x)
            body_feat = self.conv_body(self.body(feat))
            feat = feat + body_feat
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
            out = self.conv_last(self.lrelu(self.conv_hr(feat)))
            return out

    return RRDBNet


_model = None           # ленивая инициализация — грузит веса в память/на GPU
_device = None
_init_failed = False    # чтобы не пытаться грузить модель заново на каждый запрос


def _load_model():
    """Ленивая инициализация сети + весов. Возвращает (model, device) или (None, None)."""
    global _model, _device, _init_failed

    if _model is not None or _init_failed:
        return _model, _device

    if not os.path.isfile(MODEL_PATH):
        logging.warning(
            "image_upscaler: файл весов не найден: %s — апскейлинг отключён",
            MODEL_PATH,
        )
        _init_failed = True
        return None, None

    try:
        import torch

        RRDBNet = _build_net()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        net = RRDBNet(
            num_in_ch=3, num_out_ch=3, scale=SCALE,
            num_feat=NUM_FEAT, num_block=NUM_BLOCK, num_grow_ch=NUM_GROW_CH,
        )

        state = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        if "params_ema" in state:
            state = state["params_ema"]
        elif "params" in state:
            state = state["params"]
        net.load_state_dict(state, strict=True)

        net.eval()
        net.to(device)

        _model, _device = net, device
        logging.info(
            "image_upscaler: модель загружена (%s), device=%s", MODEL_PATH, device
        )
    except Exception as e:
        logging.warning(
            "image_upscaler: не удалось инициализировать модель (%s) — "
            "апскейлинг отключён, изображения будут отправляться как есть",
            e,
        )
        _model = None
        _init_failed = True

    return _model, _device


def _needs_upscale(data: bytes) -> bool:
    try:
        with Image.open(io.BytesIO(data)) as img:
            w, h = img.size
    except Exception:
        return False
    return min(w, h) < MIN_SIDE_PX


def _upscale_sync(data: bytes, source_name: str = ""):
    """Синхронный (блокирующий, CPU/GPU-нагруженный) апскейл.

    Вызывать только через run_in_executor — иначе заблокирует event loop
    aiohttp на время инференса.

    Возвращает новые байты изображения либо None, если апскейлинг
    недоступен или упал с ошибкой — в обоих случаях вызывающий код должен
    отправить исходные данные без изменений.
    """
    tag = f"[{source_name}] " if source_name else ""

    net, device = _load_model()
    if net is None:
        return None

    try:
        import numpy as np
        import torch

        with Image.open(io.BytesIO(data)) as img:
            img = img.convert("RGB")
            w0, h0 = img.size
            arr = np.asarray(img, dtype=np.float32) / 255.0

        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

        with torch.no_grad():
            out = net(tensor)

        out = out.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
        out = (out * 255.0).round().astype("uint8")
        sr_image = Image.fromarray(out)

        buf = io.BytesIO()
        sr_image.save(buf, format=OUTPUT_FORMAT)
        result = buf.getvalue()

        logging.info(
            "image_upscaler: %sапскейлено %dx%d -> %dx%d (%d -> %d байт)",
            tag, w0, h0, sr_image.width, sr_image.height, len(data), len(result),
        )
        return result
    except Exception as e:
        logging.warning(
            "image_upscaler: %sошибка при апскейлинге, отправляю оригинал: %s", tag, e
        )
        return None


async def upscale_if_needed(data: bytes, source_name: str = ""):
    tag = f"[{source_name}] " if source_name else ""

    if not _needs_upscale(data):
        logging.debug(
            "image_upscaler: %sапскейл не требуется (изображение уже >= %dpx по стороне)",
            tag, MIN_SIDE_PX,
        )
        return data, None

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _upscale_sync, data, source_name)

    if result is None:
        return data, None

    return result, f"image/{OUTPUT_FORMAT.lower()}"