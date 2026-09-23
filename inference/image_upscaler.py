"""
image_upscaler.py — опциональный апскейлинг маленьких изображений в SCALE раз
перед отправкой в vision-модель, на весах RealESRGAN_x4plus.pth.

Подключается к vision_analyzer_server.py тем же паттерном, что и
locales.py / vision_analyzer_prompt.py: если torch/numpy не установлены
или веса не удалось получить — апскейлинг тихо отключается, сервер
продолжает работать и просто отправляет изображения как есть.

Веса в репозитории не хранятся. При первом РЕАЛЬНОМ апскейле (то есть когда
изображение действительно меньше MIN_SIDE_PX) файл скачивается по WEIGHTS_URL
(по умолчанию — Hugging Face, nateraw/real-esrgan) в MODEL_PATH и остаётся
там на следующие запуски. Пока апскейл выключен (MIN_SIDE_PX = 1), ничего
не скачивается. Скачать заранее можно так:
    python image_upscaler.py

Настройка через переменные окружения (все необязательные):
    VISION_ANALYZER_UPSCALER_DIR     каталог для весов (по умолчанию ./upscaler;
                                     в контейнере укажите путь на volume, чтобы
                                     веса не скачивались заново после пересборки)
    VISION_ANALYZER_UPSCALER_URL     другой URL весов (зеркало и т.п.)
    VISION_ANALYZER_UPSCALER_SHA256  ожидаемый sha256 файла; если задан —
                                     несовпавший файл отбрасывается. Значение
                                     печатается в лог после первой загрузки.

Внешних Real-ESRGAN-пакетов не требует: архитектура сети (RRDBNet) —
это ~80 строк чистого torch, приведённых ниже, той же структуры, что и
у весов x4plus (num_feat=64, num_block=23, num_grow_ch=32, scale=4).
Чекпоинт грузится напрямую (.pth), без обёрток и конвертации.
Скачивание идёт стандартной библиотекой (urllib) — новых зависимостей нет.

Единственные новые зависимости — обычные pip-пакеты (без git, без
компиляции C-расширений):
    pip install torch numpy
(на Windows/без GPU это поставит CPU-сборку torch; для CUDA нужен свой
 индекс пакетов — см. https://pytorch.org/get-started/locally/)
"""

import asyncio
import hashlib
import io
import logging
import os
import threading
import time
import urllib.request

from PIL import Image

# ---------------------------------------------------------------------------
# Конфигурация
# ---------------------------------------------------------------------------

# по площади/меньшей стороне
MIN_SIDE_PX = 1

# Коэффициент апскейла и параметры сети — должны соответствовать весам
# в MODEL_PATH. RealESRGAN_x4plus.pth обучен именно с этими значениями,
# менять их отдельно от файла весов нельзя.
SCALE = 4
NUM_FEAT = 64
NUM_BLOCK = 23
NUM_GROW_CH = 32

WEIGHTS_FILENAME = "RealESRGAN_x4plus.pth"

WEIGHTS_URL = os.environ.get(
    "VISION_ANALYZER_UPSCALER_URL",
    "https://huggingface.co/nateraw/real-esrgan/resolve/main/" + WEIGHTS_FILENAME,
).strip()

MODEL_DIR = os.environ.get("VISION_ANALYZER_UPSCALER_DIR", "").strip() or os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "upscaler"
)
MODEL_PATH = os.path.join(MODEL_DIR, WEIGHTS_FILENAME)

# Необязательная проверка целостности: пустое значение — не проверять.
WEIGHTS_SHA256 = os.environ.get("VISION_ANALYZER_UPSCALER_SHA256", "").strip().lower() or None

# Настоящий чекпоинт весит ~67 МБ. Всё, что заметно меньше (HTML-страница
# ошибки, оборванная загрузка), считается битым файлом и не сохраняется.
MIN_WEIGHTS_BYTES = 50 * 1024 * 1024

DOWNLOAD_TIMEOUT_S = 60     # таймаут сокета (соединение и каждое чтение), не всей загрузки
DOWNLOAD_ATTEMPTS = 3       # сколько раз пробуем скачать за один заход
RETRY_COOLDOWN_S = 300      # после неудачи не долбим сеть на каждый запрос — ждём

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


# ---------------------------------------------------------------------------
# Загрузка весов
# ---------------------------------------------------------------------------

class WeightsIntegrityError(Exception):
    """Скачанный файл не прошёл проверку sha256 — повторять загрузку бессмысленно."""


def _download_weights() -> None:
    """Скачивает веса в MODEL_PATH (синхронно, в потоке вызывающего).

    Пишет во временный файл MODEL_PATH + ".part" и только после проверок
    (полный размер, минимальный размер, sha256 если задан) атомарно
    переименовывает его — недокачанный файл никогда не выглядит готовым.
    При сетевых сбоях делает до DOWNLOAD_ATTEMPTS попыток. Бросает
    исключение, если получить файл не удалось.
    """
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    tmp_path = MODEL_PATH + ".part"
    last_err = None

    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        try:
            logging.info(
                "image_upscaler: скачиваю веса (попытка %d/%d): %s",
                attempt, DOWNLOAD_ATTEMPTS, WEIGHTS_URL,
            )
            req = urllib.request.Request(
                WEIGHTS_URL, headers={"User-Agent": "vision_analyzer/1.0"}
            )
            sha = hashlib.sha256()
            done = 0
            next_pct = 25

            with urllib.request.urlopen(req, timeout=DOWNLOAD_TIMEOUT_S) as resp, \
                    open(tmp_path, "wb") as f:
                total = int(resp.headers.get("Content-Length") or 0)
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
                    sha.update(chunk)
                    done += len(chunk)
                    if total and done * 100 // total >= next_pct:
                        logging.info(
                            "image_upscaler: скачано %d%% (%d из %d МБ)", # f строки кому придумали
                            done * 100 // total, done >> 20, total >> 20,
                        )
                        next_pct += 25

            if total and done != total:
                raise OSError(f"скачано {done} из {total} байт")
            if done < MIN_WEIGHTS_BYTES:
                raise OSError(
                    f"файл слишком мал ({done} байт) — это не чекпоинт весов"
                )

            digest = sha.hexdigest()
            if WEIGHTS_SHA256 and digest != WEIGHTS_SHA256:
                raise WeightsIntegrityError(
                    f"sha256 не совпал: получен {digest}, ожидался {WEIGHTS_SHA256}" # а че тут f строки используешь?
                )

            os.replace(tmp_path, MODEL_PATH)
            logging.info(
                "image_upscaler: веса сохранены: %s (%d байт, sha256=%s)", # f строки кому придумали
                MODEL_PATH, done, digest,
            )
            return
        except WeightsIntegrityError:
            _remove_quietly(tmp_path)
            raise
        except Exception as e:
            last_err = e
            _remove_quietly(tmp_path)
            logging.warning(
                "image_upscaler: попытка %d/%d не удалась: %s", # f строки кому придумали
                attempt, DOWNLOAD_ATTEMPTS, e,
            )
            if attempt < DOWNLOAD_ATTEMPTS:
                time.sleep(2 * attempt)

    raise RuntimeError(f"не удалось скачать веса: {last_err}")


def _remove_quietly(path: str) -> None:
    try:
        os.remove(path)
    except OSError:
        pass


_model = None           # ленивая инициализация — грузит веса в память/на GPU
_device = None
_init_failed = False    # чтобы не пытаться грузить модель заново на каждый запрос
_next_attempt = 0.0     # time.monotonic(): до этого момента после неудачной загрузки не пробуем снова
_load_lock = threading.Lock()   # _load_model вызывается из потоков run_in_executor


def _load_model():
    """Ленивая инициализация сети + весов. Возвращает (model, device) или (None, None).

    Если файла весов нет — сначала скачивает его. Сетевая ошибка не
    отключает апскейл навсегда: следующая попытка будет через
    RETRY_COOLDOWN_S секунд. Постоянные ошибки (нет torch, битый файл)
    отключают его до перезапуска сервера, как и раньше.
    """
    global _model, _device, _init_failed, _next_attempt

    if _model is not None or _init_failed:
        return _model, _device
    if time.monotonic() < _next_attempt:
        return None, None

    with _load_lock:
        # Пока ждали замок, соседний поток мог уже всё сделать.
        if _model is not None or _init_failed:
            return _model, _device
        if time.monotonic() < _next_attempt:
            return None, None

        if not os.path.isfile(MODEL_PATH):
            try:
                _download_weights()
            except WeightsIntegrityError as e:
                logging.warning(
                    "image_upscaler: %s — апскейлинг отключён до перезапуска", e
                )
                _init_failed = True
                return None, None
            except Exception as e:
                logging.warning(
                    "image_upscaler: %s — апскейлинг пропущен, повторная попытка через %d с",
                    e, RETRY_COOLDOWN_S,
                )
                _next_attempt = time.monotonic() + RETRY_COOLDOWN_S
                return None, None

        try:
            import torch

            RRDBNet = _build_net()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            net = RRDBNet(
                num_in_ch=3, num_out_ch=3, scale=SCALE,
                num_feat=NUM_FEAT, num_block=NUM_BLOCK, num_grow_ch=NUM_GROW_CH,
            )

            # weights_only=True: файл теперь приходит из сети, а pickle с
            # weights_only=False может выполнить произвольный код.
            state = torch.load(MODEL_PATH, map_location=device, weights_only=True)
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
                "апскейлинг отключён, изображения будут отправляться как есть. "
                "Если файл весов повреждён, удалите %s — он скачается заново "
                "при следующем запуске",
                e, MODEL_PATH,
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
    aiohttp на время инференса (а при первом запуске — ещё и на время
    скачивания весов).

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


if __name__ == "__main__":
    # Предварительная загрузка весов: python image_upscaler.py
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    if os.path.isfile(MODEL_PATH):
        print(f"Веса уже на месте: {MODEL_PATH}")
    else:
        _download_weights()