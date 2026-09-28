#!/usr/bin/env python3
"""
warm_embeddings.py — скачивает (или подтверждает уже скачанными) веса
VISION_ANALYZER_EMBEDDING_MODEL и делает один тестовый вызов, чтобы
убедиться, что модель реально грузится и считает эмбеддинг.

Назначение: запускать на этапе СБОРКИ образа (пока есть сеть наружу), а
не на старте сервера — сам сервер грузит модель лениво, при первом
запросе (см. embeddings.py: _get_model), и если веса не прогреты
заранее, первый пользовательский запрос в проде может уйти в скачивание
с HuggingFace Hub — которое к тому же может не сработать вовсе, если у
контейнера в рантайме нет сети наружу.

Использование:
    python warm_embeddings.py
    # или явно другой моделью, не трогая VISION_ANALYZER_EMBEDDING_MODEL:
    VISION_ANALYZER_EMBEDDING_MODEL=intfloat/multilingual-e5-large python warm_embeddings.py

В Dockerfile (когда он появится в репозитории) — шаг сборки:
    RUN python inference/warm_embeddings.py
Использует переменную окружения VISION_ANALYZER_EMBEDDING_MODEL, если её
задать через --build-arg/ENV до этого шага — прогреется именно та модель,
которая пойдёт в рантайм, а не дефолт.

Выход: код 0 и OK в stdout при успехе; код 1 и текст ошибки при неудаче
(например, HuggingFace Hub недоступен) — годится как шаг сборки, который
должен явно уронить сборку образа, а не тихо оставить сервис без модели.
"""

from __future__ import annotations

import sys
import time

import config
import embeddings


def main() -> int:
    if not config.EMBEDDING_ENABLED:
        print(f"VISION_ANALYZER_EMBEDDING_ENABLED=0 — прогрев пропущен, модель {config.EMBEDDING_MODEL} не тронута.")
        return 0

    print(f"Прогреваю модель эмбеддингов: {config.EMBEDDING_MODEL} ...")
    started = time.monotonic()
    try:
        model = embeddings._get_model()
        vector = list(model.embed(["тестовый текст для проверки загрузки модели"]))[0]
    except embeddings.EmbeddingUnavailableError as exc:
        print(f"ОШИБКА: модель эмбеддингов недоступна: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 — любая иная ошибка загрузки/инференса
        print(f"ОШИБКА: не удалось прогреть модель: {exc}", file=sys.stderr)
        return 1

    elapsed = time.monotonic() - started
    print(f"OK: модель {config.EMBEDDING_MODEL} загружена и работает, размерность={len(vector)}, {elapsed:.1f} с.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
