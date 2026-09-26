"""
categories_api.py — HTTP-хендлеры для GET/POST /categories (см. TODO/
"Admin API" в докстринге categories.py). Роуты регистрируются в
server.py, вся бизнес-логика (валидация, запись файлов, reload()) живёт
в categories.py — этот модуль только разбирает HTTP-запрос и переводит
categories.CategoryError в ответ 400.

Роуты:
  GET  /categories            — categories.list_categories()
  GET  /categories/summaries  — categories.get_summaries()
  GET  /categories/<имя>      — categories.get_category(имя)
  POST /categories/order      — categories.set_order(...)
  POST /categories/<имя>      — categories.upsert_category(...)

В build_app() (server.py) статические пути /categories/summaries и
/categories/order обязаны быть зарегистрированы раньше динамического
/categories/{name} — иначе aiohttp примет "summaries"/"order" за имя
категории (резолвер проверяет роуты в порядке регистрации и
останавливается на первом совпадении).

POST-тело — как и у /lang, /config, /sampling: JSON-объект
(Content-Type: application/json) либо form-данные. Но examples (вложенный
объект) и position (число) осмысленно передавать только через JSON —
через form-поля вложенный объект не собрать.
"""

from __future__ import annotations

import logging

from aiohttp import web

import categories
import config
from analyze import _json


async def _read_body(request: web.Request) -> dict:
    if not request.can_read_body:
        return {}
    if request.content_type == "application/json":
        return await request.json() or {}
    data = await request.post()
    return dict(data)


async def handle_categories_list(request: web.Request) -> web.Response:
    return _json(categories.list_categories())


async def handle_categories_summaries(request: web.Request) -> web.Response:
    return _json(categories.get_summaries())


async def handle_category_detail(request: web.Request) -> web.Response:
    name = request.match_info["name"]
    record = categories.get_category(name)
    if record is None:
        return _json({"error": config._t("error.category_not_found", name=name)}, status=404)
    return _json(record)


async def handle_categories_order(request: web.Request) -> web.Response:
    """POST /categories/order — {"order": [список всех имён категорий
    в новом порядке]}. Не добавляет и не удаляет категории, только
    переставляет существующие."""
    try:
        body = await _read_body(request)
    except Exception:
        return _json({"error": config._t("error.categories_invalid_json")}, status=400)

    if "order" not in body:
        return _json({"error": config._t("error.categories_order_missing")}, status=400)

    try:
        order = categories.set_order(body["order"])
    except categories.CategoryError as e:
        return _json({"error": str(e)}, status=400)

    logging.info("categories: порядок категорий изменён извне: %s", order)
    return _json({"ok": True, "order": order})


async def handle_category_upsert(request: web.Request) -> web.Response:
    """POST /categories/<имя> — создаёт категорию <имя> (если её ещё
    нет: обязательны summary, full, compact) либо частично обновляет
    существующую (любое подмножество полей). Необязательные поля:
    summary, full, compact (строки), examples (объект {"en":...,
    "ru":...} или null — см. categories._merge_examples), position
    (0-based индекс в order)."""
    name = request.match_info["name"]
    try:
        body = await _read_body(request)
    except Exception:
        return _json({"error": config._t("error.categories_invalid_json")}, status=400)

    if not isinstance(body, dict) or not body:
        return _json({"error": config._t("error.categories_empty_body")}, status=400)

    kwargs = {}
    for field in ("summary", "full", "compact"):
        if field in body:
            kwargs[field] = body[field]
    if "examples" in body:
        kwargs["examples"] = body["examples"]
    if "position" in body and body["position"] is not None and body["position"] != "":
        try:
            kwargs["position"] = int(body["position"])
        except (TypeError, ValueError):
            return _json({"error": config._t("error.categories_invalid_position")}, status=400)

    try:
        record = categories.upsert_category(name, **kwargs)
    except categories.CategoryError as e:
        return _json({"error": str(e)}, status=400)

    logging.info("categories: категория %r создана/обновлена извне", name)
    return _json(record)