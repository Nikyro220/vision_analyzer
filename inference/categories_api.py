"""
categories_api.py — HTTP-хендлеры для GET /categories (см. "Admin API" в
докстринге categories.py). Роуты регистрируются в server.py, бизнес-логика
живёт в categories.py — этот модуль только разбирает HTTP-запрос.

Роуты (все read-only — дефолты правятся вручную в inference/categories/,
а разовые категории на один вызов передаются в POST /analyze, см.
analyze.py и categories.build_overlay):
  GET  /categories            — categories.list_categories()
  GET  /categories/summaries  — categories.get_summaries()
  GET  /categories/<имя>      — categories.get_category(имя)

В build_app() (server.py) статический путь /categories/summaries обязан
быть зарегистрирован раньше динамического /categories/{name} — иначе
aiohttp примет "summaries" за имя категории (резолвер проверяет роуты в
порядке регистрации и останавливается на первом совпадении).
"""

from __future__ import annotations

from aiohttp import web

import categories
import config
from analyze import _json


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