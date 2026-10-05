"""Темы: по JSON-файлу на тему в vision_app/themes/, из них собираются /themes.css и пресеты."""

import json

import pytest

from vision_app import create_app
from vision_app.themes import THEMES_DIR, load_themes, theme_presets, themes_css


@pytest.fixture()
def client(tmp_path):
    app = create_app({
        "TESTING": True, "WTF_CSRF_ENABLED": False, "SECRET_KEY": "test",
        "SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path}/t.db", "UPLOAD_FOLDER": str(tmp_path / "up"),
    })
    return app.test_client()


def _luminance(hex_color: str) -> float:
    c = [int(hex_color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in c]
    return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]


def test_one_json_file_per_theme_and_nothing_else():
    assert sorted(p.suffix for p in THEMES_DIR.iterdir()) == [".json"] * len(list(THEMES_DIR.iterdir()))
    assert {t["id"] for t in load_themes()} == {p.stem for p in THEMES_DIR.glob("*.json")}


def test_all_themes_define_same_variables():
    """Базовый набор переменных у всех тем одинаков; из него вправе выпадать только акцент."""
    optional = {"--accent", "--accent-strong", "--accent-soft", "--on-accent"}
    base = {k for k in next(t for t in load_themes() if t["id"] == "dark")["vars"]}
    for t in load_themes():
        assert set(t["vars"]) - optional == base, t["id"]
        accent_keys = set(t["vars"]) & optional
        assert accent_keys in (set(), optional), f"{t['id']}: акцент задан не полностью"


def test_mode_matches_panel_brightness():
    for t in load_themes():
        light = _luminance(t["vars"]["--bg-panel"]) > 0.4
        assert (t["mode"] == "light") == light, f"{t['id']}: mode не соответствует цвету панели"


def test_groups_are_contiguous_in_presets():
    seen, last = set(), None
    for p in theme_presets():
        if p["group"] != last:
            assert p["group"] not in seen, f"группа «{p['group']}» разорвана"
            seen.add(p["group"])
            last = p["group"]


def test_presets_carry_accent_only_when_theme_has_one():
    by_id = {t["id"]: t for t in load_themes()}
    for p in theme_presets():
        assert ("accent" in p) == ("--accent" in by_id[p["id"]]["vars"])
        assert p["panel"] == by_id[p["id"]]["vars"]["--bg-panel"]


def test_css_is_built_from_json():
    css = themes_css()
    assert ":root,\n[data-theme=\"dark\"] {" in css
    for t in load_themes():
        assert f'[data-theme="{t["id"]}"] {{' in css
    dracula = json.loads((THEMES_DIR / "dracula.json").read_text(encoding="utf-8"))
    assert f'--accent: {dracula["vars"]["--accent"]};' in css


def test_themes_css_route_is_public_and_cacheable(client):
    r = client.get("/themes.css")
    assert r.status_code == 200 and r.mimetype == "text/css"
    assert "max-age=31536000" in r.headers["Cache-Control"]
    assert "[data-theme=\"dracula\"]" in r.get_data(as_text=True)


def test_login_page_links_css_and_inlines_presets(client):
    html = client.get("/accounts/login/").get_data(as_text=True)
    assert "/themes.css?v=" in html
    assert "window.VT_PRESETS" in html and "dracula" in html


def test_order_is_unique_and_alphabetical_within_group():
    pinned = {"dark", "light", "contrast", "contrast-light"}
    groups = {}
    for t in load_themes():
        groups.setdefault(t["group"], []).append(t)
    for name, items in groups.items():
        orders = [t["order"] for t in items]
        assert orders == list(range(1, len(items) + 1)), f"{name}: order должен идти 1..N без пропусков"
        rest = [t["label"].casefold() for t in items if t["id"] not in pinned]
        assert rest == sorted(rest), f"{name}: темы не по алфавиту"
        assert all(t["id"] in pinned for t in items[: len(items) - len(rest)]), f"{name}: базовые темы должны быть первыми"
