"""SECRET_KEY: в проде обязан быть задан, иначе приложение не стартует; в тестах и debug
подставляется дефолтный ключ, чтобы всё работало как раньше (см. create_app в vision_app/__init__.py)."""

import pytest

import vision_app.config as config
from vision_app import create_app


def _base_config(**extra) -> dict:
    cfg = {"SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:", "WTF_CSRF_ENABLED": False}
    cfg.update(extra)
    return cfg


def test_production_without_secret_key_raises(monkeypatch):
    """Без FLASK_SECRET_KEY в проде create_app падает с ошибкой."""
    monkeypatch.setattr(config.Config, "SECRET_KEY", None)
    monkeypatch.delenv("FLASK_DEBUG", raising=False)
    with pytest.raises(RuntimeError, match="FLASK_SECRET_KEY"):
        create_app(_base_config())


def test_production_with_explicit_key_ok(monkeypatch):
    """Заданный ключ используется и не перезаписывается."""
    monkeypatch.setattr(config.Config, "SECRET_KEY", "real-secret")
    monkeypatch.delenv("FLASK_DEBUG", raising=False)
    app = create_app(_base_config())
    assert app.config["SECRET_KEY"] == "real-secret"


def test_empty_secret_key_is_treated_as_unset(monkeypatch):
    """Пустой FLASK_SECRET_KEY тоже считается «не задан» -> ошибка в проде."""
    monkeypatch.setattr(config.Config, "SECRET_KEY", "")
    monkeypatch.delenv("FLASK_DEBUG", raising=False)
    with pytest.raises(RuntimeError, match="FLASK_SECRET_KEY"):
        create_app(_base_config())


def test_debug_uses_default_when_missing(monkeypatch):
    """FLASK_DEBUG=1: дефолтный ключ подставляется сам."""
    monkeypatch.setattr(config.Config, "SECRET_KEY", None)
    monkeypatch.delenv("FLASK_DEBUG", raising=False)
    monkeypatch.setenv("FLASK_DEBUG", "1")
    app = create_app(_base_config())
    assert app.config["SECRET_KEY"] == "dev-insecure-change-me"


def test_testing_uses_default_when_missing(monkeypatch):
    """TESTING=True: дефолтный ключ подставляется."""
    monkeypatch.setattr(config.Config, "SECRET_KEY", None)
    app = create_app(_base_config(TESTING=True))
    assert app.config["SECRET_KEY"] == "dev-insecure-change-me"
