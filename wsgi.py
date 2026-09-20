"""Точка входа для продакшена: gunicorn wsgi:app"""
from vision_app import create_app

app = create_app()
