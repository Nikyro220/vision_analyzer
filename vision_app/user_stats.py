"""Сводка по пользователю для страниц «Профиль» и «Пользователь» в админ-панели."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

from sqlalchemy import case, func, select

from .extensions import db
from .models import AnalysisResult, ChatMessage, ChatRole, ChatSession, RiskLevel, Status, User, utcnow


def user_stats(user: User) -> dict[str, Any]:
    """Всё, что мы знаем о работе пользователя в системе. Анализы считаются все (в очереди,
    в обработке и готовые) — «сколько анализов сделал»; разбивка по риску — только по готовым."""
    now = utcnow()
    mine = AnalysisResult.user_id == user.id

    def count_if(cond):
        return func.coalesce(func.sum(case((cond, 1), else_=0)), 0)

    done = AnalysisResult.status == Status.DONE
    row = db.session.execute(
        select(
            func.count(AnalysisResult.id),
            count_if(done),
            count_if(AnalysisResult.status != Status.DONE),
            count_if(done & (AnalysisResult.risk_level == RiskLevel.HIGH)),
            count_if(done & (AnalysisResult.risk_level == RiskLevel.MEDIUM)),
            count_if(done & (AnalysisResult.risk_level == RiskLevel.LOW)),
            count_if(done & (AnalysisResult.needs_human_review.is_(True))),
            count_if(done & (AnalysisResult.error != "")),
            count_if(AnalysisResult.source_url != ""),
            count_if(AnalysisResult.created_at >= now - timedelta(days=7)),
            count_if(AnalysisResult.created_at >= now - timedelta(days=30)),
            func.min(AnalysisResult.created_at),
            func.max(AnalysisResult.created_at),
        ).where(mine)
    ).one()
    keys = (
        "total", "done", "active", "high", "medium", "low", "needs_review", "with_error", "by_link",
        "last_7_days", "last_30_days", "first_at", "last_at",
    )
    stats: dict[str, Any] = dict(zip(keys, row))
    stats["unknown"] = stats["done"] - stats["high"] - stats["medium"] - stats["low"]

    stats["chats"] = db.session.scalar(select(func.count(ChatSession.id)).where(ChatSession.user_id == user.id)) or 0
    stats["chat_messages"] = db.session.scalar(
        select(func.count(ChatMessage.id))
        .join(ChatSession, ChatSession.id == ChatMessage.session_id)
        .where(ChatSession.user_id == user.id, ChatMessage.role == ChatRole.USER)
    ) or 0

    created = user.created_at if user.created_at.tzinfo else user.created_at.replace(tzinfo=now.tzinfo)
    stats["account_days"] = max((now - created).days, 0)
    return stats
