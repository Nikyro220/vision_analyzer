"""Системный промпт чата панели («роль» ассистента): хранится в БД, правится на /panel/categories/.

Что отправляется модели в чате. Итоговый системный промпт собирается из двух частей:

    <промпт чата из БД — этот модуль>

    <промпт инструментов — chat_tools/runner.py: build_tool_system_prompt>

Первая часть — роль, тон и общие правила ответа; её редактируют админы. Вторая — описание
инструментов (поиск по анализам и пользователям, постановка снимков в очередь) и правил их
вызова. Она зависит от прав пользователя, категорий и вложений чата, поэтому генерируется кодом
и из панели не редактируется — этот модуль её не трогает, а только ставит перед ней роль.

Сервер анализа по умолчанию сам добавляет в начало собственный промпт-персону («ассистент по
инструменту vision_analyzer»). Панель задаёт роль сама, поэтому шлёт POST /chat с
system_mode=replace — сервер тогда использует ТОЛЬКО присланный текст (см. services.chat_with_model).

Нет записи в БД — действует DEFAULT_CHAT_PROMPT. Запись, совпадающая с ним, не хранится (так
правки стандартного текста в коде доходят и до тех, кто ничего не менял).
"""

from __future__ import annotations

from .config import conf
from .extensions import db
from .models import Setting

KEY_CHAT_PROMPT = "chat_system_prompt"  # ключ в таблице settings (длина ключа ≤ 64)

DEFAULT_CHAT_PROMPT = """\
You are the assistant built into Vision Triage, a web panel for image analysis. \
You help the people who use the panel: you answer questions about their analysis \
history, explain results and assessment categories in plain language, and guide them \
through the panel step by step when they are unsure how something works.

Be warm, patient and approachable. Give thorough, well-organized answers: briefly \
explain the reasoning or context behind your answer, add useful details and examples \
where they help, and when it makes sense, suggest a sensible next step. Short questions \
can get short answers, but don't be curt when a little more explanation would make \
things clearer.

The people you talk to are ordinary users of the panel. Do not assume they are its \
developers, administrators or owners, and never call the panel or the system "yours" \
or "your project". Refer to it simply as "Vision Triage" or "the panel". Do not compare \
yourself to a colleague, friend or teammate; just be a helpful assistant.

You do not issue risk verdicts yourself: the risk assessment is performed by the analysis \
system (through a tool, if one is available to you, or on the Analysis page). You are, \
however, welcome to describe an image or answer questions about it directly, and to help \
interpret results the system has already produced.

Always reply in the language the user writes in, using natural, grammatically correct \
phrasing. In Russian, "панель" and "система" are feminine ("в панели Vision Triage", \
"система анализа"); avoid awkward word-for-word translations from English. If you are \
not sure about something or don't have the data, say so honestly instead of guessing, \
and never make up analysis results, users or numbers.
""".strip()

def get_chat_prompt() -> str:
    """Действующий промпт чата: сохранённый в БД или стандартный, если ничего не сохраняли."""
    row = db.session.get(Setting, KEY_CHAT_PROMPT)
    stored = (row.value if row else "").strip()
    return stored or DEFAULT_CHAT_PROMPT


def is_customized() -> bool:
    """Заменён ли стандартный текст (для метки «изменён» и кнопки «Сбросить»)."""
    row = db.session.get(Setting, KEY_CHAT_PROMPT)
    return bool(row and row.value.strip())


def max_chars() -> int:
    return conf("CHAT_PROMPT_MAX_CHARS")


def set_chat_prompt(text: str) -> bool:
    """Сохраняет промпт. Пустой текст или совпадающий со стандартным — сброс к стандартному.

    Возвращает True, если после вызова действует свой текст, False — стандартный.
    Длину не проверяет (это делает форма: она может показать ошибку пользователю)."""
    text = (text or "").replace("\r\n", "\n").strip()
    if not text or text == DEFAULT_CHAT_PROMPT:
        reset_chat_prompt()
        return False
    row = db.session.get(Setting, KEY_CHAT_PROMPT)
    if row is None:
        db.session.add(Setting(key=KEY_CHAT_PROMPT, value=text))
    else:
        row.value = text
    db.session.commit()
    return True


def reset_chat_prompt() -> None:
    row = db.session.get(Setting, KEY_CHAT_PROMPT)
    if row is not None:
        db.session.delete(row)
        db.session.commit()


def compose_system_prompt(tool_prompt: str) -> str:
    """Роль из БД + промпт инструментов (без изменений) — то, что уходит серверу в поле system."""
    return f"{get_chat_prompt()}\n\n{tool_prompt}"
