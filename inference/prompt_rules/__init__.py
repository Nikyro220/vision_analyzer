"""Реестр категорий сигналов для <risk_signals> и <examples> системного промпта.

Как добавить новую категорию:
1. Создать файл rules_name.py с константой BLOCK — текст вида
   \"\"\"<signal_category name="...">...текст правил...</signal_category>\"\"\",
   ровно как сейчас пишутся остальные категории (см. weapons.py как
   пример самой детализированной, self_harm.py — как пример простой).
2. Импортировать модуль ниже и добавить его в REGISTRY, в том месте
   списка, где категория должна появиться в промпте.
3. Если для категории уже есть проверенные примеры (сцена + готовый
   JSON) — добавить константу EXAMPLES = {"ru": "...", "en": "..."} по
   образцу weapons.py: без интро-строки (она общая, добавляется здесь
   один раз), просто сцены. Если примеров пока нет — константу EXAMPLES
   просто не определять, сборка её пропустит сама.
4. Если категория должна влиять на "high"/"medium" в
   <risk_level_guidance> — руками добавить формулировку в prompt.py
   (RISK_LEVEL_GUIDANCE). Это НЕ автоматизировано специально: там
   естественный язык с нюансами порядка фраз, автосклейка из метаданных
   рискует дать корявый или неточный текст. Разбиение на файлы решает
   проблему организации кода, не проблему подбора точных формулировок.

Существующие категории трогать не нужно — добавление новой не требует
редактировать чужие файлы.
"""

from . import extremist_symbols, illegal_activity, minor_in_risk, self_harm, weapons

REGISTRY = [weapons, extremist_symbols, self_harm, illegal_activity, minor_in_risk]

# Склейка блоков в текст <risk_signals> — простая конкатенация с пустой
# строкой между блоками, порядок = порядок REGISTRY. Ничего не
# "понимает" в содержимом блоков, поэтому безопасна для любого текста.
RISK_SIGNALS_BLOCK = "\n\n".join(rule.BLOCK for rule in REGISTRY)

# Сборка <examples>. Интро-строка объясняет формат примеров и то, на
# каком языке написаны значения — она общая для всех категорий, не
# специфична ни для одной, поэтому живёт здесь, а не в файле категории.
# Сами сцены (EXAMPLES) категория предоставляет опционально: если у неё
# пока нет примеров, она просто не определяет константу EXAMPLES, и
# сборка её пропускает — ничего не ломается.
_EXAMPLES_INTRO = {
    "en": (
        "Each example shows only the analysis fields; write all fields from output_format.\n"
        "The examples are in English; write your own free-text values in __OUTPUT_LANGUAGE__ "
        "(feature codes W1-W6 and category names stay as shown)."
    ),
    "ru": (
        "Each example shows only the analysis fields; write all fields from output_format.\n"
        "The text values in the examples are in Russian, like your own text values."
    ),
}


def _examples_block(lang: str) -> str:
    scenes = [
        rule.EXAMPLES[lang]
        for rule in REGISTRY
        if getattr(rule, "EXAMPLES", None) and lang in rule.EXAMPLES
    ]
    parts = [_EXAMPLES_INTRO[lang], *scenes]
    return "\n\n".join(parts) + "\n"


EXAMPLES_BLOCKS = {lang: _examples_block(lang) for lang in _EXAMPLES_INTRO}