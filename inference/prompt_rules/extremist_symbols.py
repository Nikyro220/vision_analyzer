"""Категория extremist_or_hate_symbols.

BLOCK включает в себя и <known_phrase_reference> — справочник по
распознаванию арабского текста. Он в оригинальном промпте физически
идёт сразу за этой категорией и тематически с ней связан (распознавание
надписей на флагах/эмблемах), поэтому не выносится в отдельный файл.
"""

BLOCK = """\
<signal_category name="extremist_or_hate_symbols">
A patch, tattoo, flag, gesture, or emblem that matches iconography of a designated extremist, hate, or terrorist group. A black background with white Arabic religious calligraphy and a horizontal sword or blade beneath the text matches this category, because it imitates the jihadist black-standard (raya) flag; register it from the visual composition alone, without identifying the group. A green or non-black background with the same elements is the national flag of Saudi Arabia and receives no signal.
</signal_category>

<known_phrase_reference>
Before reading Arabic text letter by letter, compare it with these phrases and transcribe a close match as that phrase:
- the shahada (لا إله إلا الله محمد رسول الله)
- the takbir (الله أكبر)
- other common Quranic or devotional phrases
Read letter by letter only when the text differs from all of them, and prefix the result with "low legibility:" when the letters are hard to read.
</known_phrase_reference>\
"""
