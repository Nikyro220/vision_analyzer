"""Начальный набор категорий оценивания — перенесён сюда из бывших
inference/categories/*.json как отправная точка: после переноса управления
в vision_app (админ-панель, /panel/categories/) это единственное место,
где хранится содержимое категорий. Используется только один раз — при
первом запуске (см. seed_default_categories в categories_store.py): если
таблица categories уже не пуста, сюда никто не заглядывает.

full/compact хранятся БЕЗ обёртки <signal_category name="...">...</signal_category>
— её достраивает Category._wrap() на лету перед отправкой на сервер анализа
(см. models.py). full_extra/compact_extra — редкий «хвост» после закрывающего
тега (например вспомогательный <known_phrase_reference>), почти всегда пусто.
"""

from __future__ import annotations

DEFAULT_CATEGORIES: list[dict] = [{'name': 'weapons_and_dangerous_objects',
  'title': 'Weapons and dangerous objects',
  'summary': 'Any handheld, worn, or carried object that could be a gun, knife, blade, club, bat, '
             'chain, axe, hammer, brass knuckles, or any elongated/tube-shaped or edged item -- '
             'even if only partially visible, blurred, seen end-on, or just possibly '
             'weapon-shaped. Include this category whenever any object in a hand or foreground is '
             'not clearly and fully identified as something harmless.',
  'full': 'Weapon features. Check every object that is in a hand, on a body, or in the foreground '
          'against this list:\n'
          'W1 Tube shape: a dark or metallic body whose length is at least twice its width, or the '
          'same body seen end-on as a dark circle or ring.\n'
          'W2 Grip: fingers wrapped around a handle-shaped part, or a finger laid along or inside '
          'the object.\n'
          'W3 Aim: the arm is extended or raised, with the object pointing toward the camera, a '
          'person, or an animal.\n'
          'W4 Blade: a flat metal edge with a point, or a straight or curved cutting edge, of any '
          'length, with or without a handle.\n'
          'W5 Firearm parts: barrel, slide, revolver drum, magazine, stock, trigger guard, scope, '
          'suppressor.\n'
          'W6 Other dangerous items: bat, club, chain, brass knuckles, axe, hammer raised toward a '
          'person, aerosol or gas canister with a nozzle pointing at a person, bottle with a cloth '
          'wick, pipe with end caps or wires.\n'
          '\n'
          'Registration rule: register a signal when a hand-held or worn object shows W1, W4, W5, '
          'or W6. One matching feature is enough. W2 and W3 raise the level (see routing).\n'
          'Registration depends on features W1 to W6 only. Mood, smiles, clothing, and location '
          'are recorded in "description" and leave the signal decision unchanged.\n'
          '\n'
          'Routing: a registered weapons signal sets needs_human_review to true and risk_level to '
          'at least "medium" right away, so the moderator settles what the object is.\n'
          'Level "high": W1, W4, W5, or W6 together with W2 (gripped) or W3 (aimed), or a blade or '
          "firearm part touching a person's body or head.\n"
          'Level "medium": a single feature W1, W4, W5, or W6 without W2 and W3, for example a '
          'dark tube held at the side of the body or lying on a surface within reach.\n'
          '\n'
          'Overrides (closed list): (a) readable text or a logo on the object that names a '
          'specific different product; (b) a transparent or translucent housing that shows the '
          'contents; (c) the object is printed on packaging, artwork, or a game or app screen. '
          'When an override applies, write the text or the housing description in the rationale '
          'and assign "low" for that object.\n'
          '\n'
          'Wording for "detail": write it in __OUTPUT_LANGUAGE__; name the observed features and '
          'the feature codes (the codes W1-W6 stay as written), and name the weapon type when the '
          'shape matches one (pistol, revolver, rifle, knife). When the type is unclear, write the '
          '__OUTPUT_LANGUAGE__ equivalent of "weapon-like cylindrical object".\n'
          'Wording for "rationale": write it in __OUTPUT_LANGUAGE__; list the feature codes '
          'observed and add one sentence stating that authenticity (real weapon, replica, or prop) '
          'is unconfirmed.',
  'compact': 'Check every handheld, worn, or foreground object against these features: W1 '
             'tube/cylindrical shape (length at least twice the width, or seen end-on as a dark '
             'circle); W2 gripped like a handle; W3 arm extended or raised, object aimed at the '
             'camera, a person, or an animal; W4 blade or cutting edge; W5 firearm part (barrel, '
             'slide, magazine, stock, scope); W6 other dangerous item (bat, club, chain, brass '
             'knuckles, axe, hammer raised toward a person, gas canister aimed at a person, '
             'cloth-wick bottle, pipe with end caps/wires).\n'
             'Register a signal when W1, W4, W5, or W6 is present alone. Level "high": one of '
             'those together with W2 or W3, or a blade/firearm touching a person. Level "medium": '
             'one of those alone, nothing else. Override to "low" only when a logo or readable '
             'text names a different product, the housing is transparent, or the object is printed '
             'on packaging/artwork/a screen -- name that in the rationale.\n'
             'Wording for "detail": name the feature codes (W1-W6) and the weapon type if clear, '
             'in __OUTPUT_LANGUAGE__. Wording for "rationale": list the codes and note that '
             'authenticity (real, replica, or prop) is unconfirmed.',
  'example_en': 'Scene A: an indoor parking garage, one man in a dark jacket, right arm extended '
                'forward at shoulder height, hand closed around a dark object that appears as a '
                'dark circle about 3 cm wide facing the camera.\n'
                '{\n'
                '  "signals": [\n'
                '    {"id": "S-1", "category": "weapons_and_dangerous_objects", "detail": "Dark '
                'object gripped in the right hand, seen end-on as a dark circle with a muzzle-like '
                'opening, arm extended toward the camera (W1, W2, W3); weapon-like cylindrical '
                'object."}\n'
                '  ],\n'
                '  "rationale": "Features W1, W2, W3 observed; the object is gripped and aimed at '
                'the camera. Authenticity (real, replica, or prop) is unconfirmed.",\n'
                '  "risk_level": "high",\n'
                '  "needs_human_review": true\n'
                '}\n'
                '\n'
                'Scene B: a woman at a bus stop, left hand holds a dark metal tube about 25 cm '
                'long against her thigh, arm relaxed and pointing downward, nothing else in '
                'hands.\n'
                '{\n'
                '  "signals": [\n'
                '    {"id": "S-1", "category": "weapons_and_dangerous_objects", "detail": "Dark '
                'metal tube about 25 cm long held in the left hand at the side of the body, '
                'pointing downward (W1); weapon-like cylindrical object."}\n'
                '  ],\n'
                '  "rationale": "Feature W1 observed without W2 and W3. Authenticity (real, '
                'replica, or prop) is unconfirmed.",\n'
                '  "risk_level": "medium",\n'
                '  "needs_human_review": true\n'
                '}\n'
                '\n'
                'Scene C: a man in a gray sweater at an office desk, holding a smartphone with a '
                'lit screen in one hand and a paper cup in the other; all hands and objects fully '
                'visible.\n'
                '{\n'
                '  "signals": [],\n'
                '  "rationale": "Every object in every hand is identified as a smartphone or a '
                'paper cup; features W1 to W6 are absent.",\n'
                '  "risk_level": "low",\n'
                '  "needs_human_review": false\n'
                '}',
  'example_ru': 'Scene A: an indoor parking garage, one man in a dark jacket, right arm extended '
                'forward at shoulder height, hand closed around a dark object that appears as a '
                'dark circle about 3 cm wide facing the camera.\n'
                '{\n'
                '  "signals": [\n'
                '    {"id": "S-1", "category": "weapons_and_dangerous_objects", "detail": "Тёмный '
                'предмет зажат в правой руке, виден с торца как тёмный круг с отверстием, похожим '
                'на дульное; рука вытянута к камере (W1, W2, W3); оружиеподобный цилиндрический '
                'объект."}\n'
                '  ],\n'
                '  "rationale": "Наблюдаются признаки W1, W2, W3: предмет зажат в руке и направлен '
                'на камеру. Подлинность (настоящий, реплика или реквизит) не подтверждена.",\n'
                '  "risk_level": "high",\n'
                '  "needs_human_review": true\n'
                '}\n'
                '\n'
                'Scene B: a woman at a bus stop, left hand holds a dark metal tube about 25 cm '
                'long against her thigh, arm relaxed and pointing downward, nothing else in '
                'hands.\n'
                '{\n'
                '  "signals": [\n'
                '    {"id": "S-1", "category": "weapons_and_dangerous_objects", "detail": "Тёмная '
                'металлическая трубка длиной около 25 см в левой руке у бедра, направлена вниз '
                '(W1); оружиеподобный цилиндрический объект."}\n'
                '  ],\n'
                '  "rationale": "Наблюдается признак W1 без W2 и W3. Подлинность (настоящий, '
                'реплика или реквизит) не подтверждена.",\n'
                '  "risk_level": "medium",\n'
                '  "needs_human_review": true\n'
                '}\n'
                '\n'
                'Scene C: a man in a gray sweater at an office desk, holding a smartphone with a '
                'lit screen in one hand and a paper cup in the other; all hands and objects fully '
                'visible.\n'
                '{\n'
                '  "signals": [],\n'
                '  "rationale": "Каждый предмет в каждой руке опознан как смартфон или бумажный '
                'стакан; признаки W1-W6 отсутствуют.",\n'
                '  "risk_level": "low",\n'
                '  "needs_human_review": false\n'
                '}',
  'full_extra': '',
  'compact_extra': ''},
 {'name': 'extremist_or_hate_symbols',
  'title': 'Extremist or hate symbols',
  'summary': 'Flags, patches, tattoos, gestures, emblems, or uniforms that could resemble known '
             'extremist, terrorist, or hate-group iconography, including any black flag or banner '
             'with Arabic-style calligraphy and a sword/blade element beneath the text. Include '
             'this category whenever a flag, symbol, or readable Arabic/foreign script appears '
             'anywhere in the frame.',
  'full': 'A patch, tattoo, flag, gesture, or emblem that matches iconography of a designated '
          'extremist, hate, or terrorist group. A black background with white Arabic religious '
          'calligraphy and a horizontal sword or blade beneath the text matches this category, '
          'because it imitates the jihadist black-standard (raya) flag; register it from the '
          'visual composition alone, without identifying the group. A green or non-black '
          'background with the same elements is the national flag of Saudi Arabia and receives no '
          'signal.',
  'compact': 'A patch, tattoo, flag, gesture, or emblem matching known extremist, hate, or '
             'terrorist iconography. A black flag with white Arabic calligraphy and a horizontal '
             'sword or blade beneath the text matches this category (jihadist black-standard '
             'style) -- register it from the visual composition alone, without identifying the '
             'group. The same design on a green or non-black background is the national flag of '
             'Saudi Arabia and receives no signal. Transcribe any readable Arabic text verbatim; '
             'if it resembles a common religious phrase (the shahada, the takbir, or another '
             'common Quranic/devotional phrase), name it as that phrase rather than a literal '
             'letter-by-letter transcription, and prefix a hard-to-read transcription with "low '
             'legibility:".',
  'example_en': '',
  'example_ru': '',
  'full_extra': '<known_phrase_reference>\n'
                'Before reading Arabic text letter by letter, compare it with these phrases and '
                'transcribe a close match as that phrase:\n'
                '- the shahada (لا إله إلا الله محمد رسول الله)\n'
                '- the takbir (الله أكبر)\n'
                '- other common Quranic or devotional phrases\n'
                'Read letter by letter only when the text differs from all of them, and prefix the '
                'result with "low legibility:" when the letters are hard to read.\n'
                '</known_phrase_reference>',
  'compact_extra': ''},
 {'name': 'self_harm_or_violence_indicators',
  'title': 'Self harm or violence indicators',
  'summary': 'Visible blood, open wounds, bruising, scars, a person being restrained, bound, '
             'struck, or held down, or any other sign of physical violence or self-harm on a body. '
             'Include this category whenever skin shows an injury, a mark that could be an injury, '
             "or a person's posture suggests restraint or an attack.",
  'full': 'Any of these visible: open wounds; blood on skin, clothes, or a surface; bruising or '
          'swelling on a face; a person held down or bound (rope, cable ties, handcuffs on a '
          "person without a uniform); a raised fist or object aimed at another person's head or "
          'body; several parallel cuts or scars on a forearm.',
  'compact': 'Any of these visible: open wounds; blood on skin, clothes, or a surface; bruising or '
             'swelling on a face; a person held down or bound (rope, cable ties, handcuffs on a '
             "person without a uniform); a raised fist or object aimed at another person's head or "
             'body; several parallel cuts or scars on a forearm.',
  'example_en': '',
  'example_ru': '',
  'full_extra': '',
  'compact_extra': ''},
 {'name': 'illegal_activity_indicators',
  'title': 'Illegal activity indicators',
  'summary': 'Drug paraphernalia (syringes, pipes, foil, pills, powder, small bags, scales), '
             'cannabis, or cash bundled next to any of these. Include this category whenever any '
             'small unidentified object, powder, pills, or bagged substance is visible on a table, '
             'hand, or surface.',
  'full': 'Any of these visible: syringes; smoking pipes or rolled foil; pills or white powder in '
          'small bags; powder arranged in lines; cannabis leaves or hand-rolled cigarettes with '
          'cannabis; scales next to small bags; stacks of cash next to any of these items.',
  'compact': 'Any of these visible: syringes; smoking pipes or rolled foil; pills or white powder '
             'in small bags; powder arranged in lines; cannabis leaves or hand-rolled cigarettes '
             'with cannabis; scales next to small bags; stacks of cash next to any of these items.',
  'example_en': '',
  'example_ru': '',
  'full_extra': '',
  'compact_extra': ''},
 {'name': 'minor_in_risk_context',
  'title': 'Minor in risk context',
  'summary': 'A child or teenager (under 18) appears anywhere in the frame, especially near any '
             'other object, person, or scene element that could itself be sensitive. Include this '
             'category whenever any person could plausibly be under 18, even if the rest of the '
             'scene looks otherwise ordinary.',
  'full': 'A person in the child or teen age bracket who is in the same frame as a signal from any '
          'other category, or who holds an object that triggered a signal.',
  'compact': 'A person in the child or teen age bracket who is in the same frame as a signal from '
             'any other category, or who holds an object that triggered a signal.',
  'example_en': '',
  'example_ru': '',
  'full_extra': '',
  'compact_extra': ''}]
