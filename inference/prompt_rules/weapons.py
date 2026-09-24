"""Категория weapons_and_dangerous_objects.

Самая детализированная категория: коды признаков W1-W6, overrides,
пороги уровней. Именно поэтому она держится как цельный, вручную
выверенный текст, а не собирается из отдельных полей — разбивать её на
структурные "фичи в YAML" рискует потерять точные формулировки порогов.
"""

BLOCK = """\
<signal_category name="weapons_and_dangerous_objects">
Weapon features. Check every object that is in a hand, on a body, or in the foreground against this list:
W1 Tube shape: a dark or metallic body whose length is at least twice its width, or the same body seen end-on as a dark circle or ring.
W2 Grip: fingers wrapped around a handle-shaped part, or a finger laid along or inside the object.
W3 Aim: the arm is extended or raised, with the object pointing toward the camera, a person, or an animal.
W4 Blade: a flat metal edge with a point, or a straight or curved cutting edge, of any length, with or without a handle.
W5 Firearm parts: barrel, slide, revolver drum, magazine, stock, trigger guard, scope, suppressor.
W6 Other dangerous items: bat, club, chain, brass knuckles, axe, hammer raised toward a person, aerosol or gas canister with a nozzle pointing at a person, bottle with a cloth wick, pipe with end caps or wires.

Registration rule: register a signal when a hand-held or worn object shows W1, W4, W5, or W6. One matching feature is enough. W2 and W3 raise the level (see routing).
Registration depends on features W1 to W6 only. Mood, smiles, clothing, and location are recorded in "description" and leave the signal decision unchanged.

Routing: a registered weapons signal sets needs_human_review to true and risk_level to at least "medium" right away, so the moderator settles what the object is.
Level "high": W1, W4, W5, or W6 together with W2 (gripped) or W3 (aimed), or a blade or firearm part touching a person's body or head.
Level "medium": a single feature W1, W4, W5, or W6 without W2 and W3, for example a dark tube held at the side of the body or lying on a surface within reach.

Overrides (closed list): (a) readable text or a logo on the object that names a specific different product; (b) a transparent or translucent housing that shows the contents; (c) the object is printed on packaging, artwork, or a game or app screen. When an override applies, write the text or the housing description in the rationale and assign "low" for that object.

Wording for "detail": write it in __OUTPUT_LANGUAGE__; name the observed features and the feature codes (the codes W1-W6 stay as written), and name the weapon type when the shape matches one (pistol, revolver, rifle, knife). When the type is unclear, write the __OUTPUT_LANGUAGE__ equivalent of "weapon-like cylindrical object".
Wording for "rationale": write it in __OUTPUT_LANGUAGE__; list the feature codes observed and add one sentence stating that authenticity (real weapon, replica, or prop) is unconfirmed.
</signal_category>\
"""

# Примеры (сцены A/B/C) — единственная категория, у которой они пока
# есть. Каждый язык: сцены в английском описании + JSON-значения на
# языке ответа. Без интро-строки ("Each example shows...") — она общая
# для всех категорий и живёт в prompt_rules/__init__.py, добавляется
# один раз перед всеми собранными примерами.
EXAMPLES = {
    "en": """Scene A: an indoor parking garage, one man in a dark jacket, right arm extended forward at shoulder height, hand closed around a dark object that appears as a dark circle about 3 cm wide facing the camera.
{
  "signals": [
    {"id": "S-1", "category": "weapons_and_dangerous_objects", "detail": "Dark object gripped in the right hand, seen end-on as a dark circle with a muzzle-like opening, arm extended toward the camera (W1, W2, W3); weapon-like cylindrical object."}
  ],
  "rationale": "Features W1, W2, W3 observed; the object is gripped and aimed at the camera. Authenticity (real, replica, or prop) is unconfirmed.",
  "risk_level": "high",
  "needs_human_review": true
}

Scene B: a woman at a bus stop, left hand holds a dark metal tube about 25 cm long against her thigh, arm relaxed and pointing downward, nothing else in hands.
{
  "signals": [
    {"id": "S-1", "category": "weapons_and_dangerous_objects", "detail": "Dark metal tube about 25 cm long held in the left hand at the side of the body, pointing downward (W1); weapon-like cylindrical object."}
  ],
  "rationale": "Feature W1 observed without W2 and W3. Authenticity (real, replica, or prop) is unconfirmed.",
  "risk_level": "medium",
  "needs_human_review": true
}

Scene C: a man in a gray sweater at an office desk, holding a smartphone with a lit screen in one hand and a paper cup in the other; all hands and objects fully visible.
{
  "signals": [],
  "rationale": "Every object in every hand is identified as a smartphone or a paper cup; features W1 to W6 are absent.",
  "risk_level": "low",
  "needs_human_review": false
}""",
    "ru": """Scene A: an indoor parking garage, one man in a dark jacket, right arm extended forward at shoulder height, hand closed around a dark object that appears as a dark circle about 3 cm wide facing the camera.
{
  "signals": [
    {"id": "S-1", "category": "weapons_and_dangerous_objects", "detail": "Тёмный предмет зажат в правой руке, виден с торца как тёмный круг с отверстием, похожим на дульное; рука вытянута к камере (W1, W2, W3); оружиеподобный цилиндрический объект."}
  ],
  "rationale": "Наблюдаются признаки W1, W2, W3: предмет зажат в руке и направлен на камеру. Подлинность (настоящий, реплика или реквизит) не подтверждена.",
  "risk_level": "high",
  "needs_human_review": true
}

Scene B: a woman at a bus stop, left hand holds a dark metal tube about 25 cm long against her thigh, arm relaxed and pointing downward, nothing else in hands.
{
  "signals": [
    {"id": "S-1", "category": "weapons_and_dangerous_objects", "detail": "Тёмная металлическая трубка длиной около 25 см в левой руке у бедра, направлена вниз (W1); оружиеподобный цилиндрический объект."}
  ],
  "rationale": "Наблюдается признак W1 без W2 и W3. Подлинность (настоящий, реплика или реквизит) не подтверждена.",
  "risk_level": "medium",
  "needs_human_review": true
}

Scene C: a man in a gray sweater at an office desk, holding a smartphone with a lit screen in one hand and a paper cup in the other; all hands and objects fully visible.
{
  "signals": [],
  "rationale": "Каждый предмет в каждой руке опознан как смартфон или бумажный стакан; признаки W1-W6 отсутствуют.",
  "risk_level": "low",
  "needs_human_review": false
}""",
}