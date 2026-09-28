/* ============================================================
   Vision Triage — движок визуальной темы.

   Подключается в <head> синхронно, чтобы тема применялась до первой
   отрисовки (без вспышки тёмной темы у пользователей светлой).

   Всё хранится локально в браузере (localStorage), на сервер ничего
   не уходит. Состояние:  { v: 1, preset: "auto"|"dark"|"light"|"contrast",
                            values: { <ключ поля>: <значение> } }
   В values лежат только явные переопределения пользователя — всё, чего
   там нет, берётся из CSS (пресет + значения по умолчанию).

   Модульность: настройки описаны реестром MODULES ниже. Чтобы добавить
   новую настройку, достаточно добавить поле в подходящий модуль (или
   новый модуль) — UI (theme-ui.js) и импорт/экспорт подхватят его сами.
   ============================================================ */
(function (global) {
    "use strict";

    var KEY = "vt-theme-v1";
    var root = document.documentElement;

    // ---------------------------------------------------------------- цвет
    function parseColor(str) {
        if (!str) return null;
        str = String(str).trim();
        var m = /^#([0-9a-f]{3}|[0-9a-f]{6})$/i.exec(str);
        if (m) {
            var h = m[1];
            if (h.length === 3) h = h.replace(/./g, "$&$&");
            return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
        }
        m = /^rgba?\(\s*(\d+)[\s,]+(\d+)[\s,]+(\d+)/i.exec(str);
        if (m) return [+m[1], +m[2], +m[3]];
        return null;
    }
    function toHex(rgb) {
        return "#" + rgb.map(function (n) {
            n = Math.max(0, Math.min(255, Math.round(n)));
            return (n < 16 ? "0" : "") + n.toString(16);
        }).join("");
    }
    function mix(a, b, t) {
        return [0, 1, 2].map(function (i) { return a[i] + (b[i] - a[i]) * t; });
    }
    function luminance(rgb) {
        var c = rgb.map(function (v) {
            v /= 255;
            return v <= 0.03928 ? v / 12.92 : Math.pow((v + 0.055) / 1.055, 2.4);
        });
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2];
    }
    function contrast(a, b) {
        var la = luminance(a), lb = luminance(b);
        return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
    }
    var WHITE = [255, 255, 255], BLACK = [0, 0, 0];

    // Цвет текста поверх списка фонов (массив rgb-массивов): тёмный или белый — что читаемее.
    function onColor(bgs) {
        var dark = [11, 17, 19];
        function worst(fg) {
            return Math.min.apply(null, bgs.map(function (b) { return contrast(fg, b); }));
        }
        return worst(dark) >= worst(WHITE) ? toHex(dark) : "#ffffff";
    }

    // Подтягивает цвет к белому (тёмные темы) или к чёрному (светлая) до
    // достижения читаемого контраста с фоном панели.
    function readable(rgb, bg, isLight, min) {
        var target = isLight ? BLACK : WHITE;
        var t = 0.2, out = mix(rgb, target, t);
        while (contrast(out, bg) < min && t < 1) {
            t += 0.1;
            out = mix(rgb, target, t);
        }
        return out;
    }

    // ------------------------------------------------------------- пресеты
    var PRESETS = [
        { id: "auto", label: "Авто", hint: "Как в системе" },
        { id: "dark", label: "Тёмная", hint: "Стандартная" },
        { id: "light", label: "Светлая", hint: "Для яркого освещения" },
        { id: "contrast", label: "Контрастная", hint: "Максимальная читаемость" }
    ];
    // Фон панели каждой схемы — нужен для расчёта читаемости производных цветов
    var PANEL_BG = { dark: "#141a1e", light: "#ffffff", contrast: "#000000" };

    // ------------------------------------------------------------- шрифты
    var DEFAULT_BODY = '-apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif';
    var DEFAULT_MONO = 'ui-monospace, "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace';
    var SERIF = 'Georgia, "Iowan Old Style", "Palatino Linotype", "Times New Roman", serif';

    var FONTS_BODY = [
        { id: "system", label: "Системный (по умолчанию)", stack: null },
        { id: "humanist", label: "Гуманистический", stack: '"Segoe UI", "Trebuchet MS", "Gill Sans", Candara, Ubuntu, sans-serif' },
        { id: "grotesk", label: "Гротеск", stack: '"Helvetica Neue", Helvetica, Arial, "Liberation Sans", sans-serif' },
        { id: "readable", label: "Читаемый (широкий)", stack: '"Atkinson Hyperlegible", Verdana, Tahoma, "DejaVu Sans", sans-serif' },
        { id: "serif", label: "С засечками", stack: SERIF },
        { id: "mono", label: "Моноширинный", stack: DEFAULT_MONO },
        { id: "custom", label: "Свой шрифт…", stack: "custom" }
    ];
    var FONTS_MONO = [
        { id: "default", label: "Моноширинный (по умолчанию)", stack: null },
        { id: "coder", label: "Программистский", stack: '"JetBrains Mono", "Fira Code", "Cascadia Code", ui-monospace, Consolas, monospace' },
        { id: "body", label: "Как основной шрифт", stack: "var(--font-body)" },
        { id: "serif", label: "С засечками", stack: SERIF },
        { id: "custom", label: "Свой шрифт…", stack: "custom" }
    ];

    // В CSS попадает только «безопасное» имя шрифта: буквы, цифры, пробел, дефис, точка.
    function cleanFontName(s) {
        return String(s == null ? "" : s).replace(/[^A-Za-z0-9\u0400-\u04FF _.\-]/g, "").trim().slice(0, 60);
    }
    function fontStack(list, id, custom, fallback) {
        var item = list.filter(function (f) { return f.id === id; })[0];
        if (!item || item.stack === null) return null;
        if (item.stack === "custom") {
            var name = cleanFontName(custom);
            return name ? '"' + name + '", ' + fallback : null;
        }
        return item.stack;
    }

    // ---------------------------------------------------------------- реестр
    // kind: color | range | select | text | toggle
    // scope "palette" — сбрасывается при смене пресета; остальное пресет не трогает.
    var MODULES = [
        {
            id: "accent", title: "Акцент",
            hint: "Цвет кнопок, активных пунктов меню, ссылок и фокуса. Пресеты темы его не меняют.",
            fields: [
                { key: "accent", label: "Акцентный цвет", kind: "color", cssVar: "--accent", swatches: true },
                { key: "accentStrong", label: "Акцент: светлый вариант", kind: "color", cssVar: "--accent-strong",
                  hint: "Ссылки и наведение. По умолчанию подбирается автоматически под акцент и тему." }
            ]
        },
        {
            id: "surfaces", title: "Поверхности", scope: "palette",
            hint: "Фоны и границы. Смена пресета сбрасывает эти цвета к палитре пресета.",
            fields: [
                { key: "bg", label: "Фон страницы", kind: "color", cssVar: "--bg" },
                { key: "bgPanel", label: "Панели и карточки", kind: "color", cssVar: "--bg-panel" },
                { key: "bgElevated", label: "Поля ввода и вложенные блоки", kind: "color", cssVar: "--bg-elevated" },
                { key: "bgHover", label: "Наведение", kind: "color", cssVar: "--bg-hover" },
                { key: "border", label: "Границы", kind: "color", cssVar: "--border" },
                { key: "borderSoft", label: "Мягкие границы", kind: "color", cssVar: "--border-soft" }
            ]
        },
        {
            id: "text", title: "Текст", scope: "palette",
            hint: "Цвета текста. Рядом показан контраст с фоном (норма WCAG AA — от 4.5).",
            contrastPairs: [
                { fg: "text", bg: "bg", label: "Основной / фон" },
                { fg: "textMuted", bg: "bgPanel", label: "Приглушённый / панель" },
                { fg: "textDim", bg: "bgPanel", label: "Слабый / панель" },
                { fg: "accentStrong", bg: "bgPanel", label: "Акцент / панель" }
            ],
            fields: [
                { key: "text", label: "Основной текст", kind: "color", cssVar: "--text" },
                { key: "textMuted", label: "Приглушённый текст", kind: "color", cssVar: "--text-muted" },
                { key: "textDim", label: "Слабый текст (подписи, даты)", kind: "color", cssVar: "--text-dim" }
            ]
        },
        {
            id: "status", title: "Статусы и предупреждения", scope: "palette",
            hint: "Цвета уровней риска, ошибок и предупреждений. Цвета ролей пользователей не настраиваются.",
            fields: [
                { key: "riskLow", label: "Низкий риск / успех", kind: "color", cssVar: "--risk-low" },
                { key: "riskMedium", label: "Средний риск / предупреждение", kind: "color", cssVar: "--risk-medium" },
                { key: "riskHigh", label: "Высокий риск", kind: "color", cssVar: "--risk-high" },
                { key: "danger", label: "Опасные действия / ошибки", kind: "color", cssVar: "--danger" }
            ]
        },
        {
            id: "typography", title: "Типографика",
            hint: "Свой шрифт должен быть установлен на вашем устройстве — из интернета шрифты не подгружаются.",
            fields: [
                { key: "font", label: "Основной шрифт", kind: "font", list: FONTS_BODY, defaultId: "system", customKey: "fontCustom" },
                { key: "fontMono", label: "Шрифт заголовков и данных", kind: "font", list: FONTS_MONO, defaultId: "default", customKey: "fontMonoCustom" },
                { key: "fontSize", label: "Размер текста", kind: "range", min: 13, max: 19, step: 1, unit: "px", cssVar: "--font-size-base", def: 15 },
                { key: "lineHeight", label: "Межстрочный интервал", kind: "range", min: 1.3, max: 1.8, step: 0.05, unit: "", cssVar: "--line-height-base", def: 1.5 }
            ]
        },
        {
            id: "shape", title: "Форма и компоновка",
            hint: "Скругления и размеры основных областей.",
            fields: [
                { key: "radius", label: "Скругление углов", kind: "range", min: 0, max: 14, step: 1, unit: "px", cssVar: "--radius", def: 3 },
                { key: "contentMax", label: "Максимальная ширина контента", kind: "range", min: 800, max: 1800, step: 50, unit: "px", cssVar: "--content-max", def: 1100 },
                { key: "sidebarW", label: "Ширина боковой панели", kind: "range", min: 200, max: 320, step: 10, unit: "px", cssVar: "--sidebar-w", def: 240 }
            ]
        },
        {
            id: "behavior", title: "Поведение",
            fields: [
                { key: "reduceMotion", label: "Уменьшить анимации", kind: "toggle", def: false,
                  hint: "Отключает плавные переходы и анимации интерфейса." }
            ]
        }
    ];

    var FIELDS = {};
    MODULES.forEach(function (m) { m.fields.forEach(function (f) { FIELDS[f.key] = f; }); });
    // Ключи-«спутники» (свой шрифт) тоже сохраняются
    var EXTRA_KEYS = { fontCustom: "text", fontMonoCustom: "text" };

    // ------------------------------------------------------------- состояние
    var state = { v: 1, preset: "auto", values: {} };
    var listeners = [];
    var undoSnapshot = null;

    function clone(o) { return JSON.parse(JSON.stringify(o)); }

    // Проверка значения: то, что не прошло, просто отбрасывается.
    function sanitizeValue(key, val) {
        if (EXTRA_KEYS[key]) {
            var n = cleanFontName(val);
            return n || undefined;
        }
        var f = FIELDS[key];
        if (!f) return undefined;
        if (f.kind === "color") {
            return typeof val === "string" && /^#[0-9a-f]{6}$/i.test(val) ? val.toLowerCase() : undefined;
        }
        if (f.kind === "range") {
            val = Number(val);
            if (!isFinite(val)) return undefined;
            val = Math.max(f.min, Math.min(f.max, val));
            return Math.round(val * 100) / 100;
        }
        if (f.kind === "font") {
            return f.list.some(function (o) { return o.id === val; }) ? val : undefined;
        }
        if (f.kind === "toggle") return val === true || val === "true" ? true : undefined;
        return undefined;
    }

    function sanitizeState(raw) {
        var out = { v: 1, preset: "auto", values: {} };
        if (!raw || typeof raw !== "object") return out;
        if (PRESETS.some(function (p) { return p.id === raw.preset; })) out.preset = raw.preset;
        var vals = raw.values && typeof raw.values === "object" ? raw.values : {};
        Object.keys(vals).forEach(function (k) {
            var v = sanitizeValue(k, vals[k]);
            if (v !== undefined) out.values[k] = v;
        });
        return out;
    }

    function load() {
        try {
            var raw = localStorage.getItem(KEY);
            if (raw) state = sanitizeState(JSON.parse(raw));
        } catch (e) { /* localStorage недоступен или повреждён — работаем с дефолтами */ }
    }
    function save() {
        try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) { /* приватный режим и т.п. */ }
    }

    // --------------------------------------------------------------- применение
    var applied = [];   // inline-переменные, выставленные нами (чтобы уметь их снимать)
    var defaultsCache = {};

    function resolvedTheme() {
        if (state.preset !== "auto") return state.preset;
        return global.matchMedia && global.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
    }

    function clearInline() {
        applied.forEach(function (name) { root.style.removeProperty(name); });
        applied = [];
    }
    function setInline(name, value) {
        root.style.setProperty(name, value);
        applied.push(name);
    }

    function apply() {
        var theme = resolvedTheme();
        var v = state.values;
        root.setAttribute("data-theme", theme);
        clearInline();
        defaultsCache = {};

        // Прямые значения полей
        Object.keys(FIELDS).forEach(function (key) {
            var f = FIELDS[key];
            if (v[key] === undefined) return;
            if (f.kind === "color") setInline(f.cssVar, v[key]);
            else if (f.kind === "range") setInline(f.cssVar, v[key] + f.unit);
        });

        var fb = fontStack(FONTS_BODY, v.font, v.fontCustom, DEFAULT_BODY);
        if (fb) setInline("--font-body", fb);
        var fm = fontStack(FONTS_MONO, v.fontMono, v.fontMonoCustom, DEFAULT_MONO);
        if (fm) setInline("--font-mono", fm);

        // Производные цвета
        var isLight = theme === "light";
        var panel = parseColor(v.bgPanel) || parseColor(PANEL_BG[theme]);

        if (v.accent || theme !== "dark") {
            var base = parseColor(v.accent) || parseColor("#3fa7a0");
            if (!v.accentStrong) setInline("--accent-strong", toHex(readable(base, panel, isLight, 4.5)));
            setInline("--accent-soft", "rgba(" + base.map(Math.round).join(",") + ",0.14)");
            setInline("--on-accent", onColor([base]));
        }
        if (v.danger) {
            setInline("--danger-strong", toHex(readable(parseColor(v.danger), panel, isLight, 4.5)));
        }
        if (v.riskLow || v.riskMedium || v.riskHigh) {
            var def = getDefaults();
            var pills = [v.riskLow || def.riskLow, v.riskMedium || def.riskMedium, v.riskHigh || def.riskHigh]
                .map(parseColor).filter(Boolean);
            if (pills.length) setInline("--on-risk", onColor(pills));
        }

        root.setAttribute("data-motion", v.reduceMotion ? "reduced" : "normal");
    }

    // Значения «из CSS» (пресет + дефолты) без пользовательских переопределений.
    function getDefaults() {
        var theme = resolvedTheme();
        if (defaultsCache._theme === theme && defaultsCache.bg) return defaultsCache;
        var saved = applied.map(function (n) { return [n, root.style.getPropertyValue(n)]; });
        saved.forEach(function (p) { root.style.removeProperty(p[0]); });
        var cs = getComputedStyle(root), out = { _theme: theme };
        Object.keys(FIELDS).forEach(function (key) {
            var f = FIELDS[key];
            if (f.cssVar) out[key] = cs.getPropertyValue(f.cssVar).trim();
        });
        saved.forEach(function (p) { root.style.setProperty(p[0], p[1]); });
        defaultsCache = out;
        return out;
    }

    // Текущее эффективное значение поля (с учётом переопределения)
    function effective(key) {
        var f = FIELDS[key];
        if (state.values[key] !== undefined) return state.values[key];
        if (!f) return undefined;
        if (f.kind === "color") {
            // Производные цвета — то, что реально применено сейчас
            if (key === "accentStrong" || key === "accent") {
                return getComputedStyle(root).getPropertyValue(f.cssVar).trim();
            }
            return getDefaults()[key];
        }
        if (f.kind === "range" || f.kind === "toggle") return f.def;
        if (f.kind === "font") return f.defaultId;
        return undefined;
    }

    function notify() { listeners.forEach(function (fn) { try { fn(state); } catch (e) { /* ignore */ } }); }
    function commit() { apply(); save(); notify(); }
    function remember() { undoSnapshot = clone(state); }

    // -------------------------------------------------------------------- API
    var API = {
        KEY: KEY,
        modules: MODULES,
        presets: PRESETS,
        fields: FIELDS,
        color: { parse: parseColor, toHex: toHex, contrast: contrast },

        get: function () { return clone(state); },
        resolved: resolvedTheme,
        isOverridden: function (key) { return state.values[key] !== undefined; },
        effective: effective,
        defaultOf: function (key) {
            var f = FIELDS[key];
            if (!f) return undefined;
            if (f.kind === "color") return getDefaults()[key];
            if (f.kind === "font") return f.defaultId;
            return f.def;
        },
        hasPaletteOverrides: function () {
            return Object.keys(state.values).some(function (k) {
                return FIELDS[k] && MODULES.some(function (m) { return m.scope === "palette" && m.fields.indexOf(FIELDS[k]) >= 0; });
            });
        },

        setValue: function (key, val) {
            var clean = sanitizeValue(key, val);
            if (clean === undefined) return this.resetValue(key);
            state.values[key] = clean;
            commit();
        },
        resetValue: function (key) {
            delete state.values[key];
            var f = FIELDS[key];
            if (f && f.customKey) delete state.values[f.customKey];
            commit();
        },
        // Пресет меняет только палитру: акцент, шрифты и форма остаются как были.
        setPreset: function (id) {
            if (!PRESETS.some(function (p) { return p.id === id; })) return;
            remember();
            state.preset = id;
            MODULES.forEach(function (m) {
                if (m.scope !== "palette") return;
                m.fields.forEach(function (f) { delete state.values[f.key]; });
            });
            commit();
        },
        resetModule: function (id) {
            var m = MODULES.filter(function (x) { return x.id === id; })[0];
            if (!m) return;
            remember();
            m.fields.forEach(function (f) {
                delete state.values[f.key];
                if (f.customKey) delete state.values[f.customKey];
            });
            commit();
        },
        resetAll: function () {
            remember();
            state = { v: 1, preset: "auto", values: {} };
            commit();
        },
        canUndo: function () { return !!undoSnapshot; },
        undo: function () {
            if (!undoSnapshot) return;
            state = undoSnapshot;
            undoSnapshot = null;
            commit();
        },

        exportJSON: function () {
            return JSON.stringify({ app: "vision-triage-theme", v: 1, preset: state.preset, values: state.values }, null, 2);
        },
        importJSON: function (text) {
            var raw;
            try { raw = JSON.parse(text); } catch (e) { throw new Error("Файл не является корректным JSON."); }
            if (!raw || raw.app !== "vision-triage-theme") throw new Error("Это не файл темы Vision Triage.");
            remember();
            state = sanitizeState(raw);
            commit();
        },

        onChange: function (fn) { listeners.push(fn); }
    };
    global.VTheme = API;

    // ------------------------------------------------------------------ старт
    load();
    apply();

    // «Авто»: реагируем на смену системной темы
    if (global.matchMedia) {
        var mq = global.matchMedia("(prefers-color-scheme: light)");
        var onMq = function () { if (state.preset === "auto") { apply(); notify(); } };
        if (mq.addEventListener) mq.addEventListener("change", onMq);
        else if (mq.addListener) mq.addListener(onMq);
    }
    // Синхронизация между вкладками
    global.addEventListener("storage", function (e) {
        if (e.key !== KEY) return;
        load();
        apply();
        notify();
    });
})(window);
