/* ============================================================
   Vision Triage — интерфейс вкладки «Визуал».
   Полностью строится из реестра VTheme.modules (см. theme.js), поэтому
   новые настройки появляются здесь автоматически.
   ============================================================ */
(function () {
    "use strict";
    if (!window.VTheme) return;
    var T = window.VTheme;

    var ACCENT_SWATCHES = [
        ["#3fa7a0", "Бирюзовый"], ["#4c8dff", "Синий"], ["#8b7bff", "Фиолетовый"],
        ["#e0609a", "Розовый"], ["#e5534b", "Красный"], ["#e08a3c", "Оранжевый"],
        ["#d9c04a", "Жёлтый"], ["#4fae70", "Зелёный"]
    ];
    var SAMPLE_TEXT = "Съешь ещё этих мягких французских булок, да выпей чаю 0123456789";

    // ------------------------------------------------------------------ helpers
    function el(tag, opts) {
        var node = document.createElement(tag);
        opts = opts || {};
        if (opts.cls) node.className = opts.cls;
        if (opts.text != null) node.textContent = opts.text;
        if (opts.attrs) Object.keys(opts.attrs).forEach(function (k) { node.setAttribute(k, opts.attrs[k]); });
        if (opts.on) Object.keys(opts.on).forEach(function (k) { node.addEventListener(k, opts.on[k]); });
        (opts.kids || []).forEach(function (c) { if (c) node.appendChild(c); });
        return node;
    }
    function button(text, cls, onClick, title) {
        return el("button", { cls: cls, text: text, attrs: { type: "button", title: title || "" }, on: { click: onClick } });
    }
    var uid = 0;
    function nextId(p) { uid += 1; return "vs-" + p + "-" + uid; }

    var syncers = [];       // функции, обновляющие контролы по состоянию
    var statusEl;

    function say(text, kind) {
        if (!statusEl) return;
        statusEl.textContent = text || "";
        statusEl.className = "vs-status" + (kind ? " vs-status-" + kind : "");
    }

    // ------------------------------------------------------- индикатор «сбросить»
    function makeIndicator(key, autoLabel) {
        var box = el("span", { cls: "default-value-indicator" });
        syncers.push(function () {
            box.textContent = "";
            if (T.isOverridden(key)) {
                box.appendChild(button("сбросить", "tag tag-accent tag-reset-btn", function () { T.resetValue(key); }));
            } else {
                box.appendChild(el("span", { cls: "tag", text: autoLabel || "по умолчанию" }));
            }
        });
        return box;
    }

    // ------------------------------------------------------------------- поля
    function colorControl(f) {
        var id = nextId(f.key);
        var picker = el("input", { cls: "vs-color", attrs: { type: "color", id: id, "aria-label": f.label } });
        var text = el("input", { cls: "input input-mono vs-hex", attrs: { type: "text", maxlength: "7", spellcheck: "false", "aria-label": f.label + " (HEX)" } });
        picker.addEventListener("input", function () { T.setValue(f.key, picker.value); });
        text.addEventListener("change", function () {
            var v = text.value.trim();
            if (/^#?[0-9a-f]{3}$/i.test(v)) v = v.replace("#", "").replace(/./g, "$&$&");
            v = v.replace("#", "");
            if (/^[0-9a-f]{6}$/i.test(v)) T.setValue(f.key, "#" + v);
            else sync();
        });
        function sync() {
            var rgb = T.color.parse(T.effective(f.key));
            var hex = rgb ? T.color.toHex(rgb) : "#000000";
            if (picker.value !== hex) picker.value = hex;
            if (document.activeElement !== text) text.value = hex;
        }
        syncers.push(sync);
        var kids = [el("div", { cls: "vs-color-row", kids: [picker, text] })];
        if (f.swatches) kids.push(swatchRow(f));
        return { id: id, node: el("div", { cls: "vs-control-stack", kids: kids }) };
    }

    function swatchRow(f) {
        var row = el("div", { cls: "vs-swatches", attrs: { role: "group", "aria-label": "Быстрый выбор акцента" } });
        var buttons = ACCENT_SWATCHES.map(function (s) {
            var b = el("button", { cls: "vs-swatch", attrs: { type: "button", title: s[1], "aria-label": s[1] } });
            b.style.background = s[0];
            b.addEventListener("click", function () {
                // Цвет, совпадающий с акцентом по умолчанию (темы — собственным, иначе бирюзовым), — это сброс
                if (s[0] === String(T.defaultOf(f.key) || "").toLowerCase()) T.resetValue(f.key); else T.setValue(f.key, s[0]);
            });
            row.appendChild(b);
            return [b, s[0]];
        });
        var rnd = button("Случайный", "tag tag-reset-btn vs-random", function () {
            // Средняя яркость, чтобы акцент читался на любой теме
            var h = Math.floor(Math.random() * 360);
            var c = hslToHex(h, 60 + Math.random() * 25, 50 + Math.random() * 8);
            T.setValue(f.key, c);
        });
        row.appendChild(rnd);
        syncers.push(function () {
            var cur = String(T.effective(f.key) || "").toLowerCase();
            buttons.forEach(function (p) { p[0].classList.toggle("is-active", p[1] === cur); });
        });
        return row;
    }
    function hslToHex(h, s, l) {
        s /= 100; l /= 100;
        var k = function (n) { return (n + h / 30) % 12; };
        var a = s * Math.min(l, 1 - l);
        var f = function (n) { return l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1))); };
        return T.color.toHex([f(0) * 255, f(8) * 255, f(4) * 255]);
    }

    function rangeControl(f) {
        var id = nextId(f.key);
        var input = el("input", { cls: "vs-range", attrs: { type: "range", id: id, min: f.min, max: f.max, step: f.step } });
        var out = el("output", { cls: "vs-range-out", attrs: { for: id } });
        var dragging = false;
        if (f.deferred) {
            // Во время перетаскивания меняем только подпись значения; в тему пишем по отпусканию.
            // Так раскладка не перестраивается под курсором.
            input.addEventListener("input", function () {
                dragging = true;
                out.textContent = input.value + (f.unit || "");
            });
            input.addEventListener("change", function () {
                dragging = false;
                T.setValue(f.key, input.value);
            });
        } else {
            input.addEventListener("input", function () { T.setValue(f.key, input.value); });
        }
        var fitEl = f.fit ? el("p", { cls: "field-hint vs-fit" }) : null;
        // Сколько места реально есть под контент в этом окне (ширина окна минус боковая панель)
        function available() {
            var sb = document.querySelector(".sidebar");
            var win = document.documentElement.clientWidth;
            var side = sb && sb.offsetWidth < win - 1 ? sb.offsetWidth : 0;
            return win - side;
        }
        function syncFit() {
            if (!fitEl) return;
            var avail = available(), cur = Number(input.value);
            var full = f.disabledWhen && T.effective(f.disabledWhen);
            fitEl.classList.toggle("is-over", !full && cur > avail);
            fitEl.textContent = "Сейчас окно вмещает до " + avail + " px."
                + (full ? " Включено «на всю ширину» — ползунок не действует."
                    : cur > avail ? " Выбрано больше — фактически ширину ограничивает окно; расширьте его или включите «Контент на всю ширину окна»." : "");
        }
        if (fitEl) window.addEventListener("resize", syncFit);
        syncers.push(function () {
            if (dragging) return;
            var v = T.effective(f.key);
            if (String(input.value) !== String(v)) input.value = v;
            out.textContent = v + (f.unit || "");
            if (f.disabledWhen) input.disabled = !!T.effective(f.disabledWhen);
            syncFit();
        });
        if (fitEl) input.addEventListener("input", syncFit);
        return { id: id, node: el("div", { cls: "vs-control-stack", kids: [el("div", { cls: "vs-range-row", kids: [input, out] }), fitEl] }) };
    }

    function toggleControl(f) {
        var id = nextId(f.key);
        var input = el("input", { attrs: { type: "checkbox", id: id } });
        input.addEventListener("change", function () { T.setValue(f.key, input.checked); });
        syncers.push(function () { input.checked = !!T.effective(f.key); });
        return { id: id, node: el("label", { cls: "checkbox-label", kids: [input, el("span", { text: "включено" })] }) };
    }

    function fontControl(f) {
        var id = nextId(f.key);
        var select = el("select", { cls: "input", attrs: { id: id } });
        f.list.forEach(function (o) { select.appendChild(el("option", { text: o.label, attrs: { value: o.id } })); });
        var custom = el("input", { cls: "input", attrs: { type: "text", maxlength: "60", placeholder: "Например: Inter, PT Sans", spellcheck: "false", "aria-label": f.label + " — название шрифта" } });
        var sample = el("div", { cls: "vs-font-sample", text: SAMPLE_TEXT });
        if (f.key === "fontMono") sample.style.fontFamily = "var(--font-mono)"; else sample.style.fontFamily = "var(--font-body)";

        select.addEventListener("change", function () {
            if (select.value === f.defaultId) T.resetValue(f.key); else T.setValue(f.key, select.value);
        });
        custom.addEventListener("change", function () { T.setValue(f.customKey, custom.value); });
        syncers.push(function () {
            var cur = T.effective(f.key);
            if (select.value !== cur) select.value = cur;
            var isCustom = cur === "custom";
            custom.style.display = isCustom ? "" : "none";
            if (document.activeElement !== custom) custom.value = T.get().values[f.customKey] || "";
        });
        return { id: id, node: el("div", { cls: "vs-control-stack", kids: [select, custom, sample] }) };
    }

    function buildField(f) {
        var ctl = f.kind === "color" ? colorControl(f)
            : f.kind === "range" ? rangeControl(f)
            : f.kind === "toggle" ? toggleControl(f)
            : f.kind === "font" ? fontControl(f) : null;
        if (!ctl) return null;
        var indicatorKey = f.key;
        var label = el("div", { cls: "settings-row-label", kids: [
            el("label", { text: f.label, attrs: { for: ctl.id } }),
            makeIndicator(indicatorKey, f.key === "accentStrong" ? "авто" : null),
            f.hint ? el("p", { cls: "field-hint", text: f.hint }) : null
        ] });
        return el("div", { cls: "settings-row vs-row", kids: [label, el("div", { cls: "settings-row-control", kids: [ctl.node] })] });
    }

    // ---------------------------------------------------------------- контраст
    function contrastTable(pairs) {
        var box = el("div", { cls: "vs-contrast" });
        syncers.push(function () {
            box.textContent = "";
            pairs.forEach(function (p) {
                var fg = T.color.parse(T.effective(p.fg)), bg = T.color.parse(T.effective(p.bg));
                if (!fg || !bg) return;
                var ratio = T.color.contrast(fg, bg);
                var cls = ratio >= 4.5 ? "tag tag-ok" : ratio >= 3 ? "tag tag-warn" : "tag tag-bad";
                box.appendChild(el("div", { cls: "vs-contrast-item", kids: [
                    el("span", { text: p.label }),
                    el("span", { cls: cls, text: ratio.toFixed(1) + (ratio >= 4.5 ? " ✓" : ratio >= 3 ? " мало" : " плохо") })
                ] }));
            });
        });
        return box;
    }

    // ------------------------------------------------------------------ модули
    function buildModule(m) {
        var head = el("div", { cls: "vs-module-head", kids: [el("h2", { cls: "card-title", text: m.title })] });
        var resetBtn = button("сбросить раздел", "tag tag-reset-btn", function () { T.resetModule(m.id); });
        head.appendChild(resetBtn);
        syncers.push(function () {
            var any = m.fields.some(function (f) {
                return T.isOverridden(f.key) || (f.customKey && T.isOverridden(f.customKey));
            });
            resetBtn.style.visibility = any ? "visible" : "hidden";
        });
        var kids = [head];
        if (m.hint) kids.push(el("p", { cls: "field-hint vs-module-hint", text: m.hint }));
        m.fields.forEach(function (f) { kids.push(buildField(f)); });
        if (m.contrastPairs) kids.push(contrastTable(m.contrastPairs));
        return el("section", { cls: "card vs-module", attrs: { "data-module": m.id }, kids: kids });
    }

    // ------------------------------------------------------------------ пресеты
    function mock(themeId) {
        var box = el("div", { cls: "vs-mock", attrs: { "data-theme": themeId } });
        box.appendChild(el("span", { cls: "vs-mock-side" }));
        box.appendChild(el("span", { cls: "vs-mock-body", kids: [
            el("span", { cls: "vs-mock-line" }),
            el("span", { cls: "vs-mock-line vs-mock-line-short" }),
            el("span", { cls: "vs-mock-btn" })
        ] }));
        return box;
    }
    function presetPreview(p) {
        var thumb = p.id === "auto"
            ? el("div", { cls: "vs-mock-split", kids: [mock("dark"), mock("light")] })
            : mock(p.id);
        return [
            thumb,
            el("span", { cls: "vs-preset-name", text: p.label }),
            el("span", { cls: "vs-preset-hint", text: p.hint })
        ];
    }
    // Выбор темы: выпадающий список с группами (<optgroup>, порядок задаёт сервер) и карточка-превью
    // выбранной темы. Список из ~100 пунктов листается стрелками и ищется набором первых букв.
    function buildPresets() {
        var id = nextId("preset");
        var select = el("select", { cls: "input vs-theme-select", attrs: { id: id } });
        var preview = el("div", { cls: "vs-theme-preview", attrs: { "aria-live": "polite" } });
        var byId = {};
        var group = null, groupId = null;
        T.presets.forEach(function (p) {
            byId[p.id] = p;
            var opt = el("option", { text: p.label, attrs: { value: p.id } });
            if (!p.group) { select.appendChild(opt); return; }          // «Авто» — вне групп
            if (p.group !== groupId) {
                group = el("optgroup", { attrs: { label: p.group } });
                groupId = p.group;
                select.appendChild(group);
            }
            group.appendChild(opt);
        });
        select.addEventListener("change", function () {
            var had = T.hasPaletteOverrides();
            T.setPreset(select.value);
            say(had ? "Тема применена. Свои цвета поверхностей, текста и статусов сброшены — «Отменить» вернёт их." : "Тема применена.", "ok");
        });
        syncers.push(function () {
            var cur = T.get().preset;
            select.value = cur;
            preview.textContent = "";
            (byId[cur] ? presetPreview(byId[cur]) : []).forEach(function (n) { preview.appendChild(n); });
        });
        return el("section", { cls: "card vs-module", kids: [
            el("h2", { cls: "card-title", text: "Тема" }),
            el("p", { cls: "field-hint vs-module-hint", text: "Пресет меняет палитру, а у части тем — ещё и акцентный цвет. Шрифты и форма остаются вашими. Темы сгруппированы: тёмные, светлые и контрастные; внутри групп — по алфавиту. В открытом списке можно набрать первые буквы названия." }),
            el("div", { cls: "vs-theme-picker", kids: [
                el("div", { cls: "vs-control-stack vs-theme-select-wrap", kids: [
                    el("label", { cls: "vs-theme-label", text: "Тема оформления", attrs: { for: id } }),
                    select
                ] }),
                preview
            ] })
        ] });
    }

    // -------------------------------------------------------------- предпросмотр
    function buildPreview() {
        var sample = el("div", { cls: "vs-preview" });
        sample.innerHTML =
            '<h3>Заголовок раздела</h3>' +
            '<p>Обычный текст интерфейса. <span class="vs-muted">Приглушённый текст.</span> <span class="vs-dim">Слабый текст.</span> <a href="#" onclick="return false">Ссылка</a></p>' +
            '<div class="vs-preview-row"><span class="btn btn-primary">Основная</span><span class="btn btn-secondary">Вторичная</span><span class="btn btn-danger">Удалить</span></div>' +
            '<div class="vs-preview-row"><span class="risk-pill risk-pill-low">низкий</span><span class="risk-pill risk-pill-medium">средний</span><span class="risk-pill risk-pill-high">высокий</span>' +
            '<span class="tag tag-ok">ok</span><span class="tag tag-warn">warn</span><span class="tag tag-accent">accent</span></div>' +
            '<div class="vs-preview-row"><span class="role-badge role-user">Пользователь</span><span class="role-badge role-admin">Админ</span><span class="role-badge role-head_admin">Главный админ</span></div>' +
            '<div class="vs-preview-row"><input type="text" class="input" value="Поле ввода" tabindex="-1" aria-label="Пример поля ввода"></div>';
        return el("section", { cls: "card vs-module", kids: [el("h2", { cls: "card-title", text: "Предпросмотр" }), sample] });
    }

    // ------------------------------------------------------------------ toolbar
    function buildToolbar() {
        statusEl = el("span", { cls: "vs-status", attrs: { role: "status", "aria-live": "polite" } });
        var undoBtn = button("↶ Отменить", "btn btn-secondary vs-btn", function () { T.undo(); say("Изменение отменено.", "ok"); });
        var fileInput = el("input", { attrs: { type: "file", accept: "application/json,.json", hidden: "" } });
        fileInput.addEventListener("change", function () {
            var file = fileInput.files && fileInput.files[0];
            if (!file) return;
            var reader = new FileReader();
            reader.onload = function () {
                try { T.importJSON(String(reader.result)); say("Тема импортирована.", "ok"); }
                catch (e) { say(e.message, "error"); }
            };
            reader.readAsText(file);
            fileInput.value = "";
        });
        syncers.push(function () { undoBtn.disabled = !T.canUndo(); });

        return el("div", { cls: "card vs-toolbar", kids: [
            el("div", { cls: "vs-toolbar-buttons", kids: [
                undoBtn,
                button("Экспорт", "btn btn-secondary vs-btn", function () {
                    var blob = new Blob([T.exportJSON()], { type: "application/json" });
                    var a = document.createElement("a");
                    a.href = URL.createObjectURL(blob);
                    a.download = "vision-triage-theme.json";
                    document.body.appendChild(a);
                    a.click();
                    a.remove();
                    setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
                    say("Тема сохранена в файл.", "ok");
                }),
                button("Импорт", "btn btn-secondary vs-btn", function () { fileInput.click(); }),
                fileInput,
                button("Сбросить всё", "btn btn-danger vs-btn", function () {
                    T.resetAll();
                    say("Оформление сброшено к стандартному.", "ok");
                })
            ] }),
            el("p", { cls: "field-hint vs-toolbar-note", text: "Оформление хранится только в этом браузере и применяется автоматически. Другие пользователи и устройства его не видят." }),
            statusEl
        ] });
    }

    // -------------------------------------------------------------------- init
    function init() {
        var host = document.getElementById("visual-settings");
        if (!host || host.dataset.ready) return;
        host.dataset.ready = "1";
        host.appendChild(buildToolbar());
        host.appendChild(buildPresets());
        host.appendChild(buildPreview());
        T.modules.forEach(function (m) { host.appendChild(buildModule(m)); });

        function syncAll() { syncers.forEach(function (fn) { fn(); }); }
        T.onChange(syncAll);
        syncAll();
    }
    if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
    else init();
})();