/* Редактор аватара на странице профиля.
 *
 * Обычное фото редактируется целиком в браузере: картинка рисуется на canvas 256x256 с
 * преобразованием (сдвиг -> поворот -> масштаб с независимыми X/Y), затем попиксельно применяются
 * оттенок, насыщенность, яркость и контраст. Цвет считаем сами, а не через ctx.filter: он не везде
 * поддерживается (Safari). Готовый PNG уходит на сервер, который проверяет и перекодирует его в WebP
 * (см. avatars.py).
 *
 * Анимированный файл (GIF, анимированный WebP, APNG) на canvas превратился бы в один кадр. Поэтому для
 * него предпросмотр — живой <img>, который двигается CSS-преобразованием и CSS-фильтром (анимацию играет
 * сам браузер), а на сервер уходит ИСХОДНЫЙ файл + параметры; кадры обрабатывает сервер
 * (avatars.process_animated) по тем же формулам.
 *
 * Каждая настройка — ползунок + числовое поле + кнопка сброса; позиция (X/Y) — в пикселях холста 256x256.
 */
(function () {
    "use strict";

    var card = document.getElementById("avatar-card");
    var fileInput = document.getElementById("avatar-file");
    var editor = document.getElementById("avatar-editor");
    var canvas = document.getElementById("avatar-canvas");
    var stage = document.getElementById("avatar-stage");
    var animBox = document.getElementById("avatar-anim");
    var animImg = document.getElementById("avatar-anim-img");
    var animNote = document.getElementById("avatar-anim-note");
    var pickBtn = document.getElementById("avatar-pick");
    if (!card || !fileInput || !editor || !canvas || !stage) return;

    var ctx = canvas.getContext("2d", { willReadFrequently: true });
    var S = canvas.width;
    var MAX_FILE = 25 * 1024 * 1024; // сам файл может быть большим: на сервер уйдёт только готовый 256x256
    var MAX_ANIMATED = parseInt(card.dataset.maxAnimated, 10) || 8 * 1024 * 1024;
    var errorEl = document.getElementById("avatar-error");

    var DEFAULTS = { zoom: 1, sx: 1, sy: 1, rot: 0, hue: 0, sat: 1, bri: 1, con: 1, dx: 0, dy: 0 };
    var img = null, objectUrl = null, baseScale = 1;
    var sourceFile = null, animated = false;
    var st = null; // текущее состояние: DEFAULTS + flip
    var scheduled = false, busy = false;

    // ---------------- Поля настроек ----------------

    var ctls = {};
    Array.prototype.forEach.call(editor.querySelectorAll(".avatar-ctl[data-k]"), function (box) {
        var k = box.dataset.k;
        ctls[k] = {
            box: box,
            factor: parseFloat(box.dataset.factor) || 1,
            range: box.querySelector("input[type=range]"),
            num: box.querySelector("input[type=number]")
        };
    });

    function resetState() {
        st = Object.assign({}, DEFAULTS, { flip: 1 });
        syncControls();
    }

    function shown(k) { return Math.round(st[k] * ctls[k].factor); }

    function syncControls(except) {
        Object.keys(ctls).forEach(function (k) {
            var c = ctls[k], v = shown(k);
            c.range.value = v;
            if (c.num !== except) c.num.value = v;
            c.box.classList.toggle("is-default", v === Math.round(DEFAULTS[k] * c.factor));
        });
    }

    function showError(msg) {
        errorEl.textContent = msg || "";
        errorEl.hidden = !msg;
    }

    function setFromField(k, raw, field) {
        var c = ctls[k], v = parseFloat(raw);
        if (!isFinite(v)) return false;
        v = Math.min(parseFloat(c.range.max), Math.max(parseFloat(c.range.min), v));
        st[k] = v / c.factor;
        syncControls(field);
        schedule();
        return true;
    }

    Object.keys(ctls).forEach(function (k) {
        var c = ctls[k];
        c.range.addEventListener("input", function () { if (st) setFromField(k, c.range.value, null); });
        c.num.addEventListener("input", function () { if (st) setFromField(k, c.num.value, c.num); });
        c.num.addEventListener("change", function () { if (st) syncControls(); }); // вернуть корректное число в поле
    });

    // ---------------- Рендер ----------------

    function schedule() {
        if (scheduled) return;
        scheduled = true;
        requestAnimationFrame(function () { scheduled = false; render(); });
    }

    function render() {
        if (!img) return;
        if (animated) renderAnimated(); else renderCanvas();
    }

    function renderCanvas() {
        ctx.clearRect(0, 0, S, S);
        ctx.save();
        ctx.translate(S / 2 + st.dx, S / 2 + st.dy);
        ctx.rotate(st.rot * Math.PI / 180);
        // масштаб применяется в осях самой картинки, поэтому растяжение «едет» вместе с поворотом
        var k = baseScale * st.zoom;
        ctx.scale(k * st.sx * st.flip, k * st.sy);
        ctx.drawImage(img, -img.naturalWidth / 2, -img.naturalHeight / 2);
        ctx.restore();
        adjustColors();
    }

    function renderAnimated() {
        var ratio = stage.clientWidth / S; // CSS-пиксели на пиксель холста
        var k = baseScale * st.zoom * ratio;
        // translate(-50%,-50%) ставит центр <img> в центр кадра; дальше — те же сдвиг/поворот/масштаб, что на canvas
        animImg.style.transform = "translate(-50%, -50%) translate(" + (st.dx * ratio) + "px, " + (st.dy * ratio) + "px) " +
            "rotate(" + st.rot + "deg) scale(" + (k * st.sx * st.flip) + ", " + (k * st.sy) + ")";
        var plain = st.hue === 0 && st.sat === 1 && st.bri === 1 && st.con === 1;
        animImg.style.filter = plain ? "" :
            "hue-rotate(" + st.hue + "deg) saturate(" + st.sat + ") brightness(" + st.bri + ") contrast(" + st.con + ")";
    }

    function adjustColors() {
        var plain = st.hue === 0 && st.sat === 1 && st.bri === 1 && st.con === 1;
        if (plain) return;

        // Матрица оттенка/насыщенности — та же, что в спецификации CSS-фильтров.
        var a = st.hue * Math.PI / 180, c = Math.cos(a), s = Math.sin(a), sat = st.sat;
        var h = [
            0.213 + c * 0.787 - s * 0.213, 0.715 - c * 0.715 - s * 0.715, 0.072 - c * 0.072 + s * 0.928,
            0.213 - c * 0.213 + s * 0.143, 0.715 + c * 0.285 + s * 0.140, 0.072 - c * 0.072 - s * 0.283,
            0.213 - c * 0.213 - s * 0.787, 0.715 - c * 0.715 + s * 0.715, 0.072 + c * 0.928 + s * 0.072
        ];
        var sm = [
            0.213 + 0.787 * sat, 0.715 - 0.715 * sat, 0.072 - 0.072 * sat,
            0.213 - 0.213 * sat, 0.715 + 0.285 * sat, 0.072 - 0.072 * sat,
            0.213 - 0.213 * sat, 0.715 - 0.715 * sat, 0.072 + 0.928 * sat
        ];
        var m = new Array(9);
        for (var r = 0; r < 3; r++) {
            for (var q = 0; q < 3; q++) {
                m[r * 3 + q] = sm[r * 3] * h[q] + sm[r * 3 + 1] * h[3 + q] + sm[r * 3 + 2] * h[6 + q];
            }
        }

        var data = ctx.getImageData(0, 0, S, S), d = data.data;
        var bri = st.bri, con = st.con;
        for (var i = 0; i < d.length; i += 4) {
            if (d[i + 3] === 0) continue;
            var R = d[i], G = d[i + 1], B = d[i + 2];
            var nr = (m[0] * R + m[1] * G + m[2] * B) * bri;
            var ng = (m[3] * R + m[4] * G + m[5] * B) * bri;
            var nb = (m[6] * R + m[7] * G + m[8] * B) * bri;
            d[i] = (nr - 128) * con + 128;     // Uint8ClampedArray сам обрежет до 0..255
            d[i + 1] = (ng - 128) * con + 128;
            d[i + 2] = (nb - 128) * con + 128;
        }
        ctx.putImageData(data, 0, 0);
    }

    // ---------------- Определение анимации по байтам файла ----------------

    function ascii(b, from, to) { return String.fromCharCode.apply(null, Array.prototype.slice.call(b, from, to)); }

    function skipBlocks(b, p) {
        while (p < b.length) { var size = b[p++]; if (!size) break; p += size; }
        return p;
    }

    function gifFrames(b) {
        if (b.length < 13) return 0;
        var p = 13;
        if (b[10] & 0x80) p += 3 * (1 << ((b[10] & 7) + 1)); // глобальная палитра
        var n = 0;
        while (p < b.length) {
            var t = b[p++];
            if (t === 0x3B) break;                            // конец файла
            if (t === 0x21) { p = skipBlocks(b, p + 1); }     // расширение (в т.ч. задержка кадра)
            else if (t === 0x2C) {                            // кадр
                if (++n > 1) return n;
                var flags = b[p + 8];
                p += 9;
                if (flags & 0x80) p += 3 * (1 << ((flags & 7) + 1)); // локальная палитра
                p = skipBlocks(b, p + 1);
            } else break;
        }
        return n;
    }

    function isAnimated(buffer) {
        var b = new Uint8Array(buffer);
        if (b.length > 21 && ascii(b, 0, 4) === "RIFF" && ascii(b, 8, 12) === "WEBP") {
            return ascii(b, 12, 16) === "VP8X" && (b[20] & 0x02) !== 0; // бит «есть анимация»
        }
        if (b.length > 8 && b[0] === 0x89 && ascii(b, 1, 4) === "PNG") {  // APNG: чанк acTL до первого IDAT
            var p = 8;
            while (p + 8 <= b.length) {
                var len = ((b[p] << 24) | (b[p + 1] << 16) | (b[p + 2] << 8) | b[p + 3]) >>> 0;
                var type = ascii(b, p + 4, p + 8);
                if (type === "acTL") return true;
                if (type === "IDAT" || type === "IEND") return false;
                p += 12 + len;
            }
            return false;
        }
        if (b.length > 6 && ascii(b, 0, 3) === "GIF") return gifFrames(b) > 1;
        return false;
    }

    function readBuffer(file, done) {
        var reader = new FileReader();
        reader.onload = function () { done(reader.result); };
        reader.onerror = function () { done(null); };
        reader.readAsArrayBuffer(file);
    }

    // ---------------- Загрузка файла ----------------

    function open(file) {
        showError("");
        if (!file) return;
        if (file.size > MAX_FILE) { alert("Файл слишком большой."); return; }
        readBuffer(file, function (buffer) {
            var anim = buffer ? isAnimated(buffer) : false;
            if (anim && file.size > MAX_ANIMATED) {
                alert("Анимация слишком большая: максимум " + Math.floor(MAX_ANIMATED / 1048576) + " МБ.");
                fileInput.value = "";
                return;
            }
            if (objectUrl) URL.revokeObjectURL(objectUrl);
            objectUrl = URL.createObjectURL(file);
            var next = new Image();
            next.onload = function () {
                img = next;
                sourceFile = file;
                animated = anim;
                baseScale = S / Math.min(img.naturalWidth, img.naturalHeight); // «cover»: кадр заполнен
                resetState();
                animBox.hidden = !animated;
                animNote.hidden = !animated;
                if (animated) {
                    animImg.src = objectUrl;
                    animImg.style.width = img.naturalWidth + "px";
                    animImg.style.height = img.naturalHeight + "px";
                    ctx.clearRect(0, 0, S, S);
                } else {
                    animImg.removeAttribute("src");
                }
                editor.hidden = false;
                render();
                editor.scrollIntoView({ block: "nearest", behavior: "smooth" });
            };
            next.onerror = function () {
                alert("Не удалось открыть изображение. Попробуйте PNG, JPEG, WebP или GIF.");
            };
            next.src = objectUrl;
        });
    }

    function close() {
        editor.hidden = true;
        img = null; sourceFile = null; animated = false;
        animBox.hidden = true;
        animImg.removeAttribute("src");
        fileInput.value = "";
        showError("");
        if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
    }

    fileInput.addEventListener("change", function () { open(fileInput.files && fileInput.files[0]); });
    if (pickBtn) pickBtn.addEventListener("click", function () { fileInput.click(); }); // клик по кружку

    // ---------------- Управление ----------------

    // сдвиг перетаскиванием
    var drag = null;
    stage.addEventListener("pointerdown", function (e) {
        if (!img) return;
        drag = { x: e.clientX, y: e.clientY, dx: st.dx, dy: st.dy };
        stage.setPointerCapture(e.pointerId);
        stage.classList.add("is-dragging");
    });
    stage.addEventListener("pointermove", function (e) {
        if (!drag) return;
        var k = S / stage.getBoundingClientRect().width; // CSS-пиксели -> пиксели холста
        st.dx = Math.max(-S, Math.min(S, drag.dx + (e.clientX - drag.x) * k));
        st.dy = Math.max(-S, Math.min(S, drag.dy + (e.clientY - drag.y) * k));
        syncControls();
        schedule();
    });
    function endDrag() { drag = null; stage.classList.remove("is-dragging"); }
    stage.addEventListener("pointerup", endDrag);
    stage.addEventListener("pointercancel", endDrag);

    // масштаб колесом
    stage.addEventListener("wheel", function (e) {
        if (!img) return;
        e.preventDefault();
        st.zoom = Math.min(5, Math.max(0.2, st.zoom * Math.exp(-e.deltaY * 0.0015)));
        syncControls();
        schedule();
    }, { passive: false });

    window.addEventListener("resize", function () { if (img && animated) schedule(); });

    editor.addEventListener("click", function (e) {
        var one = e.target.closest("[data-reset]");
        if (one && img) {  // сброс одной настройки
            st[one.dataset.reset] = DEFAULTS[one.dataset.reset];
            syncControls(); schedule();
            return;
        }
        var btn = e.target.closest("[data-act]");
        if (!btn || !img) return;
        var act = btn.dataset.act;
        if (act === "rotate-left" || act === "rotate-right") {
            var r = st.rot + (act === "rotate-left" ? -90 : 90);
            st.rot = ((r + 180) % 360 + 360) % 360 - 180; // держим в диапазоне ползунка -180..180
            syncControls(); schedule();
        } else if (act === "flip") {
            st.flip = -st.flip; schedule();
        } else if (act === "reset") {
            resetState(); schedule();
        } else if (act === "cancel") {
            close();
        } else if (act === "save") {
            save(btn);
        }
    });

    // ---------------- Сохранение ----------------

    function send(btn, body) {
        body.append("csrf_token", card.dataset.csrf);
        fetch(card.dataset.uploadUrl, {
            method: "POST", body: body, credentials: "same-origin",
            headers: { "X-CSRFToken": card.dataset.csrf, "X-Requested-With": "fetch" }
        }).then(function (resp) {
            return resp.json().then(function (j) { return { ok: resp.ok, json: j }; });
        }).then(function (res) {
            if (res.ok && res.json.ok) { window.location.reload(); } // сообщение «Аватар обновлён» придёт из flash
            else { fail(btn, res.json.error || "Не удалось сохранить аватар."); }
        }).catch(function () {
            fail(btn, "Не удалось сохранить аватар. Обновите страницу и попробуйте снова.");
        });
    }

    function fail(btn, msg) { busy = false; btn.disabled = false; showError(msg); }

    function save(btn) {
        if (busy) return;
        busy = true; btn.disabled = true; showError("");
        var body = new FormData();
        if (animated) {
            // анимацию сервер собирает сам: ему нужен исходный файл и настройки
            body.append("avatar", sourceFile, sourceFile.name || "avatar");
            body.append("params", JSON.stringify({
                zoom: st.zoom, sx: st.sx, sy: st.sy, rot: st.rot, hue: st.hue, sat: st.sat,
                bri: st.bri, con: st.con, dx: st.dx, dy: st.dy, flip: st.flip
            }));
            send(btn, body);
            return;
        }
        render(); // на случай отложенного кадра
        canvas.toBlob(function (blob) {
            if (!blob) { fail(btn, "Не удалось подготовить изображение."); return; }
            body.append("avatar", blob, "avatar.png");
            send(btn, body);
        }, "image/png");
    }
})();
