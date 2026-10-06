/* Редактор аватара на странице профиля.
 *
 * Всё редактирование — в браузере: картинка рисуется на canvas 256x256 с преобразованием
 * (сдвиг -> поворот -> масштаб с независимыми X/Y), затем попиксельно применяются оттенок,
 * насыщенность, яркость и контраст. Цвет считаем сами, а не через ctx.filter: он не везде
 * поддерживается (Safari). Готовый PNG уходит на сервер, который проверяет и перекодирует его
 * в WebP (см. avatars.py).
 */
(function () {
    "use strict";

    var card = document.getElementById("avatar-card");
    var fileInput = document.getElementById("avatar-file");
    var editor = document.getElementById("avatar-editor");
    var canvas = document.getElementById("avatar-canvas");
    if (!card || !fileInput || !editor || !canvas) return;

    var ctx = canvas.getContext("2d", { willReadFrequently: true });
    var S = canvas.width;
    var MAX_FILE = 25 * 1024 * 1024; // сам файл может быть большим: на сервер уйдёт только готовый 256x256
    var errorEl = document.getElementById("avatar-error");
    var sliders = Array.prototype.slice.call(editor.querySelectorAll("input[type=range]"));

    var DEFAULTS = { zoom: 1, sx: 1, sy: 1, rot: 0, hue: 0, sat: 1, bri: 1, con: 1 };
    var img = null, objectUrl = null, baseScale = 1;
    var st = null; // текущее состояние: DEFAULTS + dx, dy, flip
    var scheduled = false, busy = false;

    function resetState() {
        st = Object.assign({}, DEFAULTS, { dx: 0, dy: 0, flip: 1 });
        syncSliders();
    }

    function fmt(k, v) {
        if (k === "rot" || k === "hue") return Math.round(v) + "°";
        return Math.round(v * 100) + "%";
    }

    function syncSliders() {
        sliders.forEach(function (el) {
            var k = el.dataset.k;
            el.value = st[k];
            el.previousElementSibling.textContent = fmt(k, st[k]);
        });
    }

    function showError(msg) {
        errorEl.textContent = msg || "";
        errorEl.hidden = !msg;
    }

    // ---------------- Рендер ----------------

    function schedule() {
        if (scheduled) return;
        scheduled = true;
        requestAnimationFrame(function () { scheduled = false; render(); });
    }

    function render() {
        if (!img) return;
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

    // ---------------- Загрузка файла ----------------

    function open(file) {
        showError("");
        if (!file) return;
        if (file.size > MAX_FILE) { alert("Файл слишком большой."); return; }
        if (objectUrl) URL.revokeObjectURL(objectUrl);
        objectUrl = URL.createObjectURL(file);
        var next = new Image();
        next.onload = function () {
            img = next;
            baseScale = S / Math.min(img.naturalWidth, img.naturalHeight); // «cover»: кадр заполнен
            resetState();
            editor.hidden = false;
            render();
            editor.scrollIntoView({ block: "nearest", behavior: "smooth" });
        };
        next.onerror = function () {
            alert("Не удалось открыть изображение. Попробуйте PNG, JPEG или WebP.");
        };
        next.src = objectUrl;
    }

    function close() {
        editor.hidden = true;
        img = null;
        fileInput.value = "";
        showError("");
        if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
    }

    fileInput.addEventListener("change", function () { open(fileInput.files && fileInput.files[0]); });

    // ---------------- Управление ----------------

    sliders.forEach(function (el) {
        el.addEventListener("input", function () {
            var k = el.dataset.k;
            st[k] = parseFloat(el.value);
            el.previousElementSibling.textContent = fmt(k, st[k]);
            schedule();
        });
    });

    // сдвиг перетаскиванием
    var drag = null;
    canvas.style.touchAction = "none";
    canvas.addEventListener("pointerdown", function (e) {
        if (!img) return;
        drag = { x: e.clientX, y: e.clientY, dx: st.dx, dy: st.dy };
        canvas.setPointerCapture(e.pointerId);
        canvas.classList.add("is-dragging");
    });
    canvas.addEventListener("pointermove", function (e) {
        if (!drag) return;
        var k = S / canvas.getBoundingClientRect().width; // CSS-пиксели -> пиксели холста
        st.dx = drag.dx + (e.clientX - drag.x) * k;
        st.dy = drag.dy + (e.clientY - drag.y) * k;
        schedule();
    });
    function endDrag() { drag = null; canvas.classList.remove("is-dragging"); }
    canvas.addEventListener("pointerup", endDrag);
    canvas.addEventListener("pointercancel", endDrag);

    // масштаб колесом
    canvas.addEventListener("wheel", function (e) {
        if (!img) return;
        e.preventDefault();
        st.zoom = Math.min(5, Math.max(0.2, st.zoom * Math.exp(-e.deltaY * 0.0015)));
        syncSliders();
        schedule();
    }, { passive: false });

    editor.addEventListener("click", function (e) {
        var btn = e.target.closest("[data-act]");
        if (!btn || !img) return;
        var act = btn.dataset.act;
        if (act === "rotate-left" || act === "rotate-right") {
            var r = st.rot + (act === "rotate-left" ? -90 : 90);
            st.rot = ((r + 180) % 360 + 360) % 360 - 180; // держим в диапазоне ползунка -180..180
            syncSliders(); schedule();
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

    function save(btn) {
        if (busy) return;
        render(); // на случай отложенного кадра
        busy = true; btn.disabled = true; showError("");
        canvas.toBlob(function (blob) {
            if (!blob) { fail("Не удалось подготовить изображение."); return; }
            var body = new FormData();
            body.append("avatar", blob, "avatar.png");
            body.append("csrf_token", card.dataset.csrf);
            fetch(card.dataset.uploadUrl, {
                method: "POST", body: body, credentials: "same-origin",
                headers: { "X-CSRFToken": card.dataset.csrf, "X-Requested-With": "fetch" }
            }).then(function (resp) {
                return resp.json().then(function (j) { return { ok: resp.ok, json: j }; });
            }).then(function (res) {
                if (res.ok && res.json.ok) { window.location.reload(); } // сообщение «Аватар обновлён» придёт из flash
                else { fail(res.json.error || "Не удалось сохранить аватар."); }
            }).catch(function () {
                fail("Не удалось сохранить аватар. Обновите страницу и попробуйте снова.");
            });
        }, "image/png");

        function fail(msg) { busy = false; btn.disabled = false; showError(msg); }
    }
})();
