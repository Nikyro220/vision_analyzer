/* ============================================================
   Vision Triage — сворачиваемая боковая панель.

   Подключается в <head> синхронно: состояние (data-sidebar="collapsed" на <html>)
   выставляется до первой отрисовки, поэтому панель не «прыгает» при переходах
   между страницами. Хранится только локально в браузере (localStorage).
   На узких экранах (<= 880px, панель — верхняя полоса) сворачивание не действует,
   см. @media в style.css.
   ============================================================ */
(function () {
    "use strict";

    var KEY = "vt-sidebar-v1";
    var root = document.documentElement;

    function read() {
        try { return localStorage.getItem(KEY) === "collapsed"; } catch (e) { return false; }
    }
    function write(collapsed) {
        try { localStorage.setItem(KEY, collapsed ? "collapsed" : "expanded"); } catch (e) { /* приватный режим и т.п. */ }
    }
    function isCollapsed() {
        return root.getAttribute("data-sidebar") === "collapsed";
    }
    function apply(collapsed) {
        if (collapsed) root.setAttribute("data-sidebar", "collapsed");
        else root.removeAttribute("data-sidebar");
    }

    // Без мигания: до отрисовки.
    apply(read());

    // Подписи в свёрнутом виде — через title (текст пунктов при этом скрыт визуально);
    // в развёрнутом title не нужен, он только мешал бы всплывающими подсказками.
    function sync(btn) {
        var collapsed = isCollapsed();
        var label = collapsed ? "Развернуть панель" : "Свернуть панель";
        btn.setAttribute("aria-expanded", collapsed ? "false" : "true");
        btn.setAttribute("aria-label", label);
        btn.setAttribute("title", label);
        var links = document.querySelectorAll(".sidebar .nav-link");
        for (var i = 0; i < links.length; i++) {
            var text = links[i].querySelector(".nav-label");
            if (!text) continue;
            if (collapsed) links[i].setAttribute("title", text.textContent.trim());
            else links[i].removeAttribute("title");
        }
    }

    document.addEventListener("DOMContentLoaded", function () {
        var btn = document.getElementById("sidebar-toggle");
        if (!btn) return;
        sync(btn);
        btn.addEventListener("click", function () {
            var next = !isCollapsed();
            apply(next);
            write(next);
            sync(btn);
        });
        // Анимацию ширины включаем только после первого кадра — иначе при загрузке
        // со свёрнутой панелью она «выезжала» бы с 240px.
        requestAnimationFrame(function () {
            requestAnimationFrame(function () { root.classList.add("sb-ready"); });
        });
    });

    // Несколько вкладок: свернули в одной — остальные подхватывают.
    window.addEventListener("storage", function (e) {
        if (e.key !== KEY) return;
        apply(e.newValue === "collapsed");
        var btn = document.getElementById("sidebar-toggle");
        if (btn) sync(btn);
    });
})();
