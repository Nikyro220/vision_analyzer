/* Автоприменяемая фильтрация (поиск/риск/проверка) для истории и «Все анализы».
 * Подгружает только фрагмент таблицы+пагинации через fetch, без перезагрузки страницы.
 * Выделенные чекбоксы («ids») запоминаются в sessionStorage по пути страницы и переживают
 * смену фильтра и переход между страницами пагинации — так можно отметить один элемент,
 * отфильтровать, отметить другой, снять фильтр и увидеть оба выделенными. */
(function () {
  "use strict";

  document.querySelectorAll("[data-list-root]").forEach(initList);

  function initList(root) {
    var endpoint = root.dataset.endpoint;
    if (!endpoint) return;

    var form = document.querySelector('[data-filter-form]');
    var storageKey = "list-select:" + location.pathname;
    var selected = loadSelection();
    var state = { page: currentPageFromUrl() };
    var debounceTimer = null;

    function loadSelection() {
      try {
        var raw = sessionStorage.getItem(storageKey);
        return raw ? new Set(JSON.parse(raw)) : new Set();
      } catch (e) {
        return new Set();
      }
    }

    function saveSelection() {
      try {
        sessionStorage.setItem(storageKey, JSON.stringify(Array.from(selected)));
      } catch (e) { /* хранилище недоступно — просто не сохраняем между перезагрузками */ }
    }

    function currentPageFromUrl() {
      var m = /[?&]page=(\d+)/.exec(location.search);
      return m ? parseInt(m[1], 10) : 1;
    }

    function fieldParams() {
      var params = new URLSearchParams();
      if (form) {
        form.querySelectorAll("[data-filter-field]").forEach(function (field) {
          if (field.type === "checkbox") {
            if (field.checked) params.set(field.name, field.value);
          } else if (field.value) {
            params.set(field.name, field.value);
          }
        });
      }
      if (state.page > 1) params.set("page", String(state.page));
      return params;
    }

    function applySelectionToDom() {
      root.querySelectorAll('input[name="ids"]').forEach(function (cb) {
        cb.checked = selected.has(cb.value);
      });
      syncSelectAll();
      updateHint();
    }

    function syncSelectAll() {
      var all = root.querySelectorAll('input[name="ids"]');
      var master = root.querySelector("[data-select-all]");
      if (!master) return;
      master.checked = all.length > 0 && Array.prototype.every.call(all, function (cb) { return cb.checked; });
    }

    function updateHint() {
      var hint = root.querySelector("[data-selected-hint]");
      if (!hint) return;
      if (selected.size > 0) {
        hint.hidden = false;
        hint.textContent = "Выбрано всего: " + selected.size + (root.dataset.multiPage === "1" ? " (включая другие страницы/фильтры)" : "");
      } else {
        hint.hidden = true;
        hint.textContent = "";
      }
    }

    function load(pushUrl) {
      var params = fieldParams();
      var url = endpoint + (params.toString() ? "?" + params.toString() : "");
      fetch(url, { headers: { "X-Requested-With": "fetch" }, credentials: "same-origin" })
        .then(function (r) {
          if (!r.ok) throw new Error("HTTP " + r.status);
          return r.json();
        })
        .then(function (data) {
          root.innerHTML = data.html;
          state.page = data.page_num || 1;
          if (state.page > 1) root.dataset.multiPage = "1";
          applySelectionToDom();
          mirrorFiltersIntoDeleteFiltered(params);
          if (pushUrl !== false) {
            var newUrl = location.pathname + (params.toString() ? "?" + params.toString() : "");
            history.pushState({ listFilter: true, params: params.toString() }, "", newUrl);
          }
        })
        .catch(function () {
          // сеть/сервер недоступны — молча остаёмся на текущей выдаче, обычная навигация всё ещё работает
        });
    }

    function mirrorFiltersIntoDeleteFiltered(params) {
      var f = root.querySelector("[data-delete-filtered-form]");
      if (!f) return;
      f.querySelectorAll("[data-filter-mirror]").forEach(function (input) {
        input.value = params.get(input.dataset.filterMirror) || "";
      });
    }

    function scheduleLoad(immediate) {
      clearTimeout(debounceTimer);
      if (immediate) { load(); return; }
      debounceTimer = setTimeout(load, 300);
    }

    // ---- поле поиска/выбора: применяем фильтр автоматически ----
    if (form) {
      form.addEventListener("input", function (e) {
        if (e.target.matches('[data-filter-field][type="search"], [data-filter-field][type="text"]')) {
          state.page = 1;
          scheduleLoad(false);
        }
      });
      form.addEventListener("change", function (e) {
        if (e.target.matches('[data-filter-field]:not([type="search"]):not([type="text"])')) {
          state.page = 1;
          scheduleLoad(true);
        }
      });
    }

    // ---- пагинация: подгружаем страницу без перезагрузки, фильтр и выделение сохраняются ----
    root.addEventListener("click", function (e) {
      var link = e.target.closest(".pagination a.page-link");
      if (!link) return;
      e.preventDefault();
      var m = /[?&]page=(\d+)/.exec(link.getAttribute("href") || "");
      state.page = m ? parseInt(m[1], 10) : 1;
      load();
    });

    // ---- чекбоксы: запоминаем выделение независимо от текущей страницы/фильтра ----
    root.addEventListener("change", function (e) {
      var box = e.target;
      if (box.matches && box.matches('input[name="ids"]')) {
        if (box.checked) selected.add(box.value); else selected.delete(box.value);
        saveSelection();
        syncSelectAll();
        updateHint();
      } else if (box.matches && box.matches("[data-select-all]")) {
        root.querySelectorAll('input[name="ids"]').forEach(function (cb) {
          if (box.checked) selected.add(cb.value); else selected.delete(cb.value);
        });
        saveSelection();
        updateHint();
      }
    });

    // ---- перед отправкой удаления добавляем скрытые поля для id, выбранных на других
    // страницах/при другом фильтре (их чекбоксов сейчас нет в DOM) ----
    root.addEventListener("submit", function (e) {
      var deleteFiltered = e.target.closest("[data-delete-filtered-form]");
      if (deleteFiltered) {
        // удаляются вообще все записи под текущим фильтром (не только выбранные) —
        // прежнее выделение после этого бессмысленно
        clearSelection();
        return;
      }

      var f = e.target.closest("[data-bulk-delete-form]");
      if (!f) return;
      f.querySelectorAll('input[data-extra-id]').forEach(function (n) { n.remove(); });
      var present = new Set(
        Array.prototype.map.call(f.querySelectorAll('input[name="ids"]'), function (cb) { return cb.value; })
      );
      selected.forEach(function (id) {
        if (present.has(id)) return;
        var input = document.createElement("input");
        input.type = "hidden";
        input.name = "ids";
        input.value = id;
        input.setAttribute("data-extra-id", "1");
        f.appendChild(input);
      });
      // сама отправка формы удалит именно эти записи — выделение больше не актуально,
      // иначе после редиректа обратно на историю счётчик «выбрано» будет врать
      clearSelection();
    });

    function clearSelection() {
      selected.clear();
      saveSelection();
      updateHint();
    }

    // ---- назад/вперёд в браузере — перечитываем список под текущий URL ----
    window.addEventListener("popstate", function () {
      state.page = currentPageFromUrl();
      load(false);
    });

    applySelectionToDom();
  }
})();
