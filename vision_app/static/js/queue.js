/* Окно очереди на странице «Новый анализ» + мелкие помощники для таблиц истории.
 * Подключается после main.js. На страницах без нужных элементов ничего не делает. */
(function () {
  "use strict";

  // ---------- «выбрать все» в таблицах истории ----------
  document.addEventListener("change", function (e) {
    var box = e.target;
    if (box.matches && box.matches("input[data-select-all]")) {
      var scope = box.closest("form") || document;
      scope.querySelectorAll('input[name="ids"]').forEach(function (cb) { cb.checked = box.checked; });
    }
  });

  // ---------- загрузка: подпись кнопки ----------
  // main.js ставит «Анализируем…» на кнопку. Анализ теперь идёт в фоне, а на кнопке
  // отражается только отправка файлов — поэтому переопределяем подпись (наш обработчик
  // на document срабатывает после обработчиков самой формы).
  document.addEventListener("submit", function (e) {
    if (e.target && e.target.id === "upload-form") {
      var btn = e.target.querySelector("#submit-btn");
      if (btn) btn.textContent = "Загружаем…";
    }
  });

  function describeFiles(input) {
    var files = input.files;
    if (!files || !files.length) return "";
    if (files.length === 1) return files[0].name;
    var names = Array.prototype.slice.call(files, 0, 3).map(function (f) { return f.name; }).join(", ");
    return "Файлов: " + files.length + " — " + names + (files.length > 3 ? "…" : "");
  }

  // ---------- список выбранных файлов: миниатюра + подпись + удаление ----------
  // Подписи (caption) идут отдельным полем "captions", по одной на файл, в ТОМ ЖЕ
  // порядке, что и файлы — forms.py.validate_image() сопоставляет их по индексу.
  // Правка одного файла не должна стирать то, что уже введено для остальных, поэтому
  // введённые подписи запоминаются в captionMap (File -> текст) и переживают
  // перерисовку списка при добавлении/удалении файлов.
  var captionMap = new WeakMap();
  var thumbUrls = [];

  function clearThumbUrls() {
    thumbUrls.forEach(function (url) { URL.revokeObjectURL(url); });
    thumbUrls = [];
  }

  function harvestCaptions(input) {
    var rows = document.querySelectorAll("#caption-list .file-row");
    Array.prototype.forEach.call(input.files, function (file, i) {
      var row = rows[i];
      var field = row && row.querySelector('input[name="captions"]');
      if (field && field.value) captionMap.set(file, field.value);
    });
  }

  function removeFileAt(input, index) {
    harvestCaptions(input); // сохранить уже введённые подписи ДО того, как список изменится
    var dt = new DataTransfer();
    Array.prototype.forEach.call(input.files, function (file, i) {
      if (i !== index) dt.items.add(file);
    });
    input.files = dt.files;
    refreshSelection(input);
  }

  function clearAllFiles(input) {
    input.files = new DataTransfer().files;
    refreshSelection(input);
  }

  function renderSelectedFiles(input) {
    var box = document.getElementById("caption-list");
    if (!box) return;
    var files = input.files;
    clearThumbUrls();
    box.textContent = "";

    if (!files || files.length < 1) {
      box.hidden = true;
      return;
    }

    var head = document.createElement("div");
    head.className = "caption-list-head";
    var count = document.createElement("span");
    count.textContent = "Выбрано: " + files.length;
    var clearBtn = document.createElement("button");
    clearBtn.type = "button";
    clearBtn.className = "caption-list-clear";
    clearBtn.textContent = "Очистить всё";
    clearBtn.addEventListener("click", function () { clearAllFiles(input); });
    head.appendChild(count);
    head.appendChild(clearBtn);
    box.appendChild(head);

    Array.prototype.forEach.call(files, function (file, i) {
      var row = document.createElement("div");
      row.className = "file-row";

      var thumb = document.createElement("img");
      thumb.className = "file-thumb";
      thumb.alt = "";
      var url = URL.createObjectURL(file);
      thumbUrls.push(url);
      thumb.src = url;

      var name = document.createElement("span");
      name.className = "caption-row-name";
      name.textContent = file.name;
      name.title = file.name;

      var field = document.createElement("input");
      field.type = "text";
      field.name = "captions";
      field.maxLength = 500;
      field.placeholder = "Комментарий к этому изображению (необязательно)";
      field.value = captionMap.get(file) || "";

      var removeBtn = document.createElement("button");
      removeBtn.type = "button";
      removeBtn.className = "file-remove-btn";
      removeBtn.title = "Убрать этот файл из выбора";
      removeBtn.setAttribute("aria-label", "Убрать этот файл из выбора");
      removeBtn.textContent = "✕";
      removeBtn.addEventListener("click", function () { removeFileAt(input, i); });

      row.appendChild(thumb);
      row.appendChild(name);
      row.appendChild(field);
      row.appendChild(removeBtn);
      box.appendChild(row);
    });

    box.hidden = false;
  }

  function refreshSelection(input) {
    var el = document.getElementById("file-name");
    if (el) el.textContent = describeFiles(input);
    renderSelectedFiles(input);
  }

  document.addEventListener("change", function (e) {
    if (e.target && e.target.id === "id_image") refreshSelection(e.target);
  });
  document.addEventListener("drop", function () {
    // main.js кладёт файлы в input в своём обработчике drop; обновим список чуть позже
    setTimeout(function () {
      var input = document.getElementById("id_image");
      if (input) refreshSelection(input);
    }, 0);
  });

  // ---------- клик по строке таблицы (история, «Все анализы») открывает результат ----------
  document.addEventListener("click", function (e) {
    var row = e.target.closest(".row-clickable");
    if (!row) return;
    var interactive = e.target.closest("a, button, input, label");
    if (interactive && row.contains(interactive)) return;
    window.location = row.dataset.href;
  });

  // ---------- окно очереди ----------
  var root = document.getElementById("queue-root");
  if (!root) return;

  var list = document.getElementById("queue-list");
  var emptyMsg = document.getElementById("queue-empty");
  var counter = document.getElementById("queue-count");
  var errorMsg = document.getElementById("queue-error");
  var recentList = document.getElementById("recent-list");
  var recentMore = document.getElementById("recent-more");
  var recentEmpty = document.getElementById("recent-empty");
  var statusUrl = root.dataset.statusUrl;
  var csrf = root.dataset.csrf;

  var timer = null;
  var failures = 0;

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function truncate(s, n) { return s.length > n ? s.slice(0, n - 1) + "…" : s; }

  function fmtElapsed(sec) {
    var m = Math.floor(sec / 60), s = sec % 60;
    return m + ":" + (s < 10 ? "0" : "") + s;
  }

  function statusText(item) {
    if (item.status === "processing") return "Обрабатывается · " + fmtElapsed(item.elapsed);
    return item.ahead > 0 ? "В очереди · перед вами: " + item.ahead : "В очереди · следующий";
  }

  // Зеркалит макрос queue_row в dashboard.html. Текст задаём через textContent — без innerHTML.
  function renderQueueItem(item) {
    var li = el("li", "queue-item queue-item--" + item.status);
    li.dataset.id = item.id;
    var icon = el("span", "queue-icon");
    icon.setAttribute("aria-hidden", "true");
    var name = el("span", "queue-name", truncate(item.name, 30));
    name.title = item.name;
    li.appendChild(icon);
    li.appendChild(name);
    li.appendChild(el("span", "queue-status", statusText(item)));

    if (item.cancel_url) {
      var form = el("form", "queue-cancel");
      form.method = "post";
      form.action = item.cancel_url;
      var token = document.createElement("input");
      token.type = "hidden"; token.name = "csrf_token"; token.value = csrf;
      var btn = el("button", "queue-cancel-btn", "✕");
      btn.type = "submit"; btn.title = "Убрать из очереди";
      btn.setAttribute("aria-label", "Убрать из очереди");
      form.appendChild(token);
      form.appendChild(btn);
      li.appendChild(form);
    }
    return li;
  }

  function renderRecentItem(item) {
    var li = document.createElement("li");
    var a = el("a", "mini-list-row");
    a.href = item.url;
    a.appendChild(el("span", "risk-dot risk-" + item.risk_level));
    a.appendChild(el("span", "mini-list-name", truncate(item.name, 28)));
    if (item.is_new) {
      var badge = el("span", "tag tag-new", "новое");
      badge.style.marginLeft = "6px";
      a.appendChild(badge);
    }
    a.appendChild(el("span", "mini-list-date", item.date));
    li.appendChild(a);
    return li;
  }

  function replaceChildren(node, children) {
    while (node.firstChild) node.removeChild(node.firstChild);
    children.forEach(function (c) { node.appendChild(c); });
  }

  function render(data) {
    replaceChildren(list, data.pending.map(renderQueueItem));
    emptyMsg.hidden = data.pending.length > 0;
    counter.hidden = data.pending.length === 0;
    counter.textContent = data.pending.length;
    root.dataset.pending = data.pending.length;

    if (recentList) {
      replaceChildren(recentList, data.recent.map(renderRecentItem));
      recentList.hidden = data.recent.length === 0;
      if (recentMore) recentMore.hidden = data.recent.length === 0;
      if (recentEmpty) recentEmpty.hidden = data.recent.length > 0;
    }
  }

  function schedule(ms) {
    clearTimeout(timer);
    timer = setTimeout(poll, ms);
  }

  function poll() {
    if (document.hidden) { schedule(3000); return; }   // вкладка неактивна — не дёргаем сервер
    fetch(statusUrl, { headers: { "Accept": "application/json" }, credentials: "same-origin", cache: "no-store" })
      .then(function (r) {
        if (r.redirected || !r.ok) throw new Error("HTTP " + r.status);  // сессия истекла и т.п.
        return r.json();
      })
      .then(function (data) {
        failures = 0;
        errorMsg.hidden = true;
        render(data);
        // пока есть незавершённое — опрашиваем часто; иначе редко (вдруг добавили с другой вкладки)
        schedule(data.pending.length ? 2000 : 15000);
      })
      .catch(function () {
        failures += 1;
        errorMsg.hidden = false;
        schedule(Math.min(3000 * failures, 15000));
      });
  }

  document.addEventListener("visibilitychange", function () { if (!document.hidden) schedule(0); });

  // Если на странице уже есть незавершённые анализы — начинаем следить сразу.
  schedule(Number(root.dataset.pending) > 0 ? 1500 : 15000);
})();