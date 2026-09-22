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

  // ---------- загрузка: подпись кнопки и список выбранных файлов ----------
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
  document.addEventListener("change", function (e) {
    if (e.target && e.target.id === "id_image") {
      var el = document.getElementById("file-name");
      if (el) el.textContent = describeFiles(e.target);
    }
  });
  document.addEventListener("drop", function () {
    // main.js кладёт файлы в input в своём обработчике drop; обновим подпись чуть позже
    setTimeout(function () {
      var input = document.getElementById("id_image");
      var el = document.getElementById("file-name");
      if (input && el) el.textContent = describeFiles(input);
    }, 0);
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
