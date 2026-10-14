/* Страница «Логи»: смена файла/уровня/поиска и автообновление без перезагрузки.
 * Подгружает только блок с записями (fetch + X-Requested-With: fetch), как list-filter.js.
 * Без JS форма работает обычной отправкой GET. */
(function () {
  "use strict";

  var form = document.getElementById("logs-form");
  var results = document.getElementById("logs-results");
  var auto = document.getElementById("logs-auto");
  if (!form || !results) return;

  var endpoint = form.dataset.endpoint;
  var timer = null;
  var debounce = null;
  var inFlight = false;

  function params() {
    var p = new URLSearchParams();
    new FormData(form).forEach(function (value, key) {
      if (value) p.set(key, value);
    });
    return p;
  }

  function load() {
    if (inFlight) return;
    inFlight = true;
    var p = params();
    fetch(endpoint + "?" + p.toString(), { headers: { "X-Requested-With": "fetch" }, credentials: "same-origin" })
      .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
      .then(function (data) {
        results.innerHTML = data.html;
        history.replaceState(null, "", "?" + p.toString());
      })
      .catch(function () { /* оставляем прошлые записи на экране; автообновление попробует снова */ })
      .then(function () { inFlight = false; });
  }

  form.addEventListener("submit", function (e) { e.preventDefault(); load(); });
  form.querySelectorAll("select").forEach(function (el) { el.addEventListener("change", load); });
  form.querySelector('input[name="q"]').addEventListener("input", function () {
    clearTimeout(debounce);
    debounce = setTimeout(load, 300);
  });

  function schedule() {
    clearInterval(timer);
    if (auto.checked) {
      timer = setInterval(function () { if (!document.hidden) load(); }, 5000);
    }
  }
  auto.addEventListener("change", schedule);
})();
