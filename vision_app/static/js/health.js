// Вкладки «Анализ» / «Чат» переключаются здесь, без запросов к серверу.
(function () {
  const page = document.getElementById("health-page");
  if (!page) return;
  const links = page.querySelectorAll("[data-tab-link]");
  let current = page.dataset.tab;

  function show(tab) {
    current = tab;
    page.dataset.tab = tab;
    links.forEach((a) => a.classList.toggle("is-active", a.dataset.tabLink === tab));
    const url = new URL(location.href);
    url.searchParams.set("tab", tab);
    url.searchParams.delete("refresh");
    history.replaceState(null, "", url);
  }

  links.forEach((a) =>
    a.addEventListener("click", (e) => {
      e.preventDefault();
      show(a.dataset.tabLink);
    })
  );

  // Формы без своей вкладки (ключ, параметры, список моделей, «Обновить статус») возвращают на открытую.
  function withTab(urlStr) {
    const url = new URL(urlStr, location.href);
    url.searchParams.set("tab", current);
    return url.pathname + url.search;
  }
  page.querySelectorAll("form").forEach((f) => {
    if (f.closest("[data-tab-only]")) return;
    f.addEventListener("submit", () => { f.action = withTab(f.getAttribute("action")); });
  });
  page.querySelectorAll("a[data-keep-tab]").forEach((a) =>
    a.addEventListener("click", () => { a.href = withTab(a.getAttribute("href")); })
  );
})();
