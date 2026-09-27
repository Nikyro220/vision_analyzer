/* Страница «Чат». История хранится на сервере (ChatSession/ChatMessage) —
 * этот файл только отправляет сообщения в уже открытую сессию и подрисовывает
 * ответ, ничего не кладёт в localStorage. */
(function () {
  "use strict";

  // ---------- подтверждение удаления чата (обычная форма, без fetch) ----------
  document.addEventListener("submit", function (e) {
    var form = e.target;
    if (form && form.dataset && form.dataset.confirm && !window.confirm(form.dataset.confirm)) {
      e.preventDefault();
    }
  });

  var root = document.getElementById("chat-root");
  if (!root) return;

  var log = document.getElementById("chat-log");
  var form = document.getElementById("chat-form");
  var input = document.getElementById("chat-input");
  var sendBtn = document.getElementById("chat-send-btn");
  var errorMsg = document.getElementById("chat-error");
  var sendUrl = root.dataset.sendUrl;
  var csrf = root.dataset.csrf;

  var sending = false;

  var ROLE_LABELS = { user: "вы", assistant: "модель", error: "ошибка" };

  function el(tag, className) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    return node;
  }

  // На случай, если markdown.js по какой-то причине не подключился —
  // не оставляем пользователя без текста, просто без форматирования.
  function escapeForFallback(text) {
    var div = el("div");
    div.textContent = text;
    return div.innerHTML.replace(/\n/g, "<br>");
  }

  // Реплики, отрисованные сервером при загрузке страницы (Jinja положил туда
  // экранированный текст как textContent) — прогоняем через тот же рендерер,
  // чтобы markdown работал одинаково и в истории, и в новых сообщениях.
  function renderExistingTurns() {
    if (window.renderMarkdownToHtml) {
      var textNodes = log.querySelectorAll(".chat-turn-text[data-raw-text]");
      for (var i = 0; i < textNodes.length; i += 1) {
        var node = textNodes[i];
        var raw = node.textContent;
        node.innerHTML = window.renderMarkdownToHtml(raw);
        node.removeAttribute("data-raw-text");
      }
    }

    var refNodes = log.querySelectorAll(".chat-refs[data-refs]");
    for (var j = 0; j < refNodes.length; j += 1) {
      var refNode = refNodes[j];
      var refs = [];
      try {
        refs = JSON.parse(refNode.getAttribute("data-refs") || "[]");
      } catch (e) {
        refs = [];
      }
      refNode.removeAttribute("data-refs");
      fillRefCards(refNode, refs);
    }
  }

  var RISK_CLASS = { low: "chat-ref-risk-low", medium: "chat-ref-risk-medium", high: "chat-ref-risk-high" };

  // Карточка-ссылка на конкретный анализ: миниатюра + название + бейдж риска.
  // Ссылки/данные приходят готовыми с сервера (analysis_query.py собирает их
  // из тех же строк БД, что и текстовую сводку для модели) — фронт их только
  // рисует, ничего не парсит из текста ответа модели.
  function buildRefCard(ref) {
    var card = document.createElement("a");
    card.className = "chat-ref-card";
    card.href = ref.url || "#";
    card.target = "_blank";
    card.rel = "noopener noreferrer";

    var thumbWrap = el("span", "chat-ref-thumb");
    if (ref.thumb_url) {
      var img = document.createElement("img");
      img.src = ref.thumb_url;
      img.alt = ref.label || "";
      img.loading = "lazy";
      thumbWrap.appendChild(img);
    } else {
      thumbWrap.classList.add("chat-ref-thumb-empty");
    }
    card.appendChild(thumbWrap);

    var info = el("span", "chat-ref-info");

    var label = el("span", "chat-ref-label");
    label.textContent = ref.label || "Анализ #" + ref.id;
    info.appendChild(label);

    var metaLine = el("span", "chat-ref-meta");
    var badge = el("span", "chat-ref-badge " + (RISK_CLASS[ref.risk_level] || ""));
    badge.textContent = ref.risk_label || ref.risk_level || "";
    metaLine.appendChild(badge);
    var dateSpan = el("span", "chat-ref-date");
    dateSpan.textContent = [ref.username, ref.date].filter(Boolean).join(" · ");
    metaLine.appendChild(dateSpan);
    info.appendChild(metaLine);

    card.appendChild(info);
    return card;
  }

  function fillRefCards(container, refs) {
    if (!refs || !refs.length) return;
    container.innerHTML = "";
    refs.forEach(function (ref) {
      container.appendChild(buildRefCard(ref));
    });
  }

  function pad2(n) {
    return (n < 10 ? "0" : "") + n;
  }

  function nowHM() {
    var d = new Date();
    return pad2(d.getHours()) + ":" + pad2(d.getMinutes());
  }

  // Реплика — не пузырь, а строка «журнала сессии»: роль + время сверху
  // (моноширинным, как остальные технические метки в панели), текст снизу.
  function renderTurn(role, text, meta, refs) {
    var emptyHint = document.getElementById("chat-empty");
    if (emptyHint) emptyHint.remove();

    var turn = el("div", "chat-turn chat-turn-" + role);

    var head = el("div", "chat-turn-head");
    var roleEl = el("span", "chat-turn-role");
    roleEl.textContent = ROLE_LABELS[role] || role;
    var timeEl = el("span", "chat-turn-time");
    timeEl.textContent = nowHM();
    head.appendChild(roleEl);
    head.appendChild(timeEl);

    var body = el("div", "chat-turn-body");
    if (text) {
      var textEl = el("span", "chat-turn-text");
      textEl.innerHTML = window.renderMarkdownToHtml ? window.renderMarkdownToHtml(text) : escapeForFallback(text);
      body.appendChild(textEl);
    }
    if (meta) {
      var metaEl = el("span", "chat-turn-meta");
      metaEl.textContent = meta;
      body.appendChild(metaEl);
    }

    turn.appendChild(head);
    turn.appendChild(body);

    if (refs && refs.length) {
      var refsEl = el("div", "chat-refs");
      fillRefCards(refsEl, refs);
      turn.appendChild(refsEl);
    }

    log.appendChild(turn);
    log.scrollTop = log.scrollHeight;
    return turn;
  }

  // Пока модель отвечает — реплика с тремя пульсирующими точками вместо текста.
  function renderPending() {
    var turn = renderTurn("assistant", "");
    turn.classList.add("chat-turn-pending");
    var body = turn.querySelector(".chat-turn-body");
    var dots = el("span", "chat-typing");
    dots.innerHTML = "<span></span><span></span><span></span>";
    body.appendChild(dots);
    return turn;
  }

  function autoGrow() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 200) + "px";
  }

  function setSending(state) {
    sending = state;
    sendBtn.disabled = state;
    input.disabled = state;
    sendBtn.textContent = state ? "Отправка…" : "Отправить";
  }

  function showError(text) {
    errorMsg.textContent = text;
    errorMsg.hidden = !text;
  }

  // Заголовок чата в сайдбаре мог быть пустым ("Новый чат") до первого сообщения —
  // как только сервер вернёт настоящий заголовок, подставляем его на месте, без перезагрузки.
  function updateSidebarTitle(title) {
    if (!title) return;
    var activeRow = document.querySelector(".chat-session-row.is-active .chat-session-name");
    if (activeRow) activeRow.textContent = title;
    document.title = title + " · " + document.title.split(" · ").slice(1).join(" · ");
  }

  function sendMessage(message) {
    renderTurn("user", message);
    showError("");
    setSending(true);

    var pending = renderPending();

    fetch(sendUrl, {
      method: "POST",
      credentials: "same-origin",
      headers: {
        "Content-Type": "application/json",
        "X-CSRFToken": csrf,
        Accept: "application/json",
      },
      body: JSON.stringify({ message: message }),
    })
      .then(function (r) {
        return r.json().then(function (data) {
          return { ok: r.ok, data: data };
        });
      })
      .then(function (result) {
        pending.remove();
        if (!result.ok) {
          showError((result.data && result.data.error) || "Не удалось получить ответ.");
          return;
        }
        var data = result.data || {};
        var meta = data.backend ? data.backend + (data.model ? " · " + data.model : "") : "";
        renderTurn("assistant", data.reply || "", meta, data.refs);
        updateSidebarTitle(data.title);
      })
      .catch(function () {
        pending.remove();
        showError("Не удалось связаться с сервером. Проверьте соединение и повторите.");
      })
      .finally(function () {
        setSending(false);
        input.focus();
      });
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    if (sending) return;
    var message = input.value.trim();
    if (!message) return;
    input.value = "";
    autoGrow();
    sendMessage(message);
  });

  input.addEventListener("input", autoGrow);
  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit ? form.requestSubmit() : form.dispatchEvent(new Event("submit", { cancelable: true }));
    }
  });

  renderExistingTurns();
  log.scrollTop = log.scrollHeight;
})();