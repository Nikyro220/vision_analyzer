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

  function el(tag, className) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    return node;
  }

  function renderMessage(role, text, meta) {
    var emptyHint = document.getElementById("chat-empty");
    if (emptyHint) emptyHint.remove();

    var bubble = el("div", "chat-msg chat-msg-" + role);
    bubble.appendChild(document.createTextNode(text));
    if (meta) {
      var metaEl = el("span", "chat-msg-meta");
      metaEl.textContent = meta;
      bubble.appendChild(metaEl);
    }
    log.appendChild(bubble);
    log.scrollTop = log.scrollHeight;
    return bubble;
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
    var activeRow = document.querySelector(".chat-session-row.is-active .mini-list-name");
    if (activeRow) activeRow.textContent = title;
    document.title = title + " · " + document.title.split(" · ").slice(1).join(" · ");
  }

  function sendMessage(message) {
    renderMessage("user", message);
    showError("");
    setSending(true);

    var pending = renderMessage("assistant", "…");
    pending.classList.add("chat-msg-pending");

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
        renderMessage("assistant", data.reply || "", meta);
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

  log.scrollTop = log.scrollHeight;
})();
