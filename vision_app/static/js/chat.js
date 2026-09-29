/* Страница «Чат». История хранится на сервере (ChatSession/ChatMessage) —
 * этот файл только отправляет сообщения в уже открытую сессию и подрисовывает
 * ответ, ничего не кладёт в localStorage.
 *
 * Вложения: изображения выбираются скрепкой, перетаскиванием или вставкой из буфера,
 * показываются превью-чипами над полем ввода и уходят вместе с сообщением как
 * multipart/form-data (поля message + images). Проверки размера/числа здесь — только
 * для удобства, настоящие лимиты применяет сервер (chat_images.py).
 *
 * Анализ изображений идёт через общую очередь и занимает время: пока в чате есть
 * задачи, ждущие результата, страница раз в несколько секунд опрашивает
 * GET .../pending, а пришедшие ответы модели дорисовывает в лог. */
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
  var fileInput = document.getElementById("chat-file-input");
  var attachBtn = document.getElementById("chat-attach-btn");
  var attachBox = document.getElementById("chat-attachments");
  var sendUrl = root.dataset.sendUrl;
  var csrf = root.dataset.csrf;
  var maxImages = parseInt(root.dataset.maxImages, 10) || 4;
  var maxImageMb = parseInt(root.dataset.maxImageMb, 10) || 15;
  var pendingUrl = root.dataset.pendingUrl;
  var pendingJobs = parseInt(root.dataset.pending, 10) || 0; // сколько изображений ещё ждут анализа

  var sending = false;
  var pending = []; // вложения, ещё не отправленные: [{file, url}]

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
  function renderTurn(role, text, meta, refs, images) {
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

    if (images && images.length) {
      var imagesEl = el("div", "chat-turn-images");
      images.forEach(function (item) {
        var link = el("a", "chat-turn-image");
        link.href = item.url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.title = item.name || "";
        var img = document.createElement("img");
        img.src = item.url;
        img.alt = item.name || "";
        link.appendChild(img);
        imagesEl.appendChild(link);
      });
      turn.appendChild(imagesEl);
    }

    // Реплика из одних картинок (без текста) — без пустого блока текста.
    if (text || meta || role !== "user" || !(images && images.length)) {
      turn.appendChild(body);
    }

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
    attachBtn.disabled = state;
    sendBtn.textContent = state ? "Отправка…" : "Отправить";
  }

  // ---------- вложения ----------

  function renderAttachments() {
    attachBox.innerHTML = "";
    attachBox.hidden = pending.length === 0;
    pending.forEach(function (item, index) {
      var chip = el("div", "chat-attach-chip");
      chip.title = item.file.name;

      var img = document.createElement("img");
      img.src = item.url;
      img.alt = item.file.name;
      chip.appendChild(img);

      var name = el("span", "chat-attach-name");
      name.textContent = item.file.name || "изображение";
      chip.appendChild(name);

      var remove = el("button", "chat-attach-remove");
      remove.type = "button";
      remove.textContent = "✕";
      remove.setAttribute("aria-label", "Убрать вложение");
      remove.addEventListener("click", function () {
        if (sending) return;
        URL.revokeObjectURL(item.url);
        pending.splice(index, 1);
        renderAttachments();
      });
      chip.appendChild(remove);

      attachBox.appendChild(chip);
    });
  }

  function addFiles(fileList) {
    var skipped = [];
    Array.prototype.forEach.call(fileList, function (file) {
      if (!file.type || file.type.indexOf("image/") !== 0) {
        skipped.push("«" + (file.name || "файл") + "» — не изображение");
        return;
      }
      if (file.size > maxImageMb * 1024 * 1024) {
        skipped.push("«" + (file.name || "файл") + "» — больше " + maxImageMb + " МБ");
        return;
      }
      if (pending.length >= maxImages) {
        skipped.push("«" + (file.name || "файл") + "» — максимум " + maxImages + " изображений в сообщении");
        return;
      }
      pending.push({ file: file, url: URL.createObjectURL(file) });
    });
    renderAttachments();
    showError(skipped.length ? "Не прикреплено: " + skipped.join("; ") + "." : "");
  }

  function clearPending() {
    // blob-URL не отзываем: превью уже перекочевали в реплику пользователя в логе.
    pending = [];
    renderAttachments();
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

  // ---------- ожидание результатов анализа из очереди ----------

  var POLL_MS = 4000;
  var pollTimer = null;
  var polling = false; // запрос в полёте — не накладываем опросы друг на друга

  function pollPending() {
    pollTimer = null;
    if (!pendingUrl || polling || pendingJobs <= 0) return;
    polling = true;
    fetch(pendingUrl, {
      credentials: "same-origin",
      headers: { Accept: "application/json" },
      cache: "no-store",
    })
      .then(function (r) {
        return r.ok ? r.json() : null;
      })
      .then(function (data) {
        if (!data) return;
        (data.messages || []).forEach(function (m) {
          var meta = m.backend ? m.backend + (m.model ? " · " + m.model : "") : "";
          renderTurn("assistant", m.reply || "", meta, m.refs);
        });
        pendingJobs = data.pending || 0;
      })
      .catch(function () {
        /* временный сбой сети — попробуем в следующий раз */
      })
      .finally(function () {
        polling = false;
        schedulePoll();
      });
  }

  function schedulePoll() {
    if (pollTimer !== null || pendingJobs <= 0) return;
    // вкладка в фоне — опрашиваем реже
    pollTimer = window.setTimeout(pollPending, document.hidden ? POLL_MS * 3 : POLL_MS);
  }

  document.addEventListener("visibilitychange", function () {
    if (!document.hidden && pendingJobs > 0 && !polling) {
      if (pollTimer !== null) window.clearTimeout(pollTimer);
      pollTimer = null;
      pollPending(); // вернулись на вкладку — проверяем сразу
    }
  });

  function sendMessage(message, attachments) {
    renderTurn(
      "user",
      message,
      "",
      null,
      attachments.map(function (item) {
        return { url: item.url, name: item.file.name };
      })
    );
    showError("");
    setSending(true);

    var pendingTurn = renderPending();

    // multipart: без ручного Content-Type — браузер сам поставит boundary.
    var formData = new FormData();
    formData.append("message", message);
    attachments.forEach(function (item) {
      formData.append("images", item.file, item.file.name || "image");
    });

    fetch(sendUrl, {
      method: "POST",
      credentials: "same-origin",
      headers: {
        "X-CSRFToken": csrf,
        Accept: "application/json",
      },
      body: formData,
    })
      .then(function (r) {
        return r.json().then(
          function (data) {
            return { ok: r.ok, data: data };
          },
          function () {
            // Не JSON (например, редирект на страницу ошибки при слишком большом запросе).
            return { ok: false, data: { error: "Сервер отклонил запрос (возможно, файлы слишком большие)." } };
          }
        );
      })
      .then(function (result) {
        pendingTurn.remove();
        if (!result.ok) {
          showError((result.data && result.data.error) || "Не удалось получить ответ.");
          return;
        }
        var data = result.data || {};
        var meta = data.backend ? data.backend + (data.model ? " · " + data.model : "") : "";
        renderTurn("assistant", data.reply || "", meta, data.refs);
        updateSidebarTitle(data.title);
        pendingJobs = data.pending || 0;
        schedulePoll();
      })
      .catch(function () {
        pendingTurn.remove();
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
    if (!message && !pending.length) return;
    var attachments = pending.slice();
    input.value = "";
    autoGrow();
    clearPending();
    sendMessage(message, attachments);
  });

  attachBtn.addEventListener("click", function () {
    if (!sending) fileInput.click();
  });
  fileInput.addEventListener("change", function () {
    if (fileInput.files && fileInput.files.length) addFiles(fileInput.files);
    fileInput.value = ""; // чтобы тот же файл можно было выбрать повторно
  });

  // Вставка изображения из буфера обмена (скриншот, «копировать картинку»).
  input.addEventListener("paste", function (e) {
    var items = (e.clipboardData && e.clipboardData.items) || [];
    var files = [];
    for (var i = 0; i < items.length; i += 1) {
      if (items[i].kind === "file" && items[i].type.indexOf("image/") === 0) {
        var file = items[i].getAsFile();
        if (file) files.push(file);
      }
    }
    if (files.length && !sending) {
      e.preventDefault(); // текст из буфера (если был) не вставляем — вставляем именно картинку
      addFiles(files);
    }
  });

  // Перетаскивание файлов на карточку чата.
  var dragDepth = 0;
  function hasFiles(e) {
    return !!(e.dataTransfer && Array.prototype.indexOf.call(e.dataTransfer.types || [], "Files") !== -1);
  }
  root.addEventListener("dragenter", function (e) {
    if (!hasFiles(e)) return;
    e.preventDefault();
    dragDepth += 1;
    root.classList.add("is-dragover");
  });
  root.addEventListener("dragover", function (e) {
    if (hasFiles(e)) e.preventDefault();
  });
  root.addEventListener("dragleave", function (e) {
    if (!hasFiles(e)) return;
    dragDepth = Math.max(0, dragDepth - 1);
    if (!dragDepth) root.classList.remove("is-dragover");
  });
  root.addEventListener("drop", function (e) {
    if (!hasFiles(e)) return;
    e.preventDefault();
    dragDepth = 0;
    root.classList.remove("is-dragover");
    if (!sending && e.dataTransfer.files.length) addFiles(e.dataTransfer.files);
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
  if (pendingJobs > 0) pollPending(); // страницу открыли/перезагрузили, пока анализ шёл — сразу проверяем
})();