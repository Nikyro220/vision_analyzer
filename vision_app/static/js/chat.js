/* Страница «Чат». История переписки хранится на сервере (ChatSession/ChatMessage) —
 * этот файл отправляет сообщения в уже открытую сессию и подрисовывает ответ.
 * В localStorage лежит только то, что относится к строке ввода:
 *   - черновик (vision_chat:draft:<id чата>) — не пропадает при перезагрузке;
 *   - «исходящее» (vision_chat:outbox:<id чата>) — текст, который сейчас уходит модели: если ход
 *     не удался, а страницу успели перезагрузить, текст возвращается в поле ввода;
 *   - история ввода (vision_chat:history) — листается стрелками ↑/↓, как в терминале.
 *
 * Сообщение пользователя сервер сохраняет сразу, до ответа модели. Поэтому после перезагрузки
 * посреди ответа страница показывает сообщение и «печатает…» и дожидается ответа опросом
 * GET .../state (resumeTurn). При ошибке текст и вложения возвращаются в поле ввода (failTurn).
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

  var stateUrl = root.dataset.stateUrl;
  var sessionId = root.dataset.sessionId || "0";
  var serverBusy = root.dataset.busy === "1"; // страницу открыли, пока модель ещё отвечает
  var busySince = parseInt(root.dataset.busySince, 10) || 0;
  var emptyHint = document.getElementById("chat-empty");

  var sending = false;
  var pending = []; // вложения, ещё не отправленные: [{file, url}]
  var seen = {}; // id сообщений, уже нарисованных в логе

  var ROLE_LABELS = { user: "вы", assistant: "модель", error: "ошибка" };

  function el(tag, className) {
    var node = document.createElement(tag);
    if (className) node.className = className;
    return node;
  }

  // ---------- localStorage: черновик, «исходящее», история ввода ----------

  var LS_PREFIX = "vision_chat:";
  var draftKey = "draft:" + sessionId;
  var outboxKey = "outbox:" + sessionId;
  var HISTORY_KEY = "history";
  var HISTORY_MAX = 100;
  var OUTBOX_MAX_AGE_MS = 60 * 60 * 1000;

  // localStorage может быть недоступен (приватный режим, запрет в настройках) — тогда всё это
  // просто не работает, но чат остаётся рабочим.
  function lsGet(key) {
    try {
      return window.localStorage.getItem(LS_PREFIX + key);
    } catch (e) {
      return null;
    }
  }

  function lsSet(key, value) {
    try {
      window.localStorage.setItem(LS_PREFIX + key, value);
    } catch (e) {
      /* переполнено или запрещено — не страшно */
    }
  }

  function lsRemove(key) {
    try {
      window.localStorage.removeItem(LS_PREFIX + key);
    } catch (e) {
      /* см. выше */
    }
  }

  function saveDraft() {
    if (input.value) lsSet(draftKey, input.value);
    else lsRemove(draftKey);
  }

  function readOutbox() {
    var raw = lsGet(outboxKey);
    if (!raw) return null;
    try {
      var data = JSON.parse(raw);
      return data && typeof data.text === "string" ? data : null;
    } catch (e) {
      return null;
    }
  }

  function writeOutbox(text, since) {
    lsSet(outboxKey, JSON.stringify({ text: text, since: since, ts: Date.now() }));
  }

  // История ввода — общая для всех чатов, как история команд в терминале.
  var sentHistory = (function () {
    try {
      var list = JSON.parse(lsGet(HISTORY_KEY) || "[]");
      return Array.isArray(list)
        ? list.filter(function (item) {
            return typeof item === "string" && item;
          })
        : [];
    } catch (e) {
      return [];
    }
  })();
  var histPos = sentHistory.length; // == length — не листаем, в поле обычный ввод
  var histDraft = ""; // что было в поле, когда начали листать историю

  // Первый запуск (истории ещё нет): берём то, что уже написано в открытом чате.
  function seedHistoryFromLog() {
    if (sentHistory.length) return;
    var nodes = log.querySelectorAll(".chat-turn-user .chat-turn-text[data-raw-text]");
    for (var i = 0; i < nodes.length; i += 1) {
      var text = nodes[i].textContent;
      if (text && sentHistory[sentHistory.length - 1] !== text) sentHistory.push(text);
    }
    sentHistory = sentHistory.slice(-HISTORY_MAX);
    histPos = sentHistory.length;
  }

  function rememberSent(text) {
    if (text && sentHistory[sentHistory.length - 1] !== text) sentHistory.push(text);
    if (sentHistory.length > HISTORY_MAX) sentHistory = sentHistory.slice(-HISTORY_MAX);
    lsSet(HISTORY_KEY, JSON.stringify(sentHistory));
    histPos = sentHistory.length;
    histDraft = "";
  }

  function setInputValue(value) {
    input.value = value;
    autoGrow();
    input.setSelectionRange(value.length, value.length);
  }

  // Шаг по истории: dir = -1 — к более старым, +1 — к более новым. Возвращает true, если нажатие
  // «съедено» историей; false — пусть стрелка работает как обычно (двигает каретку по тексту).
  function historyStep(dir) {
    if (sending || !sentHistory.length) return false;
    var start = input.selectionStart;
    if (start !== input.selectionEnd) return false; // есть выделение — не мешаем
    var value = input.value;
    var browsing = histPos < sentHistory.length;

    if (dir < 0) {
      // Вверх: поле пустое, каретка в самом начале или (при листании) на первой строке.
      var onFirstLine = value.slice(0, start).indexOf("\n") === -1;
      if (!(!value || start === 0 || (browsing && onFirstLine))) return false;
      if (histPos === 0) return true; // дошли до самого старого
      if (!browsing) histDraft = value;
      histPos -= 1;
      setInputValue(sentHistory[histPos]);
      return true;
    }

    // Вниз: только во время листания и когда каретка на последней строке.
    if (!browsing || value.indexOf("\n", start) !== -1) return false;
    histPos += 1;
    setInputValue(histPos >= sentHistory.length ? histDraft : sentHistory[histPos]);
    return true;
  }

  // Вернуть текст и вложения в строку ввода (после ошибки отправки).
  function restoreComposer(text, attachments) {
    if (text) input.value = text + (input.value ? "\n" + input.value : "");
    if (attachments && attachments.length) {
      pending = attachments.slice();
      renderAttachments();
    }
    autoGrow();
    saveDraft();
    input.setSelectionRange(input.value.length, input.value.length);
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
    if (ref.kind === "user") return buildUserCard(ref);
    if (ref.kind === "action") return buildActionCard(ref);
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

  // Карточка пользователя (результат инструмента search_users, только для админов):
  // аватар (картинка или инициалы на цвете пользователя), логин, роль и ссылка на его страницу в панели.
  var HEX_RE = /^#[0-9a-fA-F]{6}$/;

  function buildUserCard(ref) {
    var card = document.createElement("a");
    card.className = "chat-ref-card chat-ref-user" + (ref.active === false || ref.role === "blocked" ? " is-blocked" : "");
    card.href = ref.url || "#";
    card.target = "_blank";
    card.rel = "noopener noreferrer";

    var avatar = el("span", "chat-ref-avatar");
    if (HEX_RE.test(ref.avatar_color || "")) avatar.style.setProperty("--avatar-bg", ref.avatar_color);
    if (HEX_RE.test(ref.avatar_fg || "")) avatar.style.setProperty("--avatar-fg", ref.avatar_fg);
    if (ref.avatar_url) {
      var img = document.createElement("img");
      img.src = ref.avatar_url;
      img.alt = "";
      img.loading = "lazy";
      avatar.appendChild(img);
    } else {
      avatar.textContent = ref.initials || (ref.label || "?").charAt(0).toUpperCase();
    }
    card.appendChild(avatar);

    var info = el("span", "chat-ref-info");
    var label = el("span", "chat-ref-label");
    label.textContent = ref.label || "Пользователь #" + ref.id;
    if (ref.subtitle) label.title = ref.subtitle;
    info.appendChild(label);

    var metaLine = el("span", "chat-ref-meta");
    var badge = el("span", "chat-ref-badge chat-ref-role-" + String(ref.role || "").replace(/[^a-z_]/g, ""));
    badge.textContent = ref.role_label || ref.role || "";
    metaLine.appendChild(badge);
    var meta = el("span", "chat-ref-date");
    var parts = [];
    if (typeof ref.analyses_count === "number") parts.push("анализов: " + ref.analyses_count);
    if (ref.date) parts.push(ref.date);
    meta.textContent = parts.join(" · ");
    metaLine.appendChild(meta);
    info.appendChild(metaLine);

    card.appendChild(info);
    return card;
  }

  // Заявка на действие над пользователем (инструмент manage_user, только главный админ). Модель ничего
  // не выполняет: действие происходит, только когда человек нажимает «Подтвердить» (POST confirm_url).
  // Статус карточки — всегда из ответа сервера (state_url), а не из сохранённого в сообщении.
  var ACTION_STATUS = {
    running: "Выполняется…", done: "Выполнено", failed: "Не выполнено",
    cancelled: "Отменено", expired: "Срок подтверждения истёк"
  };

  function buildActionCard(initial) {
    var card = el("div", "chat-ref-card chat-action");

    function post(url, body, onDone) {
      fetch(url, {
        method: "POST",
        credentials: "same-origin",
        headers: { "Content-Type": "application/json", "X-CSRFToken": csrf },
        body: JSON.stringify(body || {})
      })
        .then(function (r) {
          return r.json().catch(function () { return {}; });
        })
        .then(function (data) {
          if (data && data.card) render(data.card, data.ok === false ? data.message : "");
          else render(initial, "Не удалось выполнить запрос.");
        })
        .catch(function () {
          render(initial, "Нет связи с сервером.");
        });
      if (onDone) onDone();
    }

    function render(ref, note) {
      initial = ref;
      card.className = "chat-ref-card chat-action" + (ref.danger ? " is-danger" : "") + " is-" + ref.status;
      card.textContent = "";

      var head = el("div", "chat-action-head");
      var title = el("span", "chat-action-title");
      title.textContent = ref.title || "Действие";
      head.appendChild(title);
      if (ref.target_url) {
        var link = document.createElement("a");
        link.href = ref.target_url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.className = "chat-action-link";
        link.textContent = ref.target_link_label || "страница пользователя";
        head.appendChild(link);
      }
      card.appendChild(head);

      var summary = el("div", "chat-action-summary");
      summary.textContent = ref.summary || "";
      card.appendChild(summary);

      if (ref.details && ref.details.length) {
        var list = el("ul", "chat-action-details");
        ref.details.forEach(function (line) {
          var li = document.createElement("li");
          li.textContent = line;
          list.appendChild(li);
        });
        card.appendChild(list);
      }

      if (ref.status === "pending") {
        var input = null;
        if (ref.type_to_confirm) {
          input = document.createElement("input");
          input.type = "text";
          input.className = "input input-mono chat-action-input";
          input.placeholder = (ref.confirm_hint || "Для удаления введите логин: ") + ref.type_to_confirm;
          input.autocomplete = "off";
          card.appendChild(input);
        }
        var row = el("div", "chat-action-buttons");
        var ok = document.createElement("button");
        ok.type = "button";
        ok.className = "btn btn-sm " + (ref.danger ? "btn-danger" : "btn-primary");
        ok.textContent = "Подтвердить";
        var cancel = document.createElement("button");
        cancel.type = "button";
        cancel.className = "btn btn-sm btn-secondary";
        cancel.textContent = "Отмена";
        function sync() {
          ok.disabled = !!input && input.value.trim() !== ref.type_to_confirm;
        }
        if (input) {
          input.addEventListener("input", sync);
          sync();
        }
        function lock() {
          ok.disabled = true;
          cancel.disabled = true;
        }
        ok.addEventListener("click", function () {
          post(ref.confirm_url, { confirm_text: input ? input.value.trim() : "" }, lock);
        });
        cancel.addEventListener("click", function () {
          post(ref.cancel_url, {}, lock);
        });
        row.appendChild(ok);
        row.appendChild(cancel);
        card.appendChild(row);
        if (ref.expires) {
          var exp = el("div", "chat-action-note");
          exp.textContent = "Действует до " + ref.expires;
          card.appendChild(exp);
        }
      } else {
        var status = el("div", "chat-action-status");
        status.textContent = (ACTION_STATUS[ref.status] || ref.status) + (ref.result ? ": " + ref.result : "");
        card.appendChild(status);
      }
      if (note) {
        var warn = el("div", "chat-action-note chat-action-error");
        warn.textContent = note;
        card.appendChild(warn);
      }
    }

    render(initial);
    // Карточка в старом сообщении могла уже быть подтверждена, отменена или просрочена.
    if (initial.state_url && (initial.status === "pending" || initial.status === "running")) {
      fetch(initial.state_url, { credentials: "same-origin", headers: { Accept: "application/json" } })
        .then(function (r) { return r.ok ? r.json() : null; })
        .then(function (data) { if (data && data.card) render(data.card); })
        .catch(function () {});
    }
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
  // opts: {id} — id сообщения в БД (для реплик, которые уже сохранены на сервере), {time} — «ЧЧ:ММ».
  function renderTurn(role, text, meta, refs, images, opts) {
    opts = opts || {};
    var hint = document.getElementById("chat-empty");
    if (hint) hint.remove();

    var turn = el("div", "chat-turn chat-turn-" + role);
    if (opts.id) {
      turn.dataset.msgId = String(opts.id);
      seen[opts.id] = true;
    }

    var head = el("div", "chat-turn-head");
    var roleEl = el("span", "chat-turn-role");
    roleEl.textContent = ROLE_LABELS[role] || role;
    var timeEl = el("span", "chat-turn-time");
    timeEl.textContent = opts.time || nowHM();
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
        img.src = item.thumb_url || item.url;
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

    // «Печатает…» всегда остаётся последней репликой: всё, что приходит, пока модель отвечает
    // (например, результат анализа из очереди), встаёт перед ней.
    var typing = log.querySelector(".chat-turn-pending");
    if (typing) log.insertBefore(turn, typing);
    else log.appendChild(turn);
    log.scrollTop = log.scrollHeight;
    return turn;
  }

  // Убрать реплику из лога (её сообщение не сохранилось / удалено на сервере).
  function forgetTurn(node) {
    if (node.dataset && node.dataset.msgId) delete seen[node.dataset.msgId];
    node.remove();
  }

  // Лог опустел (например, первое сообщение не отправилось) — возвращаем подсказку «чат пуст».
  function ensureEmptyHint() {
    if (emptyHint && !emptyHint.parentNode && !log.querySelector(".chat-turn")) log.appendChild(emptyHint);
  }

  function noteExistingIds() {
    var nodes = log.querySelectorAll(".chat-turn[data-msg-id]");
    for (var i = 0; i < nodes.length; i += 1) {
      seen[parseInt(nodes[i].dataset.msgId, 10)] = true;
    }
  }

  function maxSeenId() {
    var max = 0;
    Object.keys(seen).forEach(function (key) {
      var id = parseInt(key, 10);
      if (id > max) max = id;
    });
    return max;
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
          if (m.id && seen[m.id]) return;
          renderTurn("assistant", m.reply || "", messageMeta(m), m.refs, null, { id: m.id });
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

  // ---------- ход модели: отправка, ожидание ответа, разбор ошибок ----------
  //
  // turn — состояние одного хода: {since, text, attachments, userTurn, dotsTurn}
  //   since       — id последнего сообщения чата до этого хода (по нему запрашиваем состояние);
  //   text/attachments — что вернуть в поле ввода, если ход не удался;
  //   userTurn    — реплика пользователя, нарисованная сразу при отправке (ещё без id);
  //   dotsTurn    — реплика «печатает…».

  var STATE_POLL_MS = 2000;
  var STATE_MAX_FAILS = 4; // столько опросов подряд без связи — считаем, что сервер недоступен
  var TURN_FAILED_TEXT = "Не удалось получить ответ от модели — сообщение возвращено в поле ввода.";

  function messageMeta(m) {
    return m.backend ? m.backend + (m.model ? " · " + m.model : "") : "";
  }

  function endTurn() {
    setSending(false);
    input.focus();
  }

  function dropDots(turn) {
    if (turn.dotsTurn) {
      turn.dotsTurn.remove();
      turn.dotsTurn = null;
    }
  }

  // Ответ получен.
  function finishTurnOk(turn, data) {
    dropDots(turn);
    lsRemove(outboxKey);
    updateSidebarTitle(data.title);
    pendingJobs = data.pending || 0;
    schedulePoll();
    endTurn();
  }

  // Ход не удался: всё, что пользователь отправил, возвращается в строку ввода.
  function failTurn(turn, text) {
    dropDots(turn);
    if (turn.userTurn) forgetTurn(turn.userTurn);
    ensureEmptyHint();
    var outbox = readOutbox();
    lsRemove(outboxKey);
    restoreComposer(turn.text || (outbox ? outbox.text : ""), turn.attachments);
    showError(text);
    endTurn();
  }

  // Сверяет лог с тем, что сейчас на сервере (сообщения с id > turn.since): дорисовывает новые
  // и убирает реплики, которых на сервере уже нет (сообщение неудавшегося хода откатывается).
  function applyServerMessages(messages, turn) {
    var serverIds = {};
    messages.forEach(function (m) {
      serverIds[m.id] = true;
    });

    var nodes = log.querySelectorAll(".chat-turn[data-msg-id]");
    Array.prototype.forEach.call(nodes, function (node) {
      var id = parseInt(node.dataset.msgId, 10);
      if (id > turn.since && !serverIds[id]) forgetTurn(node);
    });

    messages.forEach(function (m) {
      if (seen[m.id]) return;
      // Реплика пользователя, нарисованная при отправке, — это то же самое сообщение: привязываем id.
      if (m.role === "user" && turn.userTurn && !turn.userTurn.dataset.msgId) {
        turn.userTurn.dataset.msgId = String(m.id);
        seen[m.id] = true;
        return;
      }
      renderTurn(m.role, m.content, messageMeta(m), m.refs, m.images, { id: m.id, time: m.time });
    });
    ensureEmptyHint();
  }

  // Опрос завершён (модель больше не отвечает): успех это или неудача?
  function settleTurn(turn, data) {
    var messages = data.messages || [];
    var userMsg = null;
    for (var i = 0; i < messages.length; i += 1) {
      if (messages[i].role === "user") {
        userMsg = messages[i];
        break;
      }
    }

    if (!userMsg) {
      // Сообщения пользователя на сервере нет — ход не удался и был откатан.
      failTurn(turn, TURN_FAILED_TEXT);
      return;
    }
    var answered = messages.some(function (m) {
      return m.role === "assistant" && m.id > userMsg.id;
    });
    if (!answered) {
      // Ход оборвался, не оставив ответа (например, сервер перезапустили): сообщение осталось в чате.
      dropDots(turn);
      lsRemove(outboxKey);
      showError("Ответ модели не получен (возможно, сервер был перезапущен). Отправьте сообщение ещё раз.");
      endTurn();
      return;
    }
    finishTurnOk(turn, data);
  }

  // Ждём ответа модели, спрашивая у сервера. Так страница переживает перезагрузку и обрыв связи
  // посреди хода: сервер доделывает ход сам, а ответ появляется в чате, когда готов.
  function resumeTurn(turn) {
    var fails = 0;
    setSending(true);
    if (!turn.dotsTurn) turn.dotsTurn = renderPending();

    function tick() {
      fetch(stateUrl + "?since=" + encodeURIComponent(turn.since), {
        credentials: "same-origin",
        headers: { Accept: "application/json" },
        cache: "no-store",
      })
        .then(function (r) {
          if (!r.ok) throw new Error("state " + r.status);
          return r.json();
        })
        .then(
          function (data) {
            fails = 0;
            updateSidebarTitle(data.title);
            applyServerMessages(data.messages || [], turn);
            if (data.busy) {
              window.setTimeout(tick, STATE_POLL_MS);
              return;
            }
            settleTurn(turn, data);
          },
          function () {
            fails += 1;
            if (fails >= STATE_MAX_FAILS) {
              failTurn(turn, "Не удалось связаться с сервером. Проверьте соединение и повторите.");
              return;
            }
            window.setTimeout(tick, STATE_POLL_MS * fails);
          }
        );
    }
    tick();
  }

  function sendMessage(message, attachments) {
    var turn = {
      since: maxSeenId(),
      text: message,
      attachments: attachments,
      userTurn: null,
      dotsTurn: null,
    };
    // Текст уже ушёл из поля ввода, но пока ответа нет — держим его в «исходящем»:
    // если ход не удастся, а страницу к тому моменту перезагрузят, текст вернётся в поле.
    writeOutbox(message, turn.since);

    turn.userTurn = renderTurn(
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
    turn.dotsTurn = renderPending();

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
      .then(
        function (result) {
          if (!result.ok) {
            failTurn(turn, (result.data && result.data.error) || "Не удалось получить ответ.");
            return;
          }
          var data = result.data || {};
          dropDots(turn);
          if (data.user_message_id) {
            turn.userTurn.dataset.msgId = String(data.user_message_id);
            seen[data.user_message_id] = true;
          }
          renderTurn("assistant", data.reply || "", messageMeta(data), data.refs, null, {
            id: data.assistant_message_id,
          });
          finishTurnOk(turn, data);
        },
        function () {
          // Связь оборвалась, но сервер мог принять сообщение и продолжает отвечать — спрашиваем у него.
          resumeTurn(turn);
        }
      );
  }

  // Страницу открыли/перезагрузили: если модель ещё отвечала или ход закончился в её отсутствие —
  // разбираемся, чем он кончился.
  function resumeAfterLoad() {
    var outbox = readOutbox();
    if (serverBusy) {
      resumeTurn({
        since: busySince,
        text: outbox ? outbox.text : "",
        attachments: [],
        userTurn: null,
        dotsTurn: null,
      });
      return;
    }
    if (!outbox) return;

    lsRemove(outboxKey);
    if (Date.now() - (outbox.ts || 0) > OUTBOX_MAX_AGE_MS) return;
    var userNodes = log.querySelectorAll(".chat-turn-user[data-msg-id]");
    for (var i = 0; i < userNodes.length; i += 1) {
      if (parseInt(userNodes[i].dataset.msgId, 10) > (outbox.since || 0)) return; // сообщение дошло
    }
    if (outbox.text) {
      restoreComposer(outbox.text, []);
      showError("Не удалось отправить прошлое сообщение — текст возвращён в поле ввода.");
    }
  }

  form.addEventListener("submit", function (e) {
    e.preventDefault();
    if (sending) return;
    var message = input.value.trim();
    if (!message && !pending.length) return;
    var attachments = pending.slice();
    rememberSent(message);
    input.value = "";
    autoGrow();
    lsRemove(draftKey);
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

  input.addEventListener("input", function () {
    autoGrow();
    saveDraft();
    if (!input.value) {
      // Поле очистили вручную — выходим из режима листания истории.
      histPos = sentHistory.length;
      histDraft = "";
    }
  });
  input.addEventListener("keydown", function (e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      form.requestSubmit ? form.requestSubmit() : form.dispatchEvent(new Event("submit", { cancelable: true }));
      return;
    }
    // ↑/↓ — листать историю введённого, как в терминале (см. historyStep).
    if (
      (e.key === "ArrowUp" || e.key === "ArrowDown") &&
      !e.shiftKey && !e.ctrlKey && !e.altKey && !e.metaKey && !e.isComposing
    ) {
      if (historyStep(e.key === "ArrowUp" ? -1 : 1)) e.preventDefault();
    }
  });

  noteExistingIds();
  seedHistoryFromLog(); // до renderExistingTurns: он заменяет исходный текст реплик разметкой
  renderExistingTurns();
  log.scrollTop = log.scrollHeight;

  // Черновик: то, что печатали и не отправили, не пропадает при перезагрузке.
  var savedDraft = lsGet(draftKey);
  if (savedDraft && !input.value) {
    input.value = savedDraft;
    autoGrow();
    input.setSelectionRange(input.value.length, input.value.length);
  }

  resumeAfterLoad();
  if (pendingJobs > 0) pollPending(); // страницу открыли/перезагрузили, пока анализ шёл — сразу проверяем
})();