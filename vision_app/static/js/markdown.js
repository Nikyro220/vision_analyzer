/* Минимальный Markdown-рендерер для чата.
 *
 * Зачем свой, а не marked.js/DOMPurify с CDN: весь остальной фронтенд
 * приложения самодостаточен (без внешних <script src> на CDN, см. base.html),
 * сервер анализа обычно работает в закрытом контуре без интернета. Тащить
 * внешнюю зависимость ради жирного набора фич markdown в чате — лишний риск
 * (CDN недоступен/заблокирован) и лишняя поверхность для XSS (нужно ещё
 * подключать санитайзер). Здесь входной текст СНАЧАЛА полностью экранируется
 * (escapeHtml), и только потом в него точечно вставляются заранее известные
 * безопасные теги — то есть непосредственно вставить произвольный HTML через
 * содержимое сообщения (своё или от модели) нельзя в принципе.
 *
 * Поддержано: заголовки (# .. ######), bold (** или __), italic (* или _),
 * inline code (обратные кавычки), блоки кода в тройных обратных кавычках
 * (с необязательным языком),
 * блок-цитаты (> ...), маркированные (-, *, +) и нумерованные (1.) списки
 * (в т.ч. вложенные по отступу), ссылки [текст](url) — только http(s)/mailto,
 * горизонтальная линия (---), переносы строк/абзацы.
 *
 * Использование: window.renderMarkdownToHtml(rawText) -> HTML-строка,
 * пригодная для innerHTML.
 */
(function (global) {
  "use strict";

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  // Разрешённые схемы ссылок — никаких javascript:/data: и т.п.
  function safeHref(url) {
    var trimmed = (url || "").trim();
    if (/^(https?:|mailto:)/i.test(trimmed)) return escapeHtml(trimmed);
    if (/^[^:]+$/.test(trimmed)) return escapeHtml(trimmed); // относительная ссылка/якорь
    return "#";
  }

  // ---------- инлайн-разметка ----------
  // Принимает СЫРОЙ (неэкранированный) текст с одной строки/абзаца и
  // возвращает безопасный HTML: экранирование — первый шаг, до разбора
  // самой markdown-разметки, так что пользовательский текст никогда не
  // становится HTML-тегами напрямую — только через явные замены ниже.
  function renderInline(rawText) {
    var text = escapeHtml(rawText);

    // 1) inline code — забираем раньше остального, чтобы `**` внутри кода не превращалось в bold
    var codeSpans = [];
    text = text.replace(/`([^`\n]+)`/g, function (_, code) {
      codeSpans.push(code);
      return "\u0000CODE" + (codeSpans.length - 1) + "\u0000";
    });

    // 2) ссылки [текст](url)
    text = text.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, function (_, label, url) {
      return '<a href="' + safeHref(url) + '" target="_blank" rel="noopener noreferrer">' + label + "</a>";
    });

    // 3) bold (** или __), затем italic (* или _) — bold раньше, чтобы ** не съелось italic-правилом
    text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    text = text.replace(/__([^_]+)__/g, "<strong>$1</strong>");
    text = text.replace(/\*([^*\n]+)\*/g, "<em>$1</em>");
    text = text.replace(/(^|[^\w])_([^_\n]+)_(?!\w)/g, "$1<em>$2</em>");

    // 4) вернуть inline code на место
    text = text.replace(/\u0000CODE(\d+)\u0000/g, function (_, idx) {
      return "<code>" + codeSpans[Number(idx)] + "</code>";
    });

    return text;
  }

  // ---------- списки (группа последовательных строк-пунктов) ----------
  function renderListBlock(lines) {
    // lines: [{indent, ordered, content}]
    var html = "";
    var stack = []; // {ordered, indent}

    function openList(ordered, indent) {
      html += ordered ? "<ol>" : "<ul>";
      stack.push({ ordered: ordered, indent: indent });
    }
    function closeList() {
      var top = stack.pop();
      html += top.ordered ? "</ol>" : "</ul>";
    }

    lines.forEach(function (item) {
      while (stack.length && item.indent < stack[stack.length - 1].indent) closeList();
      if (!stack.length || item.indent > stack[stack.length - 1].indent) {
        openList(item.ordered, item.indent);
      } else if (stack[stack.length - 1].ordered !== item.ordered) {
        closeList();
        openList(item.ordered, item.indent);
      }
      html += "<li>" + renderInline(item.content) + "</li>";
    });
    while (stack.length) closeList();
    return html;
  }

  function renderMarkdownToHtml(raw) {
    var lines = String(raw == null ? "" : raw).replace(/\r\n?/g, "\n").split("\n");

    var out = [];
    var i = 0;
    var paragraphBuf = [];
    var listBuf = [];
    var quoteBuf = [];

    function flushParagraph() {
      if (!paragraphBuf.length) return;
      out.push("<p>" + renderInline(paragraphBuf.join(" ")) + "</p>");
      paragraphBuf = [];
    }
    function flushList() {
      if (!listBuf.length) return;
      out.push(renderListBlock(listBuf));
      listBuf = [];
    }
    function flushQuote() {
      if (!quoteBuf.length) return;
      out.push("<blockquote>" + renderInline(quoteBuf.join(" ")) + "</blockquote>");
      quoteBuf = [];
    }
    function flushAll() {
      flushParagraph();
      flushList();
      flushQuote();
    }

    while (i < lines.length) {
      var line = lines[i];

      // fenced code block ```lang
      var fence = line.match(/^```(\w*)\s*$/);
      if (fence) {
        flushAll();
        var codeLines = [];
        i += 1;
        while (i < lines.length && !/^```\s*$/.test(lines[i])) {
          codeLines.push(lines[i]);
          i += 1;
        }
        i += 1; // пропустить закрывающую ```
        var langClass = fence[1] ? ' class="lang-' + escapeHtml(fence[1]) + '"' : "";
        out.push("<pre><code" + langClass + ">" + codeLines.map(escapeHtml).join("\n") + "</code></pre>");
        continue;
      }

      // горизонтальная линия
      if (/^(---|\*\*\*|___)\s*$/.test(line)) {
        flushAll();
        out.push("<hr>");
        i += 1;
        continue;
      }

      // заголовки
      var heading = line.match(/^(#{1,6})\s+(.+)$/);
      if (heading) {
        flushAll();
        var level = heading[1].length;
        out.push("<h" + level + ">" + renderInline(heading[2].trim()) + "</h" + level + ">");
        i += 1;
        continue;
      }

      // блок-цитата
      var quote = line.match(/^>\s?(.*)$/);
      if (quote) {
        flushParagraph();
        flushList();
        quoteBuf.push(quote[1]);
        i += 1;
        continue;
      }
      flushQuote();

      // элементы списка (маркированные/нумерованные, с отступом)
      var listItem = line.match(/^(\s*)([-*+]|\d+\.)\s+(.+)$/);
      if (listItem) {
        flushParagraph();
        var indent = listItem[1].replace(/\t/g, "    ").length;
        var ordered = /\d+\./.test(listItem[2]);
        listBuf.push({ indent: indent, ordered: ordered, content: listItem[3] });
        i += 1;
        continue;
      }
      flushList();

      // пустая строка — разделитель абзацев
      if (/^\s*$/.test(line)) {
        flushParagraph();
        i += 1;
        continue;
      }

      // обычная строка текста — копим в абзац
      paragraphBuf.push(line.trim());
      i += 1;
    }

    flushAll();
    return out.join("\n");
  }

  global.renderMarkdownToHtml = renderMarkdownToHtml;
})(window);