/* Единое поле «ссылки или изображения» на странице «Новый анализ».
 * Ссылки пишутся/вставляются в текстовое поле; изображения добавляются
 * вставкой из буфера (Ctrl+V), перетаскиванием на страницу или кнопкой.
 * Сам список файлов, подписи и отправка живут в queue.js (window.VisionIntake). */
(function () {
  "use strict";

  var box = document.getElementById("dropzone");
  var input = document.getElementById("id_image");
  var links = document.getElementById("id_links");
  var pickBtn = document.getElementById("pick-files");
  if (!box || !input || !links || !window.VisionIntake) return;

  function add(files) {
    if (files && files.length) window.VisionIntake.addFiles(files);
  }

  function hasFiles(dt) {
    return !!dt && Array.prototype.indexOf.call(dt.types || [], "Files") !== -1;
  }

  // ---------- кнопка «Выбрать изображения» ----------
  // Нативный выбор файлов заменяет прежний список — запоминаем его и склеиваем с новым.
  var before = null;
  pickBtn.addEventListener("click", function () {
    before = Array.prototype.slice.call(input.files);
    input.click();
  });
  input.addEventListener("change", function (e) {
    if (before === null) return;
    var fresh = Array.prototype.slice.call(input.files);
    var restore = new DataTransfer();
    before.forEach(function (f) { restore.items.add(f); });
    input.files = restore.files; // вернуть прежний выбор, чтобы подписи к нему сохранились
    before = null;
    e.stopPropagation();
    add(fresh);
  });

  // ---------- вставка из буфера ----------
  // Скриншот или скопированная картинка -> файл. Если в буфере есть и текст, вставляется текст.
  document.addEventListener("paste", function (e) {
    var cd = e.clipboardData;
    if (!cd || !cd.files || !cd.files.length) return;
    var target = e.target;
    var inOtherField = target && target !== links && target.matches && target.matches("input, textarea, [contenteditable]");
    if (inOtherField || cd.getData("text/plain")) return;
    var images = Array.prototype.filter.call(cd.files, function (f) { return /^image\//.test(f.type); });
    if (!images.length) return;
    e.preventDefault();
    add(images);
  });

  // ---------- перетаскивание: файлы добавляются, ссылка из браузера попадает в текст ----------
  var depth = 0;
  window.addEventListener("dragenter", function (e) {
    if (!hasFiles(e.dataTransfer)) return;
    depth += 1;
    box.classList.add("is-dragover");
  });
  window.addEventListener("dragleave", function (e) {
    if (!hasFiles(e.dataTransfer)) return;
    depth = Math.max(depth - 1, 0);
    if (!depth) box.classList.remove("is-dragover");
  });
  window.addEventListener("dragover", function (e) {
    // без этого браузер откроет файл вместо того, чтобы отдать его странице
    if (hasFiles(e.dataTransfer) || box.contains(e.target)) e.preventDefault();
  });
  window.addEventListener("drop", function (e) {
    var dt = e.dataTransfer;
    var overBox = box.contains(e.target);
    depth = 0;
    box.classList.remove("is-dragover");
    if (hasFiles(dt)) {
      e.preventDefault();
      add(dt.files);
    } else if (overBox) {
      var text = dt.getData("text/uri-list") || dt.getData("text/plain");
      if (!text) return;
      e.preventDefault();
      var value = links.value.replace(/\s+$/, "");
      links.value = (value ? value + "\n" : "") + text.trim() + "\n";
    }
  });
})();
