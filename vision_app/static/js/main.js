document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("upload-form");
  const submitBtn = document.getElementById("submit-btn");

  if (form && submitBtn) {
    form.addEventListener("submit", () => {
      submitBtn.disabled = true;
      submitBtn.textContent = "Анализируем…";
    });
  }
});
