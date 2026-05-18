const form = document.getElementById("upload-form");
const imageInput = document.getElementById("image");
const preview = document.getElementById("preview");
const previewSection = document.getElementById("preview-section");
const result = document.getElementById("result");

imageInput.addEventListener("change", () => {
  const file = imageInput.files[0];
  if (!file) return;

  preview.src = URL.createObjectURL(file);
  previewSection.classList.remove("hidden");
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = imageInput.files[0];

  if (!file) {
    showError("Selecciona una imagen antes de analizar.");
    return;
  }

  const data = new FormData();
  data.append("image", file);

  result.classList.remove("hidden");
  result.innerHTML = "<p>Analizando imagen...</p>";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: data,
    });

    const payload = await response.json();
    if (!response.ok) {
      showError(payload.error || "No se pudo procesar la imagen.");
      return;
    }

    const top5 = payload.top5
      .map((item, idx) => `<li>${idx + 1}. ${item.clase} - ${item.confianza}%</li>`)
      .join("");

    result.innerHTML = `
      <h2>Resultado</h2>
      <p><strong>Clase:</strong> ${payload.clase}</p>
      <p><strong>Confianza:</strong> ${payload.confianza}%</p>
      <p><strong>Top 5 predicciones:</strong></p>
      <ul>${top5}</ul>
      <p><strong>Dispositivo:</strong> ${payload.device}</p>
    `;
  } catch {
    showError("Error de conexion con el servidor.");
  }
});

function showError(message) {
  result.classList.remove("hidden");
  result.innerHTML = `<p class="error">${message}</p>`;
}
