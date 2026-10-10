// app.js — navigation between "pages" and the Identify flow.

// ---------- Navigation ----------
const pages = document.querySelectorAll(".page");
const navLinks = document.querySelectorAll("[data-nav]");
const navLinksList = document.getElementById("navLinks");
const navToggle = document.getElementById("navToggle");
const VALID_PAGES = ["home", "about", "how-it-works", "breeds", "ai-model", "identify", "history"];

function showPage(id) {
  pages.forEach((p) => p.classList.toggle("active", p.id === `page-${id}`));
  navLinks.forEach((l) => l.classList.toggle("active", l.dataset.nav === id));
  navLinksList.classList.remove("open");
  window.scrollTo({ top: 0, behavior: "instant" in window ? "instant" : "auto" });
}

// Event delegation so this also works for [data-nav] links added later
// (e.g. inside the History page's empty state, or the breed modal).
document.addEventListener("click", (e) => {
  const target = e.target.closest("[data-nav]");
  if (!target) return;
  e.preventDefault();
  const id = target.dataset.nav;
  showPage(id);
  window.history.replaceState(null, "", `#${id}`);
  if (id === "history" && typeof loadHistory === "function") loadHistory();
});

navToggle.addEventListener("click", () => navLinksList.classList.toggle("open"));

const initial = (location.hash || "#home").replace("#", "");
showPage(VALID_PAGES.includes(initial) ? initial : "home");
if (initial === "history" && typeof loadHistory === "function") loadHistory();

// ---------- Hero patch grid (decorative, ties visual to ViT patching) ----------
const patchGrid = document.getElementById("patchGrid");
if (patchGrid) {
  const hotIndices = new Set([12, 13, 20, 21, 28, 29, 36, 44]);
  for (let i = 0; i < 80; i++) {
    const span = document.createElement("span");
    if (hotIndices.has(i)) span.classList.add("hot");
    patchGrid.appendChild(span);
  }
}

// ---------- Breeds page ----------
let breedNames = [];

async function loadBreeds() {
  const grid = document.getElementById("breedsGrid");
  try {
    const data = await fetchBreeds();
    breedNames = data.breeds;
    grid.innerHTML = breedNames
      .map(
        (name) => `
        <div class="breed-card" data-breed="${name}">
          <div class="breed-thumb">${name.slice(0, 2).toUpperCase()}</div>
          <div class="breed-body">
            <h3 style="font-size:1.05rem;">${name}</h3>
            <div class="origin">Origin — details coming soon</div>
            <span class="breed-link">View Details →</span>
          </div>
        </div>`
      )
      .join("");
    grid.querySelectorAll(".breed-card").forEach((card) => {
      card.addEventListener("click", () => openBreedModal(card.dataset.breed));
    });
  } catch (err) {
    grid.innerHTML = `<div class="error-box">${err.message}</div>`;
  }
}
loadBreeds();

// ---------- Breed modal ----------
const breedModalBackdrop = document.getElementById("breedModalBackdrop");
const breedModalName = document.getElementById("breedModalName");

function openBreedModal(name) {
  breedModalName.textContent = name;
  breedModalBackdrop.classList.add("open");
}
document.getElementById("breedModalClose").addEventListener("click", () => {
  breedModalBackdrop.classList.remove("open");
});
breedModalBackdrop.addEventListener("click", (e) => {
  if (e.target === breedModalBackdrop) breedModalBackdrop.classList.remove("open");
});

// ---------- Identify flow ----------
const dropZone = document.getElementById("dropZone");
const fileInput = document.getElementById("fileInput");
const browseBtn = document.getElementById("browseBtn");
const uploadStep = document.getElementById("uploadStep");
const previewStep = document.getElementById("previewStep");
const loadingStep = document.getElementById("loadingStep");
const resultStep = document.getElementById("resultStep");
const uploadError = document.getElementById("uploadError");
const predictError = document.getElementById("predictError");

let selectedFile = null;

function resetIdentifyFlow() {
  selectedFile = null;
  fileInput.value = "";
  uploadStep.style.display = "block";
  previewStep.style.display = "none";
  loadingStep.style.display = "none";
  resultStep.style.display = "none";
  uploadError.innerHTML = "";
  predictError.innerHTML = "";
}

function showUploadError(msg) {
  uploadError.innerHTML = `<div class="error-box" style="margin-top:16px;">${msg}</div>`;
}

const ALLOWED_TYPES = ["image/jpeg", "image/png", "image/webp"];
const MAX_SIZE = 10 * 1024 * 1024;

function handleFile(file) {
  uploadError.innerHTML = "";
  if (!file) {
    showUploadError("Please upload an image to continue.");
    return;
  }
  if (!ALLOWED_TYPES.includes(file.type)) {
    showUploadError("Please upload a supported image format.");
    return;
  }
  if (file.size > MAX_SIZE) {
    showUploadError("Please choose a smaller image.");
    return;
  }
  selectedFile = file;
  const url = URL.createObjectURL(file);
  document.getElementById("previewImg").src = url;
  document.getElementById("previewName").textContent = file.name;
  document.getElementById("previewSize").textContent = `${(file.size / 1024).toFixed(0)} KB`;
  uploadStep.style.display = "none";
  previewStep.style.display = "block";
}

browseBtn.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => handleFile(e.target.files[0]));

["dragenter", "dragover"].forEach((evt) =>
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.add("dragover");
  })
);
["dragleave", "drop"].forEach((evt) =>
  dropZone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropZone.classList.remove("dragover");
  })
);
dropZone.addEventListener("drop", (e) => {
  const file = e.dataTransfer.files[0];
  handleFile(file);
});

document.getElementById("changeBtn").addEventListener("click", resetIdentifyFlow);

document.getElementById("analyzeBtn").addEventListener("click", async () => {
  if (!selectedFile) return;
  previewStep.style.display = "none";
  loadingStep.style.display = "block";
  predictError.innerHTML = "";

  const s1 = document.getElementById("loadStep1");
  const s2 = document.getElementById("loadStep2");
  const s3 = document.getElementById("loadStep3");
  s1.className = "done";
  setTimeout(() => (s2.className = "done"), 300);
  setTimeout(() => (s3.className = "active"), 500);

  try {
    const data = await predictImage(selectedFile);
    renderResult(data);
    loadingStep.style.display = "none";
    resultStep.style.display = "block";
  } catch (err) {
    loadingStep.style.display = "none";
    previewStep.style.display = "block";
    predictError.innerHTML = `<div class="error-box" style="margin-top:16px;">${err.message}</div>`;
  }
});

let lastPredictedBreed = null;

let lastResultData = null;

function renderResult(data) {
  lastResultData = data;
  const originalImgSrc = document.getElementById("previewImg").src;
  document.getElementById("resultImg").src = originalImgSrc;

  const attentionBtn = document.getElementById("attentionToggleBtn");
  if (data.attention_map) {
    attentionBtn.style.display = "inline-block";
    attentionBtn.textContent = "View Attention Map";
    attentionBtn.dataset.showingAttention = "false";
    attentionBtn.onclick = () => {
      const showingAttention = attentionBtn.dataset.showingAttention === "true";
      document.getElementById("resultImg").src = showingAttention ? originalImgSrc : data.attention_map;
      attentionBtn.dataset.showingAttention = showingAttention ? "false" : "true";
      attentionBtn.textContent = showingAttention ? "View Attention Map" : "View Original Image";
    };
  } else {
    attentionBtn.style.display = "none";
  }

  const resultBreedEl = document.getElementById("resultBreed");
  const resultConfidenceEl = document.getElementById("resultConfidence");
  const pct = (data.confidence * 100).toFixed(1);

  if (data.low_confidence) {
    resultBreedEl.textContent = "Breed Not Recognized";
    resultConfidenceEl.textContent = `Closest match: ${data.predicted_breed} — ${pct}% confidence`;
  } else {
    resultBreedEl.textContent = data.predicted_breed;
    resultConfidenceEl.textContent = `Model confidence: ${pct}%`;
  }
  lastPredictedBreed = data.predicted_breed;
  document.getElementById("viewBreedBtn").textContent = data.low_confidence
    ? "View Closest Match Details"
    : "View Breed Details";

  const saveNote = document.getElementById("saveStatusNote");
  saveNote.textContent = data.logged_in
    ? "Saved to your history."
    : "Log in to save this to your history.";

  const notice = document.getElementById("lowConfidenceNotice");
  if (data.low_confidence) {
    const threshPct = (data.low_confidence_threshold * 100).toFixed(0);
    notice.innerHTML = `
      <div class="warning-box">
        <strong>This image doesn't clearly match any of the 17 trained breeds</strong>
        Confidence is below ${threshPct}%. The closest guess is shown below for
        reference, but it should not be treated as a real identification.
      </div>`;
  } else {
    notice.innerHTML = "";
  }

  const predList = document.getElementById("predList");
  predList.innerHTML = data.top_predictions
    .map((p, i) => {
      const pct = (p.confidence * 100).toFixed(1);
      return `
        <div class="pred-row">
          <div class="pred-rank">${i + 1}</div>
          <div>
            <div class="pred-name">${p.breed}</div>
            <div class="pred-bar-track"><div class="pred-bar-fill" style="width:${pct}%"></div></div>
          </div>
          <div class="pred-pct">${pct}%</div>
        </div>`;
    })
    .join("");
}

document.getElementById("viewBreedBtn").addEventListener("click", () => {
  if (lastPredictedBreed) openBreedModal(lastPredictedBreed);
});

document.getElementById("analyzeAnotherBtn").addEventListener("click", resetIdentifyFlow);

// ---------- Copy result as text ----------
function buildResultText(data) {
  const pct = (data.confidence * 100).toFixed(1);
  const lines = ["CattleVision AI — Prediction Result", ""];

  if (data.low_confidence) {
    lines.push(
      "Result: Breed Not Recognized",
      `Closest match: ${data.predicted_breed} — ${pct}% confidence`,
      "Note: below the model's confidence threshold — this image doesn't clearly match any of the 17 trained breeds."
    );
  } else {
    lines.push(`Predicted Breed: ${data.predicted_breed}`, `Model Confidence: ${pct}%`);
  }

  lines.push("", "Top Predictions:");
  data.top_predictions.forEach((p, i) => {
    lines.push(`${i + 1}. ${p.breed} — ${(p.confidence * 100).toFixed(1)}%`);
  });
  lines.push("", "Generated by CattleVision AI (ViT-Base Cattle Breed Recognition)");
  return lines.join("\n");
}

const copyResultBtn = document.getElementById("copyResultBtn");
copyResultBtn.addEventListener("click", async () => {
  if (!lastResultData) return;
  const text = buildResultText(lastResultData);
  const originalLabel = copyResultBtn.textContent;
  try {
    await navigator.clipboard.writeText(text);
    copyResultBtn.textContent = "Copied!";
  } catch (err) {
    copyResultBtn.textContent = "Copy failed";
  }
  setTimeout(() => (copyResultBtn.textContent = originalLabel), 1800);
});

// ---------- Download result as a shareable PNG card ----------
function wrapCanvasText(ctx, text, x, y, maxWidth, lineHeight) {
  const words = text.split(" ");
  let line = "";
  let curY = y;
  words.forEach((word) => {
    const test = line + word + " ";
    if (ctx.measureText(test).width > maxWidth && line !== "") {
      ctx.fillText(line.trim(), x, curY);
      line = word + " ";
      curY += lineHeight;
    } else {
      line = test;
    }
  });
  ctx.fillText(line.trim(), x, curY);
  return curY + lineHeight;
}

async function drawResultCard(data) {
  const W = 900;
  const H = 1180;
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d");

  const bg = "#F5F2E9";
  const pastureDeep = "#17301F";
  const pasture = "#23422F";
  const hide = "#A8672F";
  const line = "#DCD5C2";
  const ink = "#1C2A1E";
  const inkSoft = "#4B5750";

  if (document.fonts && document.fonts.ready) {
    try { await document.fonts.ready; } catch (_) {}
  }

  // background
  ctx.fillStyle = bg;
  ctx.fillRect(0, 0, W, H);

  // header bar
  ctx.fillStyle = pastureDeep;
  ctx.fillRect(0, 0, W, 90);
  ctx.fillStyle = "#F5F2E9";
  ctx.font = "600 30px 'Fraunces', Georgia, serif";
  ctx.textBaseline = "middle";
  ctx.fillText("CattleVision AI", 40, 45);
  ctx.font = "500 15px 'IBM Plex Mono', monospace";
  ctx.fillStyle = "#C7C1AC";
  ctx.textAlign = "right";
  ctx.fillText("Prediction Result", W - 40, 45);
  ctx.textAlign = "left";

  // uploaded image — letterboxed to fit fully, never cropped, regardless
  // of the photo's original aspect ratio.
  const imgY = 120;
  const imgH = 380;
  await new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      const targetW = W - 80;
      const scale = Math.min(targetW / img.width, imgH / img.height);
      const drawW = img.width * scale;
      const drawH = img.height * scale;
      const dx = 40 + (targetW - drawW) / 2;
      const dy = imgY + (imgH - drawH) / 2;
      ctx.fillStyle = pastureDeep;
      ctx.fillRect(40, imgY, targetW, imgH);
      ctx.drawImage(img, dx, dy, drawW, drawH);
      resolve();
    };
    img.onerror = resolve;
    img.src = document.getElementById("resultImg").src;
  });

  // border around image
  ctx.strokeStyle = line;
  ctx.lineWidth = 2;
  ctx.strokeRect(40, imgY, W - 80, imgH);

  let y = imgY + imgH + 55;

  if (data.low_confidence) {
    ctx.fillStyle = "#E8D2B8";
    ctx.fillRect(40, y - 30, W - 80, 60);
    ctx.fillStyle = "#6B3D14";
    ctx.font = "600 15px 'IBM Plex Sans', sans-serif";
    ctx.fillText("Below confidence threshold — not a recognized breed", 60, y);
    y += 55;
  }

  ctx.fillStyle = inkSoft;
  ctx.font = "500 15px 'IBM Plex Sans', sans-serif";
  ctx.fillText(data.low_confidence ? "RESULT" : "PREDICTED BREED", 40, y);
  y += 46;

  ctx.fillStyle = pastureDeep;
  ctx.font = "500 52px 'Fraunces', Georgia, serif";
  ctx.fillText(data.low_confidence ? "Breed Not Recognized" : data.predicted_breed, 40, y);
  y += 44;

  ctx.fillStyle = hide;
  ctx.font = "500 22px 'IBM Plex Mono', monospace";
  ctx.fillText(
    data.low_confidence
      ? `Closest match: ${data.predicted_breed} — ${(data.confidence * 100).toFixed(1)}%`
      : `Model confidence: ${(data.confidence * 100).toFixed(1)}%`,
    40,
    y
  );
  y += 55;

  ctx.strokeStyle = line;
  ctx.beginPath();
  ctx.moveTo(40, y);
  ctx.lineTo(W - 40, y);
  ctx.stroke();
  y += 40;

  ctx.fillStyle = pastureDeep;
  ctx.font = "600 20px 'Fraunces', Georgia, serif";
  ctx.fillText("Top predictions", 40, y);
  y += 40;

  data.top_predictions.forEach((p, i) => {
    const pct = p.confidence * 100;
    ctx.fillStyle = ink;
    ctx.font = "500 16px 'IBM Plex Sans', sans-serif";
    ctx.fillText(`${i + 1}. ${p.breed}`, 40, y);
    ctx.textAlign = "right";
    ctx.fillStyle = inkSoft;
    ctx.font = "500 14px 'IBM Plex Mono', monospace";
    ctx.fillText(`${pct.toFixed(1)}%`, W - 40, y);
    ctx.textAlign = "left";
    y += 14;
    const barW = W - 80;
    ctx.fillStyle = line;
    ctx.fillRect(40, y, barW, 8);
    ctx.fillStyle = pasture;
    ctx.fillRect(40, y, barW * (pct / 100), 8);
    y += 38;
  });

  y += 10;
  ctx.strokeStyle = line;
  ctx.beginPath();
  ctx.moveTo(40, y);
  ctx.lineTo(W - 40, y);
  ctx.stroke();
  y += 32;

  ctx.fillStyle = inkSoft;
  ctx.font = "400 13px 'IBM Plex Sans', sans-serif";
  const note = "Scores reflect the model's confidence, not a guaranteed real-world identification. Vision Transformer (ViT-Base), fine-tuned on 3,400 cattle images across 17 breeds.";
  y = wrapCanvasText(ctx, note, 40, y, W - 80, 20);

  ctx.font = "400 12px 'IBM Plex Mono', monospace";
  ctx.fillStyle = inkSoft;
  ctx.fillText(`Generated ${new Date().toLocaleString()}`, 40, H - 30);

  return canvas;
}

const downloadResultBtn = document.getElementById("downloadResultBtn");
downloadResultBtn.addEventListener("click", async () => {
  if (!lastResultData) return;
  const originalLabel = downloadResultBtn.textContent;
  downloadResultBtn.textContent = "Preparing…";
  downloadResultBtn.disabled = true;
  try {
    const canvas = await drawResultCard(lastResultData);
    canvas.toBlob((blob) => {
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      const safeName = lastResultData.predicted_breed.replace(/\s+/g, "-").toLowerCase();
      a.href = url;
      a.download = `cattlevision-ai-result-${safeName}.png`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    }, "image/png");
  } catch (err) {
    alert("Couldn't generate the result card. Please try again.");
  } finally {
    downloadResultBtn.textContent = originalLabel;
    downloadResultBtn.disabled = false;
  }
});
