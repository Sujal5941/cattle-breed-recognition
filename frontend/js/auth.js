// auth.js — account area, login/register modal, and the History page.
// Session storage helpers (getAuthToken, setAuthSession, etc.) live in api.js.

const accountArea = document.getElementById("accountArea");
const authModalBackdrop = document.getElementById("authModalBackdrop");
const authModalClose = document.getElementById("authModalClose");
const authTabLogin = document.getElementById("authTabLogin");
const authTabRegister = document.getElementById("authTabRegister");
const authForm = document.getElementById("authForm");
const authError = document.getElementById("authError");
const authSubmitBtn = document.getElementById("authSubmitBtn");

let authMode = "login";

function renderAccountArea() {
  const username = getAuthUsername();
  if (username) {
    accountArea.innerHTML = `
      <span class="account-greeting">Hi, ${username}</span>
      <button class="btn btn-ghost" id="logoutBtn" type="button">Log Out</button>`;
    document.getElementById("logoutBtn").addEventListener("click", () => {
      clearAuthSession();
      renderAccountArea();
      showPage("home");
    });
  } else {
    accountArea.innerHTML = `<button class="btn btn-ghost" id="loginNavBtn" type="button">Log In</button>`;
    document.getElementById("loginNavBtn").addEventListener("click", () => openAuthModal("login"));
  }
}

function setAuthMode(mode) {
  authMode = mode;
  authTabLogin.classList.toggle("active", mode === "login");
  authTabRegister.classList.toggle("active", mode === "register");
  authSubmitBtn.textContent = mode === "login" ? "Log In" : "Create Account";
  authError.innerHTML = "";
}

function openAuthModal(mode = "login") {
  setAuthMode(mode);
  authModalBackdrop.classList.add("open");
}

function closeAuthModal() {
  authModalBackdrop.classList.remove("open");
  authError.innerHTML = "";
  authForm.reset();
}

authModalClose.addEventListener("click", closeAuthModal);
authModalBackdrop.addEventListener("click", (e) => {
  if (e.target === authModalBackdrop) closeAuthModal();
});
authTabLogin.addEventListener("click", () => setAuthMode("login"));
authTabRegister.addEventListener("click", () => setAuthMode("register"));

authForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  const username = document.getElementById("authUsername").value.trim();
  const password = document.getElementById("authPassword").value;
  authError.innerHTML = "";
  authSubmitBtn.disabled = true;
  try {
    const data =
      authMode === "login" ? await loginUser(username, password) : await registerUser(username, password);
    setAuthSession(data.token, data.username);
    renderAccountArea();
    closeAuthModal();
    if (document.getElementById("page-history").classList.contains("active")) loadHistory();
  } catch (err) {
    authError.innerHTML = `<div class="error-box" style="margin:10px 0;">${err.message}</div>`;
  } finally {
    authSubmitBtn.disabled = false;
  }
});

renderAccountArea();

// ---------- History page ----------
async function loadHistory() {
  const content = document.getElementById("historyContent");
  if (!getAuthToken()) {
    content.innerHTML = `
      <div class="card">
        <h3>Log in to see your history</h3>
        <p>Create a free account or log in to keep a record of the images you've analyzed.</p>
        <button class="btn btn-primary" id="historyLoginBtn" type="button">Log In / Create Account</button>
      </div>`;
    document.getElementById("historyLoginBtn").addEventListener("click", () => openAuthModal("login"));
    return;
  }

  content.innerHTML = "<p>Loading…</p>";
  try {
    const data = await fetchHistory();
    if (data.predictions.length === 0) {
      content.innerHTML = `<p>No predictions yet — analyze an image on the <a href="#identify" data-nav="identify">Identify Breed</a> page to see it here.</p>`;
      return;
    }
    content.innerHTML = `<div class="grid-4">${data.predictions
      .map(
        (p) => `
        <div class="card history-card">
          <div class="breed-thumb" style="background-image:url('${p.image_url}');"></div>
          <h3 style="margin-top:12px; font-size:1rem;">${p.low_confidence ? "Breed Not Recognized" : p.predicted_breed}</h3>
          <p style="font-size:0.85rem; margin-bottom:4px;">${(p.confidence * 100).toFixed(1)}% confidence</p>
          <p style="font-size:0.78rem; color:var(--ink-soft);">${new Date(p.created_at).toLocaleString()}</p>
        </div>`
      )
      .join("")}</div>`;
  } catch (err) {
    content.innerHTML = `<div class="error-box">${err.message}</div>`;
  }
}
