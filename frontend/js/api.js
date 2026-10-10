// api.js — all network calls to the CattleVision AI backend live here.
// The backend origin is same-origin by default (FastAPI serves this frontend
// too). If you run the frontend separately, change API_BASE below.

const API_BASE = "";

// ---------- Auth session (stored in this site's own localStorage — not a
// Claude artifact sandbox, so normal browser storage is appropriate here) ----------
const AUTH_TOKEN_KEY = "cattlevision_token";
const AUTH_USERNAME_KEY = "cattlevision_username";

function getAuthToken() {
  return localStorage.getItem(AUTH_TOKEN_KEY);
}
function getAuthUsername() {
  return localStorage.getItem(AUTH_USERNAME_KEY);
}
function setAuthSession(token, username) {
  localStorage.setItem(AUTH_TOKEN_KEY, token);
  localStorage.setItem(AUTH_USERNAME_KEY, username);
}
function clearAuthSession() {
  localStorage.removeItem(AUTH_TOKEN_KEY);
  localStorage.removeItem(AUTH_USERNAME_KEY);
}

async function _readErrorDetail(res, fallback) {
  try {
    const body = await res.json();
    return body.detail || fallback;
  } catch (_) {
    return fallback;
  }
}

async function registerUser(username, password) {
  const res = await fetch(`${API_BASE}/api/auth/register`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!res.ok) throw new Error(await _readErrorDetail(res, "Could not create account."));
  return res.json();
}

async function loginUser(username, password) {
  const res = await fetch(`${API_BASE}/api/auth/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!res.ok) throw new Error(await _readErrorDetail(res, "Login failed."));
  return res.json();
}

async function fetchHistory() {
  const token = getAuthToken();
  const res = await fetch(`${API_BASE}/api/history`, {
    headers: token ? { Authorization: `Bearer ${token}` } : {},
  });
  if (!res.ok) throw new Error(await _readErrorDetail(res, "Could not load history."));
  return res.json();
}

async function fetchBreeds() {
  const res = await fetch(`${API_BASE}/api/breeds`);
  if (!res.ok) throw new Error("Could not load breed list from the model.");
  return res.json();
}

async function sendChatMessage(message, history) {
  const res = await fetch(`${API_BASE}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, history }),
  });

  if (!res.ok) {
    let detail = "Unable to reach the chatbot right now. Please try again.";
    try {
      const body = await res.json();
      if (body.detail) detail = body.detail;
    } catch (_) {}
    throw new Error(detail);
  }

  return res.json();
}

async function predictImage(file) {
  const formData = new FormData();
  formData.append("file", file);

  const token = getAuthToken();
  const res = await fetch(`${API_BASE}/api/predict`, {
    method: "POST",
    headers: token ? { Authorization: `Bearer ${token}` } : {},
    body: formData,
  });

  if (!res.ok) throw new Error(await _readErrorDetail(res, "Unable to analyze this image right now. Please try again."));
  return res.json();
}
