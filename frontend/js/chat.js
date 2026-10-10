// chat.js — floating CattleVision Assistant widget, backed by /api/chat (Ollama).

const chatWidget = document.getElementById("chatWidget");
const chatPanel = document.getElementById("chatPanel");
const chatToggleBtn = document.getElementById("chatToggleBtn");
const chatCloseBtn = document.getElementById("chatCloseBtn");
const chatMessages = document.getElementById("chatMessages");
const chatInput = document.getElementById("chatInput");
const chatSendBtn = document.getElementById("chatSendBtn");

let chatHistory = []; // [{role, content}], sent to the backend each turn
let chatBusy = false;

chatToggleBtn.addEventListener("click", () => {
  chatPanel.classList.toggle("open");
  if (chatPanel.classList.contains("open")) chatInput.focus();
});
chatCloseBtn.addEventListener("click", () => chatPanel.classList.remove("open"));

function appendMessage(text, cls) {
  const div = document.createElement("div");
  div.className = `chat-msg ${cls}`;
  div.textContent = text;
  chatMessages.appendChild(div);
  chatMessages.scrollTop = chatMessages.scrollHeight;
  return div;
}

async function handleSend() {
  const text = chatInput.value.trim();
  if (!text || chatBusy) return;

  chatBusy = true;
  chatSendBtn.disabled = true;
  chatInput.value = "";
  appendMessage(text, "chat-msg-user");

  const typingEl = appendMessage("Thinking…", "chat-msg-bot chat-msg-typing");

  try {
    const data = await sendChatMessage(text, chatHistory);
    typingEl.remove();
    appendMessage(data.reply, "chat-msg-bot");
    chatHistory.push({ role: "user", content: text });
    chatHistory.push({ role: "assistant", content: data.reply });
    // Keep the payload small — only send recent turns.
    if (chatHistory.length > 20) chatHistory = chatHistory.slice(-20);
  } catch (err) {
    typingEl.remove();
    appendMessage(err.message, "chat-msg-error");
  } finally {
    chatBusy = false;
    chatSendBtn.disabled = false;
    chatInput.focus();
  }
}

chatSendBtn.addEventListener("click", handleSend);
chatInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") handleSend();
});
