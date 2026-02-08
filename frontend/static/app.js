const chatContainer = document.getElementById("chatContainer");
const messageInput = document.getElementById("messageInput");
const sendBtn = document.getElementById("sendBtn");
const debugPanel = document.getElementById("debugPanel");

const sessionId = "session_" + Date.now();

messageInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

function addMessage(content, role, meta) {
  const div = document.createElement("div");
  div.className = `message ${role}`;

  const contentDiv = document.createElement("div");
  contentDiv.className = "message-content";
  contentDiv.innerHTML = formatMarkdown(content);
  div.appendChild(contentDiv);

  if (meta) {
    const metaDiv = document.createElement("div");
    metaDiv.className = "message-meta";
    metaDiv.textContent = meta;
    div.appendChild(metaDiv);
  }

  chatContainer.appendChild(div);
  chatContainer.scrollTop = chatContainer.scrollHeight;
  return div;
}

function formatMarkdown(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/^- (.+)$/gm, "<li>$1</li>")
    .replace(/(<li>.*<\/li>)/gs, "<ul>$1</ul>")
    .replace(/\n\n/g, "</p><p>")
    .replace(/\n/g, "<br>")
    .replace(/^(.+)$/, "<p>$1</p>");
}

async function sendMessage() {
  const message = messageInput.value.trim();
  if (!message) return;

  addMessage(message, "user", "You");
  messageInput.value = "";
  sendBtn.disabled = true;

  const loadingDiv = addMessage("Thinking...", "assistant loading", "Coach");

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, session_id: sessionId }),
    });

    const data = await response.json();

    chatContainer.removeChild(loadingDiv);
    addMessage(data.response, "assistant", `Coach (${data.agent_used})`);

    updateDebug(data);
  } catch (error) {
    chatContainer.removeChild(loadingDiv);
    addMessage(
      "Sorry, something went wrong. Please try again.",
      "assistant",
      "System"
    );
  }

  sendBtn.disabled = false;
  messageInput.focus();
}

function updateDebug(data) {
  document.getElementById("debugPredicates").textContent = JSON.stringify(
    data.predicates,
    null,
    2
  );
  document.getElementById("debugASP").textContent = JSON.stringify(
    data.asp_directives,
    null,
    2
  );
  document.getElementById("debugAgent").textContent = data.agent_used;
}

function toggleDebug() {
  debugPanel.classList.toggle("visible");
}
