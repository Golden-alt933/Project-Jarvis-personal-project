const statusLine = document.querySelector("#status");
const details = document.querySelector("#details");
const portInput = document.querySelector("#port");
const tokenInput = document.querySelector("#token");

function describe(reply) {
  if (!reply) return "The extension is starting. Try again in a moment.";
  if (reply.error) return reply.error;
  if (!reply.paired) return "Not paired yet. Paste the pairing token below.";
  if (reply.connected && reply.selected) return "Connected to Jarvis.";
  if (reply.connected) return "Connected as a standby Chrome profile.";
  if (reply.socketOpen) return "Connected locally; waiting for Jarvis to answer.";
  return "Not connected. Start Jarvis (uv run jarvis); this connects by itself within a few seconds.";
}

async function refresh() {
  const reply = await chrome.runtime.sendMessage({ type: "status" });
  statusLine.textContent = describe(reply);
  details.textContent = [
    `Profile id: ${reply?.clientId || "initializing"}`,
    `Extension: ${reply?.extensionVersion || "unknown"}  Protocol: ${reply?.protocolVersion ?? "unknown"}`,
    reply?.reconnectAttempt ? `Reconnect attempts: ${reply.reconnectAttempt}` : "",
  ].filter(Boolean).join("\n");
}

async function loadSaved() {
  const stored = await chrome.storage.local.get(["jarvisPort", "jarvisToken"]);
  portInput.value = stored.jarvisPort || 8765;
  if (stored.jarvisToken) tokenInput.placeholder = "Token saved (paste a new one to replace it)";
}

document.querySelector("#save").addEventListener("click", async () => {
  const port = Number(portInput.value);
  if (!Number.isInteger(port) || port < 1 || port > 65535) {
    statusLine.textContent = "Enter a port between 1 and 65535.";
    return;
  }
  const update = { jarvisPort: port };
  const token = tokenInput.value.trim();
  if (token) update.jarvisToken = token;
  await chrome.storage.local.set(update);
  tokenInput.value = "";
  statusLine.textContent = "Connecting...";
  await chrome.runtime.sendMessage({ type: "reconnect" });
  setTimeout(refresh, 800);
  setTimeout(loadSaved, 100);
});

document.querySelector("#reconnect").addEventListener("click", async () => {
  statusLine.textContent = "Connecting...";
  await chrome.runtime.sendMessage({ type: "reconnect" });
  setTimeout(refresh, 800);
});

loadSaved();
refresh();
