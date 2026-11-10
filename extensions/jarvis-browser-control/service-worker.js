"use strict";

// Jarvis Browser Control: a Manifest V3 service worker that connects to the
// local Jarvis bridge (ws://127.0.0.1:<port>) and acts only on tabs inside the
// "JARVIS" tab group. Port and pairing token live in chrome.storage.local and
// are set once from the popup; nothing is baked into this file.

const GROUP_TITLE = "JARVIS";
const GROUP_COLOR = "cyan";
const PROTOCOL_VERSION = 2;
const DEFAULT_PORT = 8765;
const HEARTBEAT_MS = 20000;
const SERVER_SILENCE_MS = 70000;
const LOAD_TIMEOUT_MS = 15000;
const CAPABILITIES = Object.freeze([
  "status", "open", "list_tabs", "switch_tab", "close_tab", "back", "forward",
  "reload", "screenshot", "scroll", "read", "inspect", "describe", "click",
  "type", "fill_form", "upload", "wait",
]);

let socket = null;
let connectPromise = null;
let reconnectTimer = null;
let keepAliveTimer = null;
let reconnectAttempt = 0;
let handshakeReady = false;
let serverSelected = false;
let paired = false;
let authFailed = false;
let clientId = null;
let lastError = null;
let lastServerMessage = 0;
let activeTabId = null;
let managedGroupId = null;
let commandQueue = Promise.resolve();

class CommandError extends Error {
  constructor(type, message) {
    super(message);
    this.type = type;
  }
}

function isLive(candidate) {
  return Boolean(candidate) && [WebSocket.OPEN, WebSocket.CONNECTING].includes(candidate.readyState);
}

function scheduleReconnect() {
  if (reconnectTimer || authFailed) return;
  const delay = Math.min(30000, 1000 * (2 ** Math.min(reconnectAttempt, 5)));
  reconnectAttempt += 1;
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    connect();
  }, delay);
}

async function getClientId() {
  if (clientId) return clientId;
  const stored = await chrome.storage.local.get("jarvisClientId");
  clientId = String(stored?.jarvisClientId || "");
  if (!clientId) {
    clientId = globalThis.crypto?.randomUUID?.()
      || `client-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;
    await chrome.storage.local.set({ jarvisClientId: clientId });
  }
  return clientId;
}

async function loadConfig() {
  const stored = await chrome.storage.local.get(["jarvisPort", "jarvisToken"]);
  return {
    port: Number(stored?.jarvisPort) || DEFAULT_PORT,
    token: String(stored?.jarvisToken || "").trim(),
  };
}

function startHeartbeat(connection, instanceId) {
  clearInterval(keepAliveTimer);
  lastServerMessage = Date.now();
  keepAliveTimer = setInterval(() => {
    if (socket !== connection || connection.readyState !== WebSocket.OPEN) return;
    if (Date.now() - lastServerMessage > SERVER_SILENCE_MS) {
      connection.close();
      return;
    }
    connection.send(JSON.stringify({ type: "heartbeat", clientId: instanceId }));
  }, HEARTBEAT_MS);
}

async function runQueued(connection, message) {
  try {
    // WebSocket listeners are concurrent; tab selection and input must not be.
    const pending = commandQueue.then(() => handleCommand(message.action, message.parameters || {}));
    commandQueue = pending.catch(() => {});
    const result = await pending;
    if (connection.readyState === WebSocket.OPEN) {
      connection.send(JSON.stringify({ type: "result", id: message.id, ok: true, result }));
    }
  } catch (error) {
    if (connection.readyState === WebSocket.OPEN) {
      connection.send(JSON.stringify({
        type: "result",
        id: message.id,
        ok: false,
        error: String(error?.message || error),
        errorType: error?.type || "failed",
      }));
    }
  }
}

function connect() {
  if (connectPromise) return connectPromise;
  if (isLive(socket)) return Promise.resolve();
  connectPromise = (async () => {
    const config = await loadConfig();
    paired = Boolean(config.token);
    if (!paired) return;
    const instanceId = await getClientId();
    if (isLive(socket)) return;
    let connection;
    try {
      connection = new WebSocket(`ws://127.0.0.1:${config.port}`);
    } catch (error) {
      lastError = String(error?.message || error);
      scheduleReconnect();
      return;
    }
    socket = connection;
    handshakeReady = false;
    serverSelected = false;
    connection.addEventListener("open", () => {
      if (socket !== connection) return;
      connection.send(JSON.stringify({
        type: "hello",
        token: config.token,
        protocolVersion: PROTOCOL_VERSION,
        clientId: instanceId,
        extensionVersion: chrome.runtime.getManifest().version,
        capabilities: CAPABILITIES,
      }));
    });
    connection.addEventListener("message", async (event) => {
      if (socket !== connection) return;
      lastServerMessage = Date.now();
      let message;
      try {
        message = JSON.parse(event.data);
      } catch (_error) {
        return;
      }
      if (message.type === "ready") {
        handshakeReady = true;
        serverSelected = Boolean(message.selected);
        lastError = null;
        reconnectAttempt = 0;
        startHeartbeat(connection, instanceId);
      } else if (message.type === "upgrade_required") {
        lastError = String(message.error || "This extension build is incompatible with Jarvis.");
        connection.close();
      } else if (message.type === "selected") {
        serverSelected = Boolean(message.selected);
      } else if (message.type === "command" && handshakeReady) {
        await runQueued(connection, message);
      }
    });
    connection.addEventListener("close", (event) => {
      if (socket !== connection) return;
      socket = null;
      handshakeReady = false;
      serverSelected = false;
      clearInterval(keepAliveTimer);
      keepAliveTimer = null;
      if (event?.code === 1008) {
        authFailed = true;
        lastError = "Jarvis rejected the pairing token. Paste the current one from `jarvis --browser-token`.";
        return;
      }
      scheduleReconnect();
    });
    connection.addEventListener("error", () => connection.close());
  })().finally(() => {
    connectPromise = null;
  });
  return connectPromise;
}

function reconnectNow() {
  const old = socket;
  socket = null;
  handshakeReady = false;
  clearTimeout(reconnectTimer);
  reconnectTimer = null;
  authFailed = false;
  reconnectAttempt = 0;
  try { old?.close(); } catch (_error) { /* already closed */ }
  return connect();
}

chrome.runtime.onInstalled.addListener(connect);
chrome.runtime.onStartup.addListener(connect);
// Alarms wake a suspended service worker; the top-level connect() below then
// restores the socket. The alarm is also the backstop when timers were lost.
chrome.alarms.create("jarvis-reconnect", { periodInMinutes: 1 });
chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === "jarvis-reconnect" && !authFailed && !isLive(socket)) connect();
});
chrome.storage.onChanged.addListener((changes, area) => {
  if (area === "local" && (changes.jarvisToken || changes.jarvisPort)) reconnectNow();
});
chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
  if (message?.type === "status") {
    sendResponse({
      paired,
      connected: socket?.readyState === WebSocket.OPEN && handshakeReady,
      socketOpen: socket?.readyState === WebSocket.OPEN,
      selected: serverSelected,
      clientId,
      extensionVersion: chrome.runtime.getManifest().version,
      protocolVersion: PROTOCOL_VERSION,
      reconnectAttempt,
      error: lastError,
    });
  } else if (message?.type === "reconnect") {
    reconnectNow();
    sendResponse({ ok: true });
  }
  return true;
});
connect();

// ---- tab group ------------------------------------------------------------

async function findManagedGroup() {
  if (managedGroupId !== null) {
    try {
      const group = await chrome.tabGroups.get(managedGroupId);
      if (group?.title === GROUP_TITLE) return group;
    } catch (_error) {
      managedGroupId = null;
    }
  }
  const groups = await chrome.tabGroups.query({ title: GROUP_TITLE });
  const group = groups.at(-1) || null;
  managedGroupId = group?.id ?? null;
  return group;
}

async function managedTabs() {
  const group = await findManagedGroup();
  return group ? chrome.tabs.query({ groupId: group.id }) : [];
}

async function currentManagedTab(requestedId) {
  const tabs = await managedTabs();
  if (requestedId !== undefined && requestedId !== null) {
    const requested = tabs.find((candidate) => candidate.id === Number(requestedId));
    if (!requested) throw new CommandError("not_in_group", "The requested tab is not in the JARVIS tab group.");
    return requested;
  }
  let tab = tabs.find((candidate) => candidate.id === activeTabId);
  tab ||= tabs.find((candidate) => candidate.active);
  tab ||= tabs.at(-1);
  if (!tab) throw new CommandError("no_tab", "No JARVIS tab exists yet. Open a page first.");
  activeTabId = tab.id;
  return tab;
}

async function focusTab(tab) {
  await chrome.tabs.update(tab.id, { active: true });
  await chrome.windows.update(tab.windowId, { focused: true });
  activeTabId = tab.id;
}

async function addTabToGroup(tab) {
  const group = await findManagedGroup();
  if (group && group.windowId !== tab.windowId) {
    tab = await chrome.tabs.move(tab.id, { windowId: group.windowId, index: -1 });
  }
  if (group) {
    await chrome.tabs.group({ groupId: group.id, tabIds: tab.id });
  } else {
    managedGroupId = await chrome.tabs.group({ tabIds: tab.id });
    await chrome.tabGroups.update(managedGroupId, {
      title: GROUP_TITLE,
      color: GROUP_COLOR,
      collapsed: false,
    });
  }
  await focusTab(tab);
  return tab;
}

// Resolves when the tab reports "complete" or the timeout passes. Register
// before triggering navigation so the event cannot be missed.
function loadWaiter(tabId, timeoutMs = LOAD_TIMEOUT_MS) {
  let done;
  const finished = new Promise((resolve) => { done = resolve; });
  const listener = (id, info) => {
    if (id === tabId && info.status === "complete") done(true);
  };
  chrome.tabs.onUpdated?.addListener(listener);
  const timer = setTimeout(() => done(false), timeoutMs);
  return finished.finally(() => {
    clearTimeout(timer);
    chrome.tabs.onUpdated?.removeListener(listener);
  });
}

async function openManagedTab(url, forceNew, requestedId) {
  if (!/^https?:\/\//i.test(String(url || ""))) {
    throw new CommandError("invalid", "Only http and https URLs can be opened.");
  }
  const tabs = await managedTabs();
  if (!forceNew && tabs.length) {
    const tab = await currentManagedTab(requestedId);
    const waiter = loadWaiter(tab.id);
    await chrome.tabs.update(tab.id, { url, active: true });
    await focusTab(tab);
    return { tab, loaded: await waiter };
  }
  const group = await findManagedGroup();
  const created = await chrome.tabs.create({
    url,
    active: true,
    ...(group ? { windowId: group.windowId } : {}),
  });
  const waiter = loadWaiter(created.id);
  const tab = await addTabToGroup(created);
  return { tab, loaded: await waiter };
}

// ---- page and debugger helpers ---------------------------------------------

async function runPageAction(tabId, action, parameters) {
  const results = await chrome.scripting.executeScript({
    target: { tabId },
    func: pageAction,
    args: [action, parameters],
  });
  const result = results[0]?.result;
  if (result?.__error) throw new CommandError(result.__error.type, result.__error.message);
  return result;
}

async function adoptOpenedTabs(beforeIds, openerId) {
  const allTabs = await chrome.tabs.query({});
  const opened = allTabs.filter((tab) => !beforeIds.has(tab.id) && tab.openerTabId === openerId);
  for (const tab of opened) await addTabToGroup(tab);
  return opened.at(-1) || null;
}

// Runs fn(send) with the debugger attached and always detaches, which is what
// removes Chrome's "debugging this browser" banner again.
async function withDebugger(tabId, fn) {
  const debuggee = { tabId };
  await chrome.debugger.attach(debuggee, "1.3");
  try {
    return await fn((method, params) => chrome.debugger.sendCommand(debuggee, method, params));
  } finally {
    await chrome.debugger.detach(debuggee);
  }
}

async function realClick(tabId, point) {
  let started = false;
  try {
    await withDebugger(tabId, async (send) => {
      for (const type of ["mouseMoved", "mousePressed", "mouseReleased"]) {
        if (type === "mousePressed") started = true;
        await send("Input.dispatchMouseEvent", {
          type, x: point.x, y: point.y,
          ...(type === "mouseMoved" ? {} : { button: "left", clickCount: 1 }),
        });
      }
    });
  } catch (error) {
    if (!started) throw error;
    throw new CommandError(
      "uncertain_action",
      `The click may or may not have happened (${error?.message || error}). Inspect the page before retrying.`,
    );
  }
}

async function realType(tabId, text, submit) {
  let started = false;
  try {
    await withDebugger(tabId, async (send) => {
      started = true;
      if (text) await send("Input.insertText", { text });
      if (submit) {
        const key = { key: "Enter", code: "Enter", windowsVirtualKeyCode: 13, nativeVirtualKeyCode: 13 };
        await send("Input.dispatchKeyEvent", { type: "keyDown", text: "\r", ...key });
        await send("Input.dispatchKeyEvent", { type: "keyUp", ...key });
      }
    });
  } catch (error) {
    if (!started) throw error;
    throw new CommandError(
      "uncertain_action",
      `Typing may be partly done (${error?.message || error}). Inspect the page before retrying.`,
    );
  }
}

async function uploadFile(tab, parameters) {
  const ref = Number(parameters.ref);
  if (!Number.isInteger(ref) || !parameters.path) {
    throw new CommandError("invalid", "A file input ref and a path are required.");
  }
  await runPageAction(tab.id, "check_file_input", { ref });
  await withDebugger(tab.id, async (send) => {
    const evaluated = await send("Runtime.evaluate", {
      expression: `document.querySelector('[data-jarvis-ref="${ref}"]')`,
      returnByValue: false,
    });
    const objectId = evaluated?.result?.objectId;
    if (!objectId) throw new CommandError("stale_ref", "The file input was not found; inspect the page again.");
    await send("DOM.setFileInputFiles", { files: [String(parameters.path)], objectId });
  });
  return { message: "Selected the file in the page's file input. Nothing was submitted." };
}

function tabSummary(tab, index) {
  return { id: tab.id, index: index + 1, title: tab.title || "", url: tab.url || "", active: tab.id === activeTabId };
}

// ---- command dispatch -------------------------------------------------------

const PAGE_ACTIONS = new Set(["read", "inspect", "describe", "scroll", "wait"]);

async function handleCommand(action, parameters = {}) {
  if (action === "status" || action === "list_tabs") {
    const tabs = await managedTabs();
    return { connected: true, groupTitle: GROUP_TITLE, tabs: tabs.map(tabSummary) };
  }
  if (action === "open") {
    const { tab, loaded } = await openManagedTab(parameters.url, Boolean(parameters.new_tab), parameters.tab_id);
    const updated = await chrome.tabs.get(tab.id);
    return {
      message: `Opened ${parameters.url} in the JARVIS tab group.`,
      tabId: tab.id, url: updated.url, title: updated.title, loaded,
    };
  }
  const tab = await currentManagedTab(parameters.tab_id);
  if (action === "switch_tab") {
    await focusTab(tab);
    return { message: `Switched to ${tab.title || tab.url}.`, tabId: tab.id, url: tab.url, title: tab.title };
  }
  if (action === "close_tab") {
    await chrome.tabs.remove(tab.id);
    if (activeTabId === tab.id) activeTabId = null;
    return { message: "Closed the tab.", tabId: tab.id };
  }
  if (action === "back" || action === "forward" || action === "reload") {
    const waiter = loadWaiter(tab.id);
    if (action === "back") await chrome.tabs.goBack(tab.id);
    else if (action === "forward") await chrome.tabs.goForward(tab.id);
    else await chrome.tabs.reload(tab.id);
    const loaded = await waiter;
    const updated = await chrome.tabs.get(tab.id);
    return { message: `Navigated ${action}.`, tabId: tab.id, url: updated.url, title: updated.title, loaded };
  }
  if (action === "screenshot") {
    await focusTab(tab);
    const dataUrl = await chrome.tabs.captureVisibleTab(tab.windowId, { format: "png" });
    return { message: "Captured the tab.", dataUrl, tabId: tab.id, url: tab.url };
  }
  if (action === "upload") return uploadFile(tab, parameters);

  const beforeIds = new Set((await chrome.tabs.query({})).map((item) => item.id));
  let result;
  if (action === "click") {
    await focusTab(tab);
    const point = await runPageAction(tab.id, "click_point", parameters);
    await realClick(tab.id, point);
    result = { message: `Clicked '${point.label}'.` };
    await new Promise((resolve) => setTimeout(resolve, 300));
    const opened = await adoptOpenedTabs(beforeIds, tab.id);
    if (opened) result.message += ` A new tab opened at ${opened.url} and joined the JARVIS group.`;
    const current = await chrome.tabs.get(tab.id);
    if (current.status === "loading") await loadWaiter(tab.id, 8000);
  } else if (action === "type") {
    await focusTab(tab);
    const prepared = await runPageAction(tab.id, "prepare_type", parameters);
    const text = String(parameters.text ?? "");
    await realType(tab.id, text, Boolean(parameters.submit));
    result = {
      message: `Typed ${text.length} characters into '${prepared.label}'${parameters.submit ? " and pressed Enter" : ""}.`,
    };
  } else if (action === "fill_form" || PAGE_ACTIONS.has(action)) {
    result = await runPageAction(tab.id, action, parameters);
  } else {
    throw new CommandError("unsupported", `Unsupported browser action: ${action}`);
  }
  const updated = await chrome.tabs.get(tab.id);
  return { tabId: tab.id, url: updated.url, title: updated.title, ...result };
}

// ---- injected page function --------------------------------------------------
// Serialised by chrome.scripting, so it must stay self-contained. It never
// throws: failures come back as {__error: {type, message}}.

async function pageAction(action, parameters) {
  const fail = (type, message) => {
    const error = new Error(message);
    error.type = type;
    throw error;
  };
  try {
    const state = (globalThis.__jarvis ||= { url: location.href, refs: new Map(), byElement: new WeakMap(), next: 1 });
    if (state.url !== location.href) {
      Object.assign(state, { url: location.href, refs: new Map(), byElement: new WeakMap(), next: 1 });
    }
    const SELECTOR = [
      "a[href]", "button", "input:not([type=hidden])", "textarea", "select", "summary",
      "[role='button']", "[role='link']", "[role='tab']", "[role='checkbox']", "[role='radio']",
      "[role='switch']", "[role='menuitem']", "[role='option']", "[role='combobox']",
      "[contenteditable='']", "[contenteditable='true']",
      ".monaco-editor", ".ace_editor", ".CodeMirror", ".cm-editor",
    ].join(",");
    const clean = (value, limit = 160) => String(value ?? "").replace(/\s+/g, " ").trim().slice(0, limit);
    const isFile = (element) => element.matches?.("input[type=file]");
    const visible = (element) => {
      const style = getComputedStyle(element);
      const rect = element.getBoundingClientRect();
      return style.visibility !== "hidden" && style.display !== "none" && rect.width > 0 && rect.height > 0;
    };
    const usable = (element) => visible(element) || isFile(element);
    const label = (element) => clean(
      element.getAttribute("aria-label")
      || element.labels?.[0]?.innerText
      || element.getAttribute("placeholder")
      || element.innerText
      || (/^(submit|button|reset)$/i.test(element.type || "") ? element.value : "")
      || element.getAttribute("title")
      || element.querySelector?.("img[alt]")?.getAttribute("alt")
      || element.getAttribute("name")
      || element.tagName,
    );
    const controls = () => Array.from(document.querySelectorAll(SELECTOR)).filter(usable);
    const refOf = (element) => {
      let ref = state.byElement.get(element);
      if (!ref) {
        ref = state.next++;
        state.byElement.set(element, ref);
        element.setAttribute("data-jarvis-ref", String(ref));
      }
      state.refs.set(ref, { element, label: label(element) });
      return ref;
    };
    const resolve = (target) => {
      const text = String(target ?? "").trim();
      if (!text) fail("invalid", "A ref number or label is required.");
      if (/^\d+$/.test(text)) {
        const saved = state.refs.get(Number(text));
        if (!saved || !saved.element.isConnected || !usable(saved.element) || label(saved.element) !== saved.label) {
          fail("stale_ref", "That element reference is no longer valid. Inspect the page again.");
        }
        return saved.element;
      }
      const lowered = text.toLowerCase();
      let matches = controls().filter((element) => label(element).toLowerCase() === lowered);
      if (!matches.length) matches = controls().filter((element) => label(element).toLowerCase().includes(lowered));
      if (!matches.length) fail("not_found", `No control matches '${text}'. Inspect the page.`);
      if (matches.length > 1) fail("ambiguous", `'${text}' matches ${matches.length} controls. Use a ref from inspect.`);
      return matches[0];
    };
    const isSensitive = (element) => {
      const type = (element.type || "").toLowerCase();
      const auto = (element.getAttribute?.("autocomplete") || "").toLowerCase();
      return type === "password" || /\b(cc-|one-time-code|current-password|new-password)/.test(auto);
    };
    const formInfo = (element) => {
      const form = element.form || element.closest?.("form");
      if (!form) return null;
      let action = form.getAttribute("action") || "";
      try { action = new URL(action || location.href, location.href).href; } catch (_error) { /* keep raw */ }
      const submitters = Array.from(form.querySelectorAll("button, input[type=submit], input[type=image]"))
        .filter((item) => !item.type || item.type === "submit" || item.type === "image").slice(0, 5);
      return {
        action,
        method: (form.getAttribute("method") || "get").toLowerCase(),
        submitLabels: submitters.map((item) => label(item)),
      };
    };
    const describe = (element) => {
      const type = (element.type || "").toLowerCase();
      const sensitive = isSensitive(element);
      return {
        ref: refOf(element),
        tag: element.tagName.toLowerCase(),
        role: element.getAttribute("role") || "",
        type,
        label: label(element),
        text: clean(element.innerText),
        ariaLabel: clean(element.getAttribute("aria-label")),
        title: clean(element.getAttribute("title")),
        name: clean(element.getAttribute("name")),
        id: clean(element.id),
        placeholder: clean(element.getAttribute("placeholder")),
        autocomplete: clean(element.getAttribute("autocomplete")),
        inputmode: clean(element.getAttribute("inputmode")),
        href: element.href ? String(element.href) : "",
        disabled: Boolean(element.disabled),
        checked: typeof element.checked === "boolean" ? element.checked : null,
        value: sensitive || type === "file" ? "" : clean(element.value, 80),
        form: formInfo(element),
      };
    };
    const setValue = (element, value) => {
      if (isSensitive(element)) fail("refused", "Credential fields must be filled in by the user.");
      const type = (element.type || "").toLowerCase();
      if (element.tagName === "SELECT") {
        const wanted = String(value).toLowerCase();
        const option = Array.from(element.options).find((item) =>
          item.label.toLowerCase() === wanted || item.value.toLowerCase() === wanted)
          || Array.from(element.options).find((item) => item.label.toLowerCase().includes(wanted));
        if (!option) fail("not_found", `The dropdown has no option '${value}'.`);
        element.value = option.value;
        element.dispatchEvent(new Event("input", { bubbles: true }));
        element.dispatchEvent(new Event("change", { bubbles: true }));
        return;
      }
      if (type === "checkbox" || type === "radio") {
        const want = /^(true|yes|on|1|checked)$/i.test(String(value).trim());
        if (element.checked !== want) element.click();
        return;
      }
      if (element.isContentEditable) {
        element.textContent = String(value);
        element.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertText", data: String(value) }));
        return;
      }
      if (!/^(INPUT|TEXTAREA)$/.test(element.tagName)) fail("invalid", "That control is not a text field.");
      const prototype = element instanceof HTMLTextAreaElement ? HTMLTextAreaElement.prototype : HTMLInputElement.prototype;
      const setter = Object.getOwnPropertyDescriptor(prototype, "value")?.set;
      if (setter) setter.call(element, String(value)); else element.value = String(value);
      element.dispatchEvent(new Event("input", { bubbles: true }));
      element.dispatchEvent(new Event("change", { bubbles: true }));
    };

    if (action === "read") {
      const element = parameters.ref !== undefined && parameters.ref !== null ? resolve(parameters.ref) : document.body;
      const full = String(element.innerText || "").replace(/[ \t]+\n/g, "\n").replace(/\n{3,}/g, "\n\n").trim();
      const limit = Math.max(200, Math.min(50000, Number(parameters.max_chars) || 8000));
      return { text: full.slice(0, limit), total: full.length, truncated: full.length > limit };
    }
    if (action === "inspect") {
      const filter = clean(parameters.filter).toLowerCase();
      const matching = controls().filter((element) => {
        if (!filter) return true;
        return `${label(element)} ${element.getAttribute("role") || ""} ${element.tagName} ${element.type || ""} ${element.href || ""}`
          .toLowerCase().includes(filter);
      });
      const offset = Math.max(0, Number(parameters.offset) || 0);
      const limit = 80;
      const items = matching.slice(offset, offset + limit).map((element) => {
        const type = (element.type || "").toLowerCase();
        const sensitive = isSensitive(element);
        const item = {
          ref: refOf(element),
          tag: element.tagName.toLowerCase(),
          role: element.getAttribute("role") || "",
          type,
          label: label(element),
          value: sensitive || type === "file" || /^(button|submit|reset|image)$/.test(type) ? "" : clean(element.value, 60),
          disabled: Boolean(element.disabled),
          checked: typeof element.checked === "boolean" ? element.checked : null,
          sensitive,
        };
        if (element.href) item.href = clean(element.href, 120);
        if (element.tagName === "SELECT") item.options = Array.from(element.options).slice(0, 15).map((o) => clean(o.label, 40));
        return item;
      });
      return { items, total: matching.length, next_offset: offset + limit < matching.length ? offset + limit : null };
    }
    if (action === "describe") {
      return { elements: (parameters.targets || []).map((target) => describe(resolve(target))) };
    }
    if (action === "scroll") {
      const direction = parameters.direction || "down";
      const pixels = Math.max(50, Math.min(5000, Number(parameters.pixels) || 700));
      if (direction === "top") scrollTo({ top: 0, behavior: "instant" });
      else if (direction === "bottom") scrollTo({ top: document.documentElement.scrollHeight, behavior: "instant" });
      else scrollBy({ top: direction === "up" ? -pixels : pixels, behavior: "instant" });
      const maxScroll = Math.max(0, document.documentElement.scrollHeight - innerHeight);
      return { scrollY: Math.round(scrollY), maxScroll: Math.round(maxScroll), message: "Scrolled." };
    }
    if (action === "click_point") {
      const element = resolve(parameters.ref);
      if (element.disabled) fail("disabled", "That control is disabled.");
      if (isFile(element)) fail("invalid", "That is a file input. Use browser_upload instead of clicking it.");
      element.scrollIntoView({ block: "center", inline: "center" });
      const rect = element.getBoundingClientRect();
      const x = Math.max(0, Math.min(innerWidth - 1, rect.x + rect.width / 2));
      const y = Math.max(0, Math.min(innerHeight - 1, rect.y + rect.height / 2));
      const hit = document.elementFromPoint(x, y);
      if (hit !== element && !element.contains(hit)) {
        fail("covered", "The control is covered by another element (a dialog or banner?). Inspect the page.");
      }
      return { x, y, label: label(element) };
    }
    if (action === "prepare_type") {
      const element = resolve(parameters.ref);
      if (isSensitive(element)) fail("refused", "Credential fields must be filled in by the user.");
      if (element.disabled || element.readOnly) fail("disabled", "That field is disabled or read-only.");
      element.scrollIntoView({ block: "center" });
      element.focus();
      if (document.activeElement !== element && !element.contains(document.activeElement)) {
        fail("not_focusable", "The field could not be focused.");
      }
      const clear = parameters.clear !== false;
      if (/^(INPUT|TEXTAREA)$/.test(element.tagName)) {
        try {
          if (clear) element.select();
          else element.setSelectionRange(element.value.length, element.value.length);
        } catch (_error) { /* types like email/number have no selection API */ }
      } else {
        const range = document.createRange();
        range.selectNodeContents(element);
        if (!clear) range.collapse(false);
        const selection = getSelection();
        selection.removeAllRanges();
        selection.addRange(range);
      }
      return { label: label(element) };
    }
    if (action === "fill_form") {
      const fields = Array.isArray(parameters.fields) ? parameters.fields : [];
      const elements = fields.map((field) => resolve(field.ref));
      elements.forEach((element) => { if (isSensitive(element)) fail("refused", "Credential fields must be filled in by the user."); });
      elements.forEach((element, index) => setValue(element, fields[index].value));
      return { message: `Filled ${elements.length} field(s). No final form action was taken.` };
    }
    if (action === "check_file_input") {
      const element = resolve(parameters.ref);
      if (!isFile(element)) fail("invalid", "That element is not a file input.");
      return { message: "ok" };
    }
    if (action === "wait") {
      const deadline = Date.now() + Math.min(Math.max(Number(parameters.timeout_ms) || 5000, 100), 30000);
      const wanted = String(parameters.text || "").toLowerCase();
      return await new Promise((resolvePromise, reject) => {
        const check = () => {
          const ready = wanted
            ? document.body.innerText.toLowerCase().includes(wanted)
            : document.readyState !== "loading";
          if (ready) resolvePromise({ message: wanted ? "The text appeared." : "The page is ready." });
          else if (Date.now() >= deadline) {
            const error = new Error(wanted ? `The text '${parameters.text}' did not appear in time.` : "The page did not finish loading.");
            error.type = "timeout";
            reject(error);
          } else setTimeout(check, 100);
        };
        check();
      });
    }
    return fail("unsupported", `Unsupported page action: ${action}`);
  } catch (error) {
    return { __error: { type: error?.type || "failed", message: String(error?.message || error) } };
  }
}
