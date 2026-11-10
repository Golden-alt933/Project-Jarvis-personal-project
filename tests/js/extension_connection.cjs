// Connection contract for the service worker, with a stubbed chrome API and WebSocket.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const WORKER = path.join(__dirname, '..', '..', 'extensions', 'jarvis-browser-control', 'service-worker.js');

function harness(stored = { jarvisToken: 'fixture-token', jarvisPort: 1234 }) {
  const sockets = [];
  const timers = [];
  class Socket {
    static OPEN = 1;
    static CONNECTING = 0;
    static CLOSED = 3;
    readyState = Socket.CONNECTING;
    listeners = {};
    sent = [];
    constructor(url) { this.url = url; sockets.push(this); }
    addEventListener(name, fn) { this.listeners[name] = fn; }
    send(value) { this.sent.push(JSON.parse(value)); }
    close() { this.readyState = Socket.CLOSED; }
  }
  const context = vm.createContext({
    WebSocket: Socket, Date, Math, console,
    setTimeout(fn, delay) { timers.push({ fn, delay }); return timers.length; },
    clearTimeout() {}, setInterval() { return 1; }, clearInterval() {},
    chrome: {
      storage: {
        local: {
          async get(keys) {
            const wanted = Array.isArray(keys) ? keys : [keys];
            return Object.fromEntries(wanted.map((key) => [key, stored[key]]));
          },
          async set(value) { Object.assign(stored, value); },
        },
      },
      runtime: { getManifest() { return { version: '1.0.0' }; } },
    },
  });
  const source = fs.readFileSync(WORKER, 'utf8');
  vm.runInContext(source.slice(0, source.indexOf('chrome.runtime.onInstalled')), context);
  return { context, sockets, timers, stored, Socket };
}

const tick = () => new Promise((resolve) => setImmediate(resolve));

test('concurrent reconnect triggers share one socket and stale closes cannot discard it', async () => {
  const { context, sockets, Socket } = harness();
  await Promise.all([context.connect(), context.connect(), context.connect()]);
  assert.equal(sockets.length, 1);

  sockets[0].readyState = Socket.CLOSED;
  await context.connect();
  assert.equal(sockets.length, 2);
  sockets[0].listeners.close({ code: 1006 });
  await context.connect();
  assert.equal(sockets.length, 2, 'an old close event must not discard the replacement socket');
});

test('the port and token come from storage, never from the code', async () => {
  const { context, sockets, Socket } = harness({ jarvisToken: ' pasted-token ', jarvisPort: 9911 });
  await context.connect();
  assert.equal(sockets[0].url, 'ws://127.0.0.1:9911');
  sockets[0].readyState = Socket.OPEN;
  sockets[0].listeners.open();
  assert.equal(sockets[0].sent[0].token, 'pasted-token');
  assert.ok(!fs.readFileSync(WORKER, 'utf8').includes('pasted-token'));
});

test('the default port is 8765 and nothing connects before pairing', async () => {
  const { context, sockets } = harness({});
  await context.connect();
  assert.equal(sockets.length, 0, 'no token, no connection');
  const paired = harness({ jarvisToken: 't' });
  await paired.context.connect();
  assert.equal(paired.sockets[0].url, 'ws://127.0.0.1:8765');
});

test('hello identifies the profile, protocol version and capabilities', async () => {
  const { context, sockets, stored, Socket } = harness();
  await context.connect();
  sockets[0].readyState = Socket.OPEN;
  sockets[0].listeners.open();

  const hello = sockets[0].sent[0];
  assert.equal(hello.type, 'hello');
  assert.equal(hello.protocolVersion, 2);
  assert.equal(hello.clientId, stored.jarvisClientId);
  assert.equal(hello.extensionVersion, '1.0.0');
  for (const name of ['open', 'click', 'type', 'inspect', 'describe', 'upload']) {
    assert.ok(hello.capabilities.includes(name), name);
  }
});

test('reconnect backoff is bounded and resets after a successful handshake', async () => {
  const { context, sockets, timers, Socket } = harness();
  await context.connect();
  for (let attempt = 0; attempt < 7; attempt += 1) {
    const current = sockets.at(-1);
    current.readyState = Socket.CLOSED;
    current.listeners.close({ code: 1006 });
    const timer = timers.at(-1);
    assert.ok(timer.delay <= 30000);
    timer.fn();
    await tick();
  }
  assert.equal(timers.at(-1).delay, 30000);

  const current = sockets.at(-1);
  current.readyState = Socket.OPEN;
  current.listeners.open();
  await current.listeners.message({ data: JSON.stringify({ type: 'ready', selected: true }) });
  assert.equal(vm.runInContext('reconnectAttempt', context), 0);
  assert.equal(vm.runInContext('handshakeReady', context), true);
});

test('a rejected token stops the retry loop until the user pastes a new one', async () => {
  const { context, sockets, timers, Socket } = harness();
  await context.connect();
  sockets[0].readyState = Socket.CLOSED;
  const before = timers.length;
  sockets[0].listeners.close({ code: 1008 });
  assert.equal(timers.length, before, 'no reconnect timer after an authentication failure');
  assert.equal(vm.runInContext('authFailed', context), true);
  assert.match(vm.runInContext('lastError', context), /pairing token/);
});

test('command failures are sent back with their error type', async () => {
  const { context, sockets, Socket } = harness();
  await context.connect();
  const socket = sockets[0];
  socket.readyState = Socket.OPEN;
  socket.listeners.open();
  await socket.listeners.message({ data: JSON.stringify({ type: 'ready', selected: true }) });
  // An unsupported action fails inside handleCommand with a typed error.
  await socket.listeners.message({ data: JSON.stringify({ type: 'command', id: 'c1', action: 'teleport', parameters: {} }) });
  const reply = socket.sent.at(-1);
  assert.equal(reply.type, 'result');
  assert.equal(reply.id, 'c1');
  assert.equal(reply.ok, false);
  assert.equal(typeof reply.errorType, 'string');
});
