// Chrome API contract tests. No signed-in browser or network is touched.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const { test } = require('node:test');

const WORKER = path.join(__dirname, '..', '..', 'extensions', 'jarvis-browser-control', 'service-worker.js');

function runtime() {
  const events = [];
  const tabs = [
    { id: 10, groupId: 1, windowId: 1, active: false, status: 'complete', url: 'https://fixture.example/page', title: 'Fixture' },
    { id: 20, groupId: -1, windowId: 1, active: true, status: 'complete', url: 'https://personal.example', title: 'Personal' },
  ];
  const listener = { addListener() {}, removeListener() {} };
  const context = vm.createContext({
    importScripts() {}, setTimeout, clearTimeout, setInterval, clearInterval, console, Date, Math,
    chrome: {
      runtime: { onInstalled: listener, onStartup: listener, onMessage: listener, getManifest: () => ({ version: '1.0.0' }) },
      storage: { local: { get: async () => ({}), set: async () => {} }, onChanged: listener },
      alarms: { create() {}, onAlarm: listener },
      tabGroups: {
        query: async () => [{ id: 1, title: 'JARVIS', windowId: 1 }],
        get: async () => ({ id: 1, title: 'JARVIS', windowId: 1 }),
      },
      tabs: {
        onUpdated: listener,
        query: async (filter) => tabs.filter((tab) => filter.groupId === undefined || tab.groupId === filter.groupId),
        get: async (id) => tabs.find((tab) => tab.id === id),
        update: async (id, update) => { events.push(['update', id]); Object.assign(tabs.find((tab) => tab.id === id), update); },
        remove: async (id) => { events.push(['remove', id]); },
        captureVisibleTab: async () => { events.push(['capture']); return 'data:image/png;base64,fixture'; },
        group: async ({ tabIds }) => { events.push(['group', tabIds]); tabs.find((tab) => tab.id === tabIds).groupId = 1; },
        create: async () => { events.push(['create']); const tab = { id: 99, groupId: -1, windowId: 1, status: 'complete' }; tabs.push(tab); return tab; },
      },
      windows: { update: async () => {} },
      scripting: {
        executeScript: async ({ args }) => {
          events.push(['page', args[0]]);
          return [{ result: args[0] === 'click_point' ? { x: 5, y: 6, label: 'Go' } : { label: 'Field', message: 'observed' } }];
        },
      },
      debugger: {
        attach: async (target) => events.push(['attach', target.tabId]),
        sendCommand: async (target, method, params) => events.push([method, target.tabId, params]),
        detach: async (target) => events.push(['detach', target.tabId]),
      },
    },
  });
  vm.runInContext(fs.readFileSync(WORKER, 'utf8'), context);
  return { context, events, tabs };
}

test('explicit tab IDs reject unrelated personal tabs', async () => {
  const { context, events } = runtime();
  await assert.rejects(context.handleCommand('read', { tab_id: 20 }), /not in the JARVIS tab group/);
  assert.equal(events.length, 0);
  const result = await context.handleCommand('read', { tab_id: 10 });
  assert.equal(result.tabId, 10);
  assert.equal(result.url, 'https://fixture.example/page');
});

test('the tab group is the only thing listed', async () => {
  const { context } = runtime();
  const listed = await context.handleCommand('list_tabs', {});
  assert.deepEqual(listed.tabs.map((tab) => tab.id), [10]);
});

test('listing an empty group succeeds without selecting or opening a tab', async () => {
  const { context, tabs, events } = runtime();
  tabs.splice(0, 1);
  const result = await context.handleCommand('list_tabs', {});
  assert.deepEqual(result.tabs, []);
  assert.equal(events.length, 0);
});

test('only http and https can be opened', async () => {
  const { context, events } = runtime();
  for (const url of ['file:///C:/secret.txt', 'chrome://settings', 'javascript:alert(1)', 'about:blank']) {
    await assert.rejects(context.handleCommand('open', { url, new_tab: true }), /http and https/);
  }
  assert.equal(events.length, 0);
});

test('failures carry a type the bridge can pass on', async () => {
  const { context } = runtime();
  await assert.rejects(context.handleCommand('read', { tab_id: 20 }), (error) => error.type === 'not_in_group');
  await assert.rejects(context.handleCommand('teleport', {}), (error) => error.type === 'unsupported');
});

test('screenshots focus the managed tab before capture', async () => {
  const { context, events } = runtime();
  await context.handleCommand('screenshot', { tab_id: 10 });
  assert.deepEqual(events, [['update', 10], ['capture']]);
});

test('only popups from the acted-on tab are adopted', async () => {
  const { context, tabs, events } = runtime();
  tabs.push({ id: 30, groupId: -1, windowId: 1, openerTabId: 20 });
  tabs.push({ id: 40, groupId: -1, windowId: 1, openerTabId: 10 });
  await context.adoptOpenedTabs(new Set([10, 20]), 10);
  assert.deepEqual(events.filter((event) => event[0] === 'group'), [['group', 40]]);
});

test('clicks use real mouse events and the debugger always detaches', async () => {
  const { context, events } = runtime();
  await context.handleCommand('click', { tab_id: 10, ref: 3 });
  const mouse = events.filter((event) => event[0] === 'Input.dispatchMouseEvent').map((event) => event[2].type);
  assert.deepEqual(mouse, ['mouseMoved', 'mousePressed', 'mouseReleased']);
  assert.deepEqual(events.filter((event) => ['attach', 'detach'].includes(event[0])).map((e) => e[0]), ['attach', 'detach']);
});

test('a click that fails after the press is reported as an uncertain action', async () => {
  const { context, events } = runtime();
  context.chrome.debugger.sendCommand = async (target, method, params) => {
    if (params?.type === 'mouseReleased') throw new Error('target closed');
    events.push([method, params?.type]);
  };
  await assert.rejects(
    context.handleCommand('click', { tab_id: 10, ref: 3 }),
    (error) => error.type === 'uncertain_action',
  );
  assert.deepEqual(events.at(-1), ['detach', 10]);
});

test('typing inserts text through the debugger, presses Enter on submit, and detaches on failure', async () => {
  const { context, events } = runtime();
  await context.handleCommand('type', { tab_id: 10, ref: 3, text: 'hello', submit: true });
  assert.ok(events.some((event) => event[0] === 'Input.insertText' && event[2].text === 'hello'));
  const keys = events.filter((event) => event[0] === 'Input.dispatchKeyEvent').map((event) => event[2].type);
  assert.deepEqual(keys, ['keyDown', 'keyUp']);
  context.chrome.debugger.sendCommand = async () => { throw new Error('input failed'); };
  await assert.rejects(context.handleCommand('type', { tab_id: 10, ref: 3, text: 'x' }), (e) => e.type === 'uncertain_action');
  assert.deepEqual(events.at(-1), ['detach', 10]);
});

test('page-side refusals come back as typed errors', async () => {
  const { context } = runtime();
  context.chrome.scripting.executeScript = async () => [{ result: { __error: { type: 'refused', message: 'password field' } } }];
  await assert.rejects(context.handleCommand('type', { tab_id: 10, ref: 3, text: 'x' }), (e) => e.type === 'refused');
});
