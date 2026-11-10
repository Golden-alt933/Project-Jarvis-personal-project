# Jarvis Browser Control

A Manifest V3 Chrome extension that lets the local Jarvis assistant act in the
Chrome profile you already use, signed in as you. It creates a cyan tab group
named **JARVIS** and only ever reads, clicks or types in tabs inside that group.
Your other tabs are never touched.

## Install

1. In Chrome, open `chrome://extensions`.
2. Turn on **Developer mode** (top right).
3. Click **Load unpacked** and pick this folder (`extensions/jarvis-browser-control`).
4. In a terminal, run `jarvis --browser-token`. It prints the pairing token and
   this folder's path.
5. Click the Jarvis Browser Control icon in the Chrome toolbar (pin it from the
   puzzle-piece menu if it is hidden), paste the token into **Pairing token**,
   leave the port at `8765` (change it only if you set `JARVIS_BROWSER_PORT`),
   and click **Save and connect**.

The token and port are stored in this profile's extension storage. Nothing is
generated into this folder, so it is safe to keep it in git.

Jarvis starts its side of the connection the first time it uses a browser tool,
so the popup may say "Not connected" until you ask Jarvis to do something in
the browser. The extension keeps retrying with backoff and reconnects by itself
after Chrome suspends its background worker.

## Security model

- The bridge listens only on `127.0.0.1` and accepts only `chrome-extension://`
  origins. The first message must carry the pairing token.
- Commands are JSON in, JSON out. Clicks and typing are sent as real input
  events through the Chrome debugger protocol. While that happens Chrome shows
  a "Jarvis Browser Control started debugging this browser" banner; it goes
  away when the action finishes.
- Only `http` and `https` pages are opened. Credential, card and one-time-code
  fields are refused here as well as in Jarvis.

## Several Chrome profiles

Each profile has its own client id (shown in the popup). Jarvis uses the first
one that connects; the others stay on standby.

## Tests

`node --test tests/js/extension_connection.cjs tests/js/extension_runtime.cjs` from the repository root runs the contract tests for
this worker with a stubbed `chrome` API. No browser is needed.
