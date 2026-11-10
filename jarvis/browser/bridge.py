"""Loopback WebSocket bridge to the Jarvis Browser Control Chrome extension.

The extension runs inside the user's signed-in Chrome and dials in; Jarvis never
dials out. The server binds 127.0.0.1 only, accepts only ``chrome-extension://``
origins, and requires the first message to be a ``hello`` carrying the pairing
token (compared in constant time).

Wire protocol (version 2), JSON text frames:

    extension -> bridge   {"type": "hello", "token", "protocolVersion", "clientId",
                           "extensionVersion", "capabilities": [...]}
                          {"type": "heartbeat", "clientId"}
                          {"type": "result", "id", "ok", "result" | "error", "errorType"}
    bridge -> extension   {"type": "ready", "selected": bool, "protocolVersion"}
                          {"type": "selected", "selected": bool}
                          {"type": "upgrade_required", "error"}
                          {"type": "heartbeat_ack"}
                          {"type": "command", "id", "action", "parameters"}

With several Chrome profiles connected, one is *selected* (the first to connect;
when it leaves, the next one takes over). Commands go to the selected client.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import logging
import re
import secrets
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from websockets.asyncio.server import Server, ServerConnection, serve
from websockets.exceptions import ConnectionClosed

if TYPE_CHECKING:
    from jarvis.config import Settings

log = logging.getLogger(__name__)

PROTOCOL_VERSION = 2
HELLO_TIMEOUT = 5.0
HEARTBEAT_TIMEOUT = 75.0
MAX_MESSAGE_BYTES = 16 * 1024 * 1024
ORIGIN_PATTERN = re.compile(r"chrome-extension://[a-p]{32}")
CLIENT_ID_PATTERN = re.compile(r"[A-Za-z0-9._:-]{8,128}")

# errorType values the extension may send; "uncertain_action" is also produced
# here when the socket drops after a command was delivered.
UNCERTAIN = "uncertain_action"
EXTENSION_DIR = Path(__file__).resolve().parents[2] / "extensions" / "jarvis-browser-control"


class BridgeError(Exception):
    """Base class for bridge failures."""


class BridgeNotConnected(BridgeError):
    """No authenticated extension is connected."""


class BridgeStartError(BridgeError):
    """The server could not start (for example the port is taken)."""


class BridgeTimeout(BridgeError):
    """The extension did not answer in time; the action may still have happened."""


class BridgeCommandError(BridgeError):
    """The extension reported a failure (or the link broke mid-command)."""

    def __init__(self, error_type: str, message: str) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.message = message

    @property
    def uncertain(self) -> bool:
        return self.error_type == UNCERTAIN


def token_path(settings: "Settings") -> Path:
    return settings.data_dir / "browser-token"


def load_or_create_token(settings: "Settings") -> str:
    """Read the pairing token, creating it on first use."""
    path = token_path(settings)
    try:
        existing = path.read_text(encoding="utf-8").strip()
    except OSError:
        existing = ""
    if existing:
        return existing
    token = secrets.token_urlsafe(32)
    path.write_text(token, encoding="utf-8")
    try:
        path.chmod(0o600)  # no-op on Windows, where the profile folder is already private
    except OSError:
        pass
    return token


def extension_dir() -> Path:
    return EXTENSION_DIR


@dataclass
class _Client:
    client_id: str
    extension_version: str
    capabilities: frozenset[str]
    connection: ServerConnection
    connected_at: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


@dataclass
class _Pending:
    future: "asyncio.Future[Any]"
    client_id: str


class BrowserBridge:
    def __init__(self, port: int, token: str, *, heartbeat_timeout: float = HEARTBEAT_TIMEOUT) -> None:
        self.port = port
        self._token = token.encode()
        self._heartbeat_timeout = heartbeat_timeout
        self._clients: dict[str, _Client] = {}
        self._selected: str | None = None
        self._pending: dict[str, _Pending] = {}
        self._server: Server | None = None
        self._start_task: "asyncio.Task[None] | None" = None
        self._last_handshake_error: str | None = None

    # -- lifecycle -----------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._server is not None

    async def start(self) -> None:
        """Start listening. Idempotent and safe to call concurrently."""
        if self._server is not None:
            return
        if self._start_task is None:
            self._start_task = asyncio.ensure_future(self._start())
        try:
            await asyncio.shield(self._start_task)
        except BridgeStartError:
            self._start_task = None  # allow a retry after the port frees up
            raise

    async def _start(self) -> None:
        try:
            self._server = await serve(
                self._handle,
                "127.0.0.1",
                self.port,
                origins=[ORIGIN_PATTERN],
                max_size=MAX_MESSAGE_BYTES,
                ping_interval=20,
                ping_timeout=20,
            )
        except OSError as exc:
            raise BridgeStartError(
                f"Port {self.port} is not available for the Chrome bridge ({exc.strerror or exc}). "
                "Another program (or another Jarvis) is using it. Free the port, or set "
                "JARVIS_BROWSER_PORT to a different number and enter the same port in the "
                "extension popup."
            ) from exc
        log.info("browser bridge listening on 127.0.0.1:%s", self.port)

    async def close(self) -> None:
        server, self._server, self._start_task = self._server, None, None
        clients = list(self._clients.values())
        self._clients.clear()
        self._selected = None
        self._fail_all_pending(BridgeNotConnected("Jarvis is shutting down."))
        if server is not None:
            server.close()
        for client in clients:
            await _close_quietly(client.connection, 1001, "Jarvis is shutting down")
        if server is not None:
            await server.wait_closed()

    # -- state -----------------------------------------------------------------
    @property
    def connected(self) -> bool:
        return self._selected is not None and self._selected in self._clients

    def status(self) -> dict[str, Any]:
        """Non-secret diagnostics."""
        return {
            "running": self.running,
            "port": self.port,
            "connected": self.connected,
            "selected_client": self._selected,
            "clients": [
                {
                    "client_id": c.client_id,
                    "selected": c.client_id == self._selected,
                    "extension_version": c.extension_version,
                    "capabilities": sorted(c.capabilities),
                    "connected_at": c.connected_at,
                    "last_seen": c.last_seen,
                }
                for c in self._clients.values()
            ],
            "pending_commands": len(self._pending),
            "handshake_error": self._last_handshake_error,
        }

    async def select_client(self, client_id: str) -> None:
        if client_id not in self._clients:
            raise BridgeNotConnected(f"Chrome client '{client_id}' is not connected.")
        previous, self._selected = self._selected, client_id
        await self._announce_selection(previous)

    async def _announce_selection(self, previous: str | None) -> None:
        for cid in {previous, self._selected} - {None}:
            client = self._clients.get(cid)  # type: ignore[arg-type]
            if client is not None:
                await _send_quietly(client.connection, {"type": "selected", "selected": cid == self._selected})

    # -- commands ----------------------------------------------------------------
    async def command(
        self, action: str, parameters: dict[str, Any] | None = None, timeout: float = 30.0
    ) -> dict[str, Any]:
        """Send one command to the selected extension and wait for its reply.

        Raises BridgeNotConnected, BridgeTimeout or BridgeCommandError. Replies
        are matched by id, so a late reply to a timed-out command is ignored.
        """
        client = self._clients.get(self._selected or "")
        if client is None:
            raise BridgeNotConnected("The Jarvis Browser Control extension is not connected.")
        if action not in client.capabilities:
            raise BridgeCommandError(
                "unsupported",
                f"The loaded extension (version {client.extension_version}) does not support "
                f"'{action}'. Reload the unpacked extension from the Jarvis folder.",
            )
        command_id = uuid.uuid4().hex
        future: asyncio.Future[Any] = asyncio.get_running_loop().create_future()
        self._pending[command_id] = _Pending(future, client.client_id)
        try:
            try:
                await client.connection.send(
                    json.dumps({"type": "command", "id": command_id, "action": action, "parameters": parameters or {}})
                )
            except ConnectionClosed as exc:
                raise BridgeNotConnected("The extension disconnected before the command was sent.") from exc
            try:
                return await asyncio.wait_for(future, timeout)
            except asyncio.TimeoutError as exc:
                raise BridgeTimeout(
                    f"The browser did not answer '{action}' within {timeout:.0f}s. "
                    "The action may or may not have happened."
                ) from exc
        finally:
            self._pending.pop(command_id, None)

    def _fail_all_pending(self, error: Exception) -> None:
        for pending in list(self._pending.values()):
            if not pending.future.done():
                pending.future.set_exception(error)

    def _fail_pending_for(self, client_id: str) -> None:
        error = BridgeCommandError(
            UNCERTAIN,
            "The browser connection dropped after the command was sent; the action may have "
            "happened. Inspect the page before retrying.",
        )
        for pending in list(self._pending.values()):
            if pending.client_id == client_id and not pending.future.done():
                pending.future.set_exception(error)

    # -- connection handling -------------------------------------------------------
    async def _handle(self, connection: ServerConnection) -> None:
        client = await self._handshake(connection)
        if client is None:
            return
        try:
            while True:
                raw = await asyncio.wait_for(connection.recv(), self._heartbeat_timeout)
                client.last_seen = time.time()
                self._dispatch(client, raw)
                if _message_type(raw) == "heartbeat":
                    await connection.send('{"type":"heartbeat_ack"}')
        except asyncio.TimeoutError:
            log.warning("browser extension %s stopped sending heartbeats", client.client_id)
            await _close_quietly(connection, 1011, "heartbeat timeout")
        except ConnectionClosed:
            pass
        finally:
            await self._remove(client)

    def _dispatch(self, client: _Client, raw: str | bytes) -> None:
        try:
            message = json.loads(raw)
        except (ValueError, TypeError):
            return
        if not isinstance(message, dict) or message.get("type") != "result":
            return
        pending = self._pending.get(str(message.get("id") or ""))
        if pending is None or pending.client_id != client.client_id or pending.future.done():
            return
        if message.get("ok"):
            result = message.get("result")
            pending.future.set_result(result if isinstance(result, dict) else {"value": result})
        else:
            pending.future.set_exception(
                BridgeCommandError(
                    str(message.get("errorType") or "failed"),
                    str(message.get("error") or "The browser action failed."),
                )
            )

    async def _handshake(self, connection: ServerConnection) -> _Client | None:
        try:
            raw = await asyncio.wait_for(connection.recv(), HELLO_TIMEOUT)
            hello = json.loads(raw)
        except (asyncio.TimeoutError, ConnectionClosed, ValueError, TypeError):
            await _close_quietly(connection, 1008, "hello required")
            return None
        if not isinstance(hello, dict) or hello.get("type") != "hello":
            await _close_quietly(connection, 1008, "hello required")
            return None
        supplied = str(hello.get("token") or "").encode()
        if not hmac.compare_digest(supplied, self._token):
            log.warning("browser bridge: rejected a connection with a wrong token")
            await _close_quietly(connection, 1008, "authentication failed")
            return None
        problem = self._validate_hello(hello)
        if problem:
            self._last_handshake_error = problem
            await _send_quietly(connection, {"type": "upgrade_required", "error": problem})
            await _close_quietly(connection, 1002, "incompatible extension")
            return None
        return await self._register(connection, hello)

    @staticmethod
    def _validate_hello(hello: dict[str, Any]) -> str | None:
        if hello.get("protocolVersion") != PROTOCOL_VERSION:
            return (
                f"The extension speaks protocol {hello.get('protocolVersion')!r}, Jarvis expects "
                f"{PROTOCOL_VERSION}. Reload the unpacked extension from the Jarvis folder."
            )
        caps = hello.get("capabilities")
        if (
            not CLIENT_ID_PATTERN.fullmatch(str(hello.get("clientId") or ""))
            or not isinstance(caps, list)
            or not all(isinstance(c, str) and c for c in caps)
        ):
            return "The extension's hello message is incomplete. Reload the unpacked extension."
        return None

    async def _register(self, connection: ServerConnection, hello: dict[str, Any]) -> _Client:
        client_id = str(hello["clientId"])
        previous = self._clients.get(client_id)
        client = _Client(
            client_id=client_id,
            extension_version=str(hello.get("extensionVersion") or "unknown")[:64],
            capabilities=frozenset(hello["capabilities"]),
            connection=connection,
        )
        self._clients[client_id] = client
        if self._selected is None:
            self._selected = client_id
        self._last_handshake_error = None
        await _send_quietly(
            connection,
            {"type": "ready", "protocolVersion": PROTOCOL_VERSION, "selected": self._selected == client_id},
        )
        if previous is not None and previous.connection is not connection:
            self._fail_pending_for(client_id)
            await _close_quietly(previous.connection, 1000, "same Chrome profile reconnected")
        log.info("browser extension %s connected (selected=%s)", client_id, self._selected == client_id)
        return client

    async def _remove(self, client: _Client) -> None:
        if self._clients.get(client.client_id) is not client:
            return  # replaced by a newer connection from the same profile
        del self._clients[client.client_id]
        self._fail_pending_for(client.client_id)
        if self._selected == client.client_id:
            self._selected = next(iter(self._clients), None)
            await self._announce_selection(None)
        log.info("browser extension %s disconnected", client.client_id)


def _message_type(raw: str | bytes) -> str | None:
    try:
        message = json.loads(raw)
    except (ValueError, TypeError):
        return None
    return message.get("type") if isinstance(message, dict) else None


async def _send_quietly(connection: ServerConnection, payload: dict[str, Any]) -> None:
    try:
        await connection.send(json.dumps(payload))
    except ConnectionClosed:
        pass


async def _close_quietly(connection: ServerConnection, code: int, reason: str) -> None:
    try:
        await connection.close(code, reason)
    except Exception:  # already closed or half-open; nothing useful to do
        pass


# -- process-wide instance ------------------------------------------------------
# app.py does not own the bridge, so it is created lazily on first tool use,
# inside whichever event loop is running. A new loop (tests, a restarted frontend)
# gets a fresh bridge; the old one died with its loop.
_state: tuple[asyncio.AbstractEventLoop, int, BrowserBridge] | None = None


async def get_bridge(settings: "Settings") -> BrowserBridge:
    """Return the shared bridge, starting its server on first use."""
    global _state
    loop = asyncio.get_running_loop()
    port = settings.browser_bridge_port
    if _state is None or _state[0] is not loop or _state[1] != port:
        _state = (loop, port, BrowserBridge(port, load_or_create_token(settings)))
    bridge = _state[2]
    await bridge.start()
    return bridge


async def shutdown() -> None:
    """Stop the shared bridge, if one was started."""
    global _state
    state, _state = _state, None
    if state is not None and state[0] is asyncio.get_running_loop():
        await state[2].close()
