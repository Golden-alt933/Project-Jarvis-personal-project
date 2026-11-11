"""Terminal frontend: streaming chat over the agent's events."""

from __future__ import annotations

import argparse
import asyncio
import logging
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from pydantic import ValidationError
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from jarvis.agent.events import AgentEvent, Notice, TextChunk, ToolFinished, ToolStarted, TurnDone
from jarvis.config import Settings, config_files, get_settings, user_config_file
from jarvis.interfaces.app import App, build_app
from jarvis.llm.providers import ConfigError, describe
from jarvis.llm.types import LLMClient, ToolCall
from jarvis.proactive.scheduler import ReminderWatcher
from jarvis.tools.base import Risk

EXIT_WORDS = {"exit", "quit"}
STOP_REASONS = {
    "max_steps": "Stopped: this turn reached the step limit.",
    "time_budget": "Stopped: this turn ran out of time.",
    "cancelled": "Cancelled.",
    "error": "Stopped: something went wrong (details are in the log file).",
}
HELP_TEXT = """\
/help          this list
/new           start a new conversation
/local         use the local Ollama model
/cloud         go back to the configured cloud model
/model         show the active model
/tools         list loaded tools by group, with risk
/memory        list saved facts
/forget <id>   delete a saved fact
/exit          quit (also: exit, quit, Ctrl+D)

Alt+Enter inserts a newline; pasted multi-line text is kept as one message.
Ctrl+C during a reply cancels it."""


# -- small formatting helpers ----------------------------------------------------
def summarize_args(arguments: dict[str, Any], limit: int = 80) -> str:
    """One-line `key=value` summary of tool arguments."""
    parts = []
    for key, value in (arguments or {}).items():
        text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
        parts.append(f"{key}={' '.join(text.split())}")
    line = ", ".join(parts)
    return line if len(line) <= limit else line[: limit - 1] + "…"


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def format_footer(done: TurnDone) -> str:
    parts = [f"{done.seconds:.1f}s", _plural(done.steps, "step"), _plural(done.tool_calls, "tool call"), done.model]
    if done.usage is not None:
        parts.append(f"{done.usage.prompt_tokens} in / {done.usage.completion_tokens} out tokens")
    return " · ".join(parts)


# -- rendering ---------------------------------------------------------------------
class TurnRenderer:
    """Turns AgentEvents into terminal output.

    Assistant text is shown with rich Live + Markdown. Anything else (tool lines,
    notices, confirmation panels) first finishes the live block so the two never
    fight over the cursor.
    """

    def __init__(self, console: Console, verbose: bool = False) -> None:
        self.console = console
        self.verbose = verbose
        self._buffer = ""
        self._live: Live | None = None
        self._streamed = False

    def handle(self, event: AgentEvent) -> None:
        if isinstance(event, TextChunk):
            self._text(event.text)
        elif isinstance(event, ToolStarted):
            self._tool_started(event.call)
        elif isinstance(event, ToolFinished):
            self._tool_finished(event)
        elif isinstance(event, Notice):
            self.pause()
            self.console.print(Text(event.text, style="yellow"))
        elif isinstance(event, TurnDone):
            self._done(event)

    # Live text block
    def _text(self, chunk: str) -> None:
        self._buffer += chunk
        self._streamed = True
        if not self.console.is_terminal:
            return  # no cursor control: print the finished block in pause()
        if self._live is None:
            self._live = Live(
                console=self.console, refresh_per_second=8, vertical_overflow="visible", transient=False
            )
            self._live.start()
        self._live.update(Markdown(self._buffer), refresh=False)

    def pause(self) -> None:
        """Finish the current text block (used before printing anything else)."""
        if self._live is not None:
            if self._buffer.strip():
                self._live.update(Markdown(self._buffer))
            self._live.stop()
            self._live = None
        elif self._buffer.strip():
            self.console.print(Markdown(self._buffer))
        self._buffer = ""

    close = pause

    # Tools
    def _tool_started(self, call: ToolCall) -> None:
        self.pause()
        line = Text("  → ", style="dim")
        line.append(call.name, style="dim bold")
        if call.arguments:
            line.append(f"  {summarize_args(call.arguments, 300 if self.verbose else 80)}", style="dim")
        self.console.print(line)

    def _tool_finished(self, event: ToolFinished) -> None:
        self.pause()
        ok = event.result.ok
        line = Text(f"  {'✓' if ok else '✗'} {event.call.name} {event.seconds:.1f}s", style="dim" if ok else "dim red")
        if not ok:
            content = event.result.content.strip()
            first = content.splitlines()[0] if content else ""
            line.append(f"  {first[:100]}")
        self.console.print(line)
        if self.verbose and event.result.content:
            preview = " ".join(event.result.content.split())[:200]
            self.console.print(Text(f"    {preview}", style="dim"))

    # End of turn
    def _done(self, done: TurnDone) -> None:
        self.pause()
        if not self._streamed and done.text.strip():
            style = "red" if done.stopped == "error" else ""
            self.console.print(Text(done.text, style=style))
        if done.stopped != "done":
            self.console.print(Text(STOP_REASONS.get(done.stopped, f"Stopped: {done.stopped}."), style="yellow"))
        self.console.print(Text(format_footer(done), style="dim"))
        self._streamed = False

    def interrupted(self) -> None:
        self.pause()
        self.console.print(Text("Cancelled.", style="yellow"))
        self._streamed = False


# -- confirmation ------------------------------------------------------------------
AskFn = Callable[[str], Awaitable[str]]


class Confirmer:
    """The `confirm` callback handed to the agent. Default answer is No."""

    def __init__(self, console: Console, renderer: TurnRenderer, ask: AskFn, auto_yes: bool = False) -> None:
        self.console = console
        self.renderer = renderer
        self.ask = ask
        self.auto_yes = auto_yes

    async def __call__(self, description: str) -> bool:
        self.renderer.pause()
        self.console.print(Panel(Text(description), title="Confirm action", border_style="yellow"))
        if self.auto_yes:
            self.console.print(Text("Allowed (--yes).", style="dim"))
            return True
        try:
            answer = await self.ask("Allow? [y/N] ")
        except (KeyboardInterrupt, EOFError):
            return False
        return answer.strip().lower() in ("y", "yes")


async def notify_line(console: Console, renderer: TurnRenderer, text: str) -> None:
    renderer.pause()
    console.print(Text(f"  … {text}", style="dim"))


# -- slash commands ----------------------------------------------------------------
def parse_command(line: str) -> tuple[str, str] | None:
    """'/forget 3' -> ('forget', '3'); anything that is not a slash command -> None."""
    stripped = line.strip()
    if not stripped.startswith("/") or stripped.startswith("//"):
        return None
    name, _, arg = stripped[1:].partition(" ")
    return name.lower(), arg.strip()


class SlashCommands:
    def __init__(self, app: App, console: Console) -> None:
        self.app = app
        self.console = console
        self.local = app.primary.is_local
        self._handlers: dict[str, Callable[[str], bool]] = {
            "help": self._help, "new": self._new, "local": self._local, "cloud": self._cloud,
            "model": self._model, "tools": self._tools, "memory": self._memory,
            "forget": self._forget, "exit": self._exit,
        }

    def dispatch(self, line: str) -> bool:
        """Run a slash command. Returns True when the session should end."""
        parsed = parse_command(line)
        if parsed is None:
            return False
        name, arg = parsed
        handler = self._handlers.get(name)
        if handler is None:
            self._say(f"Unknown command /{name}. Type /help.", "yellow")
            return False
        return handler(arg)

    def _say(self, text: str, style: str = "") -> None:
        self.console.print(Text(text, style=style))

    def _help(self, _: str) -> bool:
        self._say(HELP_TEXT)
        return False

    def _exit(self, _: str) -> bool:
        return True

    def _new(self, _: str) -> bool:
        self.app.agent.new_conversation()
        self._say("Started a new conversation.", "dim")
        return False

    def _has_local(self) -> bool:
        a = self.app.agent
        return a.primary.is_local or (a.fallback is not None and a.fallback.is_local)

    def _active(self) -> LLMClient:
        a = self.app.agent
        if self.local and not a.primary.is_local and a.fallback is not None:
            return a.fallback
        return a.primary

    def _local(self, _: str) -> bool:
        if not self._has_local():
            self._say("No local model is configured.", "yellow")
            return False
        self.app.agent.use_local(True)
        self.local = True
        self._say(f"Using the local model: {describe(self._active())}", "dim")
        return False

    def _cloud(self, _: str) -> bool:
        if self.app.primary.is_local:
            self._say(
                "The primary model is already the local one; set JARVIS_PROVIDER (see `jarvis --config-path`) to use a cloud model.",
                "yellow",
            )
            return False
        self.app.agent.use_local(False)
        self.local = False
        self._say(f"Using the cloud model: {describe(self.app.primary)}", "dim")
        return False

    def _model(self, _: str) -> bool:
        self._say(f"Active model: {describe(self._active())}")
        return False

    def _tools(self, _: str) -> bool:
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
        for col in ("group", "tool", "risk", "description"):
            table.add_column(col)
        for spec in self.app.registry.specs():
            risk = spec.risk.value if isinstance(spec.risk, Risk) else "per call"
            table.add_row(spec.group, spec.name, risk, spec.description.split(". ")[0][:70])
        self.console.print(table)
        return False

    def _memory(self, _: str) -> bool:
        facts = self.app.store.list_facts()
        if not facts:
            self._say("No saved facts yet.", "dim")
            return False
        for f in facts:
            self._say(f"{f.id:>4}  [{f.category}] {f.text}")
        return False

    def _forget(self, arg: str) -> bool:
        try:
            fact_id = int(arg)
        except ValueError:
            self._say("Usage: /forget <id>   (ids are shown by /memory)", "yellow")
            return False
        done = self.app.store.forget(fact_id)
        self._say(f"Forgot fact {fact_id}." if done else f"No fact with id {fact_id}.", "dim" if done else "yellow")
        return False


# -- startup -----------------------------------------------------------------------
def print_banner(console: Console, app: App, resumed: bool) -> None:
    body = Text()
    body.append("JARVIS", style="bold cyan")
    body.append(f"   {describe(app.primary)}", style="bold")
    body.append(f"\n{len(app.registry)} tools loaded · ", style="dim")
    body.append("resumed conversation" if resumed else "new conversation", style="dim")
    body.append(f"\ndata: {app.settings.data_dir}", style="dim")
    body.append("\n/help for commands · Ctrl+D to exit", style="dim")
    console.print(Panel(body, border_style="cyan", expand=False))


async def local_warning(app: App) -> str | None:
    """Warn when a local model will be needed but Ollama is not answering."""
    client = app.primary if app.primary.is_local else app.fallback
    check = getattr(client, "is_available", None)
    if client is None or check is None or await check():
        return None
    extra = "" if app.primary.is_local else " Until then the local fallback is unavailable."
    return f"Ollama is not ready for {client.model}. Start it (`ollama serve`) and run `ollama pull {client.model}`.{extra}"


def config_error_message(message: str) -> str:
    return (
        f"{message}\n\n"
        "Run `jarvis --setup` to choose a provider and enter a key. It is saved to\n"
        f"{user_config_file()}\n"
        "(`jarvis --config-path` shows the file in use), or run `jarvis --local` to use\n"
        "the Ollama model without any key."
    )


def apply_overrides(
    settings: Settings, provider: str | None, model: str | None, no_wake: bool = False
) -> Settings:
    update: dict[str, Any] = {}
    if provider:
        update["provider"] = provider
    if model:
        update["model"] = model
    if no_wake:
        update["require_wake_word"] = False
    return settings.model_copy(update=update) if update else settings


# -- running turns -----------------------------------------------------------------
async def run_one_turn(
    app: App, text: str, console: Console, renderer: TurnRenderer, confirm: Confirmer
) -> TurnDone | None:
    """Drive one agent turn. Returns the TurnDone, or None if cancelled by Ctrl+C."""
    agent = app.agent
    gen = agent.run_turn(text, confirm=confirm, notify=lambda t: notify_line(console, renderer, t))
    task = asyncio.current_task()
    done: TurnDone | None = None
    try:
        async for event in gen:
            renderer.handle(event)
            if isinstance(event, TurnDone):
                done = event
    except (KeyboardInterrupt, asyncio.CancelledError):
        agent.cancel()
        if task is not None:
            task.uncancel()  # asyncio.run turns Ctrl+C into a cancel of the main task
        renderer.interrupted()
        try:
            await gen.aclose()
        except BaseException:
            pass
        return None
    finally:
        renderer.close()
    return done


async def run_once(app: App, text: str, console: Console, *, yes: bool, verbose: bool) -> int:
    renderer = TurnRenderer(console, verbose)

    async def ask(_: str) -> str:
        return "n"

    confirm = Confirmer(console, renderer, ask, auto_yes=yes)
    done = await run_one_turn(app, text, console, renderer, confirm)
    if done is None:
        return 130
    return 1 if done.stopped == "error" else 0


@dataclass
class LineReader:
    """prompt_toolkit when attached to a terminal, plain stdin otherwise."""

    history_path: Path
    _session: Any = field(default=None, init=False)
    _confirm_session: Any = field(default=None, init=False)

    def _make_sessions(self) -> None:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.key_binding import KeyBindings

        kb = KeyBindings()

        @kb.add("enter")
        def _submit(event: Any) -> None:
            event.current_buffer.validate_and_handle()

        @kb.add("escape", "enter")
        def _newline(event: Any) -> None:
            event.current_buffer.insert_text("\n")

        self._session = PromptSession(
            history=FileHistory(str(self.history_path)),
            multiline=True,
            key_bindings=kb,
            prompt_continuation="  ",
        )
        self._confirm_session = PromptSession()

    async def read(self, prompt: str, *, confirm: bool = False) -> str:
        if not sys.stdin.isatty():
            return await asyncio.to_thread(self._plain)
        if self._session is None:
            self._make_sessions()
        from prompt_toolkit.patch_stdout import patch_stdout

        session = self._confirm_session if confirm else self._session
        with patch_stdout(raw=True):  # lets reminders print above the prompt
            return await session.prompt_async(prompt)

    @staticmethod
    def _plain() -> str:
        line = sys.stdin.readline()
        if not line:
            raise EOFError
        return line.rstrip("\n")


async def interactive(app: App, console: Console, *, verbose: bool) -> int:
    reader = LineReader(app.settings.data_dir / "history.txt")
    renderer = TurnRenderer(console, verbose)
    commands = SlashCommands(app, console)

    async def ask(prompt: str) -> str:
        return await reader.read(prompt, confirm=True)

    confirm = Confirmer(console, renderer, ask)
    busy = False  # reminders wait while a turn is streaming

    async def deliver(text: str) -> None:
        renderer.pause()
        console.print(Text(f"  ⏰ {text}", style="bold yellow"))

    watcher = ReminderWatcher(getattr(app, "store", None), deliver, lambda: not busy)
    watcher.start()
    warned = False
    try:
        while True:
            try:
                line = (await reader.read("› ")).strip()
                warned = False
            except KeyboardInterrupt:
                if warned:
                    return 0
                warned = True
                console.print(Text("Press Ctrl+C again (or Ctrl+D) to exit.", style="dim"))
                continue
            except EOFError:
                return 0
            if not line:
                continue
            if line.lower() in EXIT_WORDS:
                return 0
            if parse_command(line) is not None:
                if commands.dispatch(line):
                    return 0
                continue
            busy = True
            try:
                await run_one_turn(app, line, console, renderer, confirm)
            finally:
                busy = False
            console.print()
    finally:
        await watcher.stop()


# -- entry point -------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="jarvis", description="Jarvis, a personal desktop agent.")
    p.add_argument("--local", action="store_true", help="use the local Ollama model")
    p.add_argument("--new", action="store_true", help="start a fresh conversation instead of resuming")
    p.add_argument("--once", metavar="TEXT", help="run a single turn, print the reply and exit")
    p.add_argument("--yes", action="store_true", help="auto-approve confirmations (only with --once)")
    p.add_argument("--provider", help="override JARVIS_PROVIDER for this run")
    p.add_argument("--model", help="override JARVIS_MODEL for this run")
    p.add_argument("--verbose", action="store_true", help="show tool arguments and result previews")
    p.add_argument("--voice", action="store_true", help="talk to Jarvis: always-open microphone and spoken replies")
    p.add_argument("--hud", action="store_true", help="open the HUD window with voice")
    p.add_argument("--hud-demo", action="store_true", help="open the HUD with a scripted demo (no microphone or model)")
    p.add_argument("--no-voice", action="store_true", help="with --hud: text-only HUD, no microphone or speech models")
    p.add_argument("--devices", action="store_true", help="list audio devices and exit")
    p.add_argument("--browser-token", action="store_true", help="print the Chrome extension pairing token and folder, then exit")
    p.add_argument("--no-wake", action="store_true", help="voice: treat every utterance as addressed to Jarvis")
    p.add_argument("--setup", action="store_true", help="first-run setup: choose a provider, store the API key, run the doctor")
    p.add_argument("--doctor", action="store_true", help="check the installation and configuration, with a fix for each problem")
    p.add_argument("--offline", action="store_true", help="with --doctor or --setup: do not contact the hosted model")
    p.add_argument("--config-path", action="store_true", help="print the config file in use and exit")
    p.add_argument("--version", action="store_true", help="print the version and exit")
    return p


def package_version() -> str:
    from importlib import metadata

    try:
        return metadata.version("jarvis")
    except metadata.PackageNotFoundError:
        return "unknown (not installed as a package)"


def print_config_path() -> int:
    """Print the highest-priority config file that exists (plain stdout, scriptable)."""
    found = config_files()
    if found:
        print(found[0])
        for other in found[1:]:
            print(f"also read: {other} (lower priority)", file=sys.stderr)
        return 0
    print(user_config_file())
    print("(no config file yet; `jarvis --setup` creates it)", file=sys.stderr)
    return 0


def _force_utf8() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except (OSError, ValueError):
                pass


async def _interactive_main(app: App, console: Console, verbose: bool) -> int:
    warning = await local_warning(app)
    if warning:
        console.print(Text(warning, style="yellow"))
    return await interactive(app, console, verbose=verbose)


def voice_install_message(missing: str | None) -> str:
    needs = missing or "numpy, sounddevice, faster-whisper, piper"
    return (
        f"Voice mode needs the optional voice dependencies ({needs}).\n"
        "Install them with:\n\n    uv sync --extra voice\n\nthen run this command again."
    )


def print_devices(console: Console) -> int:
    try:
        from jarvis.voice.audio import list_devices

        listing = list_devices()
    except ImportError as exc:
        console.print(Text(voice_install_message(exc.name), style="yellow"))
        return 2
    console.print(Text(listing))
    return 0


def print_browser_token() -> int:
    """Print what the user needs to pair the Chrome extension (plain stdout, so it can be copied)."""
    from jarvis.browser.bridge import extension_dir, load_or_create_token

    settings = get_settings()
    print(f"Pairing token: {load_or_create_token(settings)}")
    print(f"Extension folder: {extension_dir()}")
    print(f"Bridge port: {settings.browser_bridge_port}")
    print("Load the folder at chrome://extensions (Developer mode, Load unpacked), then paste the token in its popup.")
    return 0


async def _start_browser_bridge(settings: Any) -> None:
    """Listen for the Chrome extension from startup, so it has connected by the time
    it is needed. Never fatal: without it only the browser tools are affected."""
    try:
        from jarvis.browser.bridge import get_bridge

        await get_bridge(settings)
    except ImportError:  # the browser extra is not installed
        pass
    except Exception as exc:  # e.g. the port is held by another Jarvis
        logging.getLogger(__name__).warning("browser bridge not started: %s", exc)


async def _with_cleanup(main: Any, settings: Any = None) -> int:
    """Run a frontend, then stop background services in the loop that started them."""
    if settings is not None:
        await _start_browser_bridge(settings)
    try:
        return await main
    finally:
        try:
            from jarvis.browser.bridge import shutdown as shutdown_bridge
        except ImportError:  # the browser extra is not installed
            pass
        else:
            await shutdown_bridge()


def _run_hud_mode(app: Any, console: Console, *, voice: bool, demo: bool) -> int:
    from jarvis.interfaces import hud

    if not hud.hud_available():
        console.print(Text(hud.hud_install_message(), style="yellow"))
        return 2
    # run_hud runs _with_cleanup inside its own Qt event loop
    return hud.run_hud(app, console, voice=voice, demo=demo)


def main(argv: list[str] | None = None) -> int:
    _force_utf8()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.voice and args.once is not None:
        parser.error("--voice and --once cannot be combined")
    if (args.hud or args.hud_demo) and (args.voice or args.once is not None):
        parser.error("--hud cannot be combined with --voice or --once")
    if args.version:
        print(f"jarvis {package_version()}")
        return 0
    if args.config_path:
        return print_config_path()
    if args.browser_token:
        return print_browser_token()
    console = Console()
    if args.setup:
        from jarvis.doctor import run_doctor
        from jarvis.setup_wizard import run_setup

        return run_setup(console, doctor=lambda settings: run_doctor(settings, console, offline=args.offline))
    if args.doctor:
        from jarvis.doctor import run_doctor

        try:
            return run_doctor(get_settings(), console, offline=args.offline)
        except ValidationError as exc:
            Console(stderr=True).print(Text(f"The configuration is not valid: {exc}", style="red"))
            return 2
    if args.devices:
        return print_devices(console)
    try:
        settings = apply_overrides(get_settings(), args.provider, args.model, args.no_wake)
        app = build_app(settings, force_local=args.local)
    except (ConfigError, ValidationError) as exc:
        Console(stderr=True).print(
            Panel(Text(config_error_message(str(exc))), title="Jarvis is not configured", border_style="red")
        )
        return 2

    try:
        resumed = False if args.new else app.resume()
        if args.once is not None:
            return asyncio.run(_with_cleanup(run_once(app, args.once, console, yes=args.yes, verbose=args.verbose), app.settings))
        if args.hud or args.hud_demo:
            return _run_hud_mode(app, console, voice=not args.no_voice, demo=args.hud_demo)
        if args.voice:
            try:
                from jarvis.interfaces.voice import run_voice
            except ImportError as exc:
                console.print(Text(voice_install_message(exc.name), style="yellow"))
                return 2
            return asyncio.run(_with_cleanup(run_voice(app, console), app.settings))
        if args.yes:
            console.print(Text("--yes only applies with --once; ignoring it.", style="yellow"))
        print_banner(console, app, resumed)
        return asyncio.run(_with_cleanup(_interactive_main(app, console, args.verbose), app.settings))
    except KeyboardInterrupt:
        return 130
    finally:
        app.close()


if __name__ == "__main__":
    sys.exit(main())
