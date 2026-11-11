"""Voice frontend: always-open microphone, spoken replies, compact status display."""

from __future__ import annotations

import asyncio
import logging
from collections import deque

from rich.console import Console, Group
from rich.live import Live
from rich.text import Text

from jarvis.interfaces.app import App
from jarvis.proactive.runtime import Background, start_background
from jarvis.proactive.scheduler import ReminderWatcher
from jarvis.voice.session import (
    ConfirmRequested,
    ConfirmResolved,
    Heard,
    Info,
    Levels,
    MicMuted,
    Spoke,
    StateChanged,
    ToolActivity,
    TurnFinished,
    UserText,
    VoiceEvent,
    VoiceSession,
    VoiceState,
)

log = logging.getLogger(__name__)

STATE_STYLE = {
    VoiceState.IDLE: "dim green",
    VoiceState.LISTENING: "bold yellow",
    VoiceState.THINKING: "bold magenta",
    VoiceState.SPEAKING: "bold cyan",
}
GREETING = "Online and ready."
BARS = " ▁▂▃▄▅▆▇█"


def meter(level: float, width: int = 12) -> str:
    """Level bar; sqrt scale so quiet speech is still visible."""
    filled = int(round(min(1.0, max(0.0, level) ** 0.5) * width))
    return "█" * filled + "·" * (width - filled)


class VoiceDisplay:
    """Renders VoiceEvents as a few lines; rich.Live calls __rich__ on every refresh."""

    def __init__(self, wake_word: str) -> None:
        self.wake_word = wake_word
        self.state = VoiceState.IDLE
        self.mic = 0.0
        self.speaker = 0.0
        self.heard: Heard | None = None
        self.reply: list[str] = []
        self.tools: deque[str] = deque(maxlen=4)
        self.info = ""
        self.muted = False
        self.confirm = ""  # description of the action awaiting yes/no, else ""
        self.footer = ""

    def handle(self, event: VoiceEvent) -> None:
        if isinstance(event, StateChanged):
            if event.state is VoiceState.THINKING:
                self.reply.clear()
                self.tools.clear()
                self.footer = ""
            self.state = event.state
        elif isinstance(event, Levels):
            self.mic, self.speaker = event.mic, event.speaker
        elif isinstance(event, Heard):
            self.heard = event
        elif isinstance(event, UserText):
            self.heard = Heard(event.text, True, 1.0)
        elif isinstance(event, ConfirmRequested):
            self.confirm = event.description
        elif isinstance(event, ConfirmResolved):
            self.confirm = ""
        elif isinstance(event, MicMuted):
            self.muted = event.muted
        elif isinstance(event, TurnFinished):
            self.footer = f"{event.seconds:.1f}s, {event.tool_calls} tool calls, {event.model}"
        elif isinstance(event, Spoke):
            self.reply.append(event.text)
        elif isinstance(event, ToolActivity):
            self._tool(event)
        elif isinstance(event, Info):
            self.info = event.text

    def _tool(self, event: ToolActivity) -> None:
        if not event.done:
            self.tools.append(f"  → {event.name}")
        else:
            mark = "✓" if event.ok else "✗"
            self.tools.append(f"  {mark} {event.name}")

    def __rich__(self) -> Group:
        head = Text()
        head.append(f"● {self.state.value.upper():<10}", style=STATE_STYLE[self.state])
        head.append(f" mic {meter(self.mic)} ", style="green")
        head.append(f"out {meter(self.speaker)}", style="cyan")
        if self.muted:
            head.append("  MUTED", style="bold red")
        lines: list[Text] = [head]
        if self.heard is not None:
            style = "" if self.heard.addressed else "dim"
            suffix = "" if self.heard.addressed else f"  (not for {self.wake_word})"
            lines.append(Text(f"  you: {self.heard.text}{suffix}", style=style))
        for tool in self.tools:
            lines.append(Text(tool, style="dim"))
        if self.reply:
            lines.append(Text(f"  jarvis: {' '.join(self.reply)}", style="bold"))
        if self.confirm:
            lines.append(Text(f"  confirm? {self.confirm} (say yes or no)", style="bold yellow"))
        if self.footer:
            lines.append(Text(f"  {self.footer}", style="dim"))
        if self.info:
            lines.append(Text(f"  {self.info}", style="yellow"))
        return Group(*lines)


def _missing_voice_message(exc: ImportError) -> str:
    from jarvis.interfaces.cli import voice_install_message

    return voice_install_message(exc.name)


async def run_voice(app: App, console: Console | None = None) -> int:
    console = console or Console()
    try:
        from jarvis.voice.audio import open_audio
        from jarvis.voice.stt import build_transcriber
        from jarvis.voice.tts import build_synthesizer
        from jarvis.voice.vad import Endpointer, build_vad
    except ImportError as exc:
        console.print(Text(_missing_voice_message(exc), style="yellow"))
        return 2

    settings = app.settings
    try:
        mic, speaker = open_audio(settings)
        endpointer = Endpointer(build_vad(), end_silence_ms=settings.end_silence_ms)
        transcriber = build_transcriber(settings)
        synthesizer = build_synthesizer(settings)
    except ImportError as exc:
        console.print(Text(_missing_voice_message(exc), style="yellow"))
        return 2
    except Exception as exc:
        log.exception("could not set up voice mode")
        console.print(Text(f"Could not start voice mode: {exc}", style="red"))
        return 1

    display = VoiceDisplay(settings.wake_word)
    voice_model = app.use_voice_model()
    if voice_model:
        console.print(Text(f"Voice turns use {voice_model}.", style="dim"))
    session = VoiceSession(app.agent, mic, speaker, endpointer, transcriber, synthesizer, settings, display.handle)
    console.print(Text(f"Loading {transcriber.name} and {synthesizer.name} ...", style="dim"))
    log.info("Speech engines: recognition %s; synthesis %s", transcriber.name, synthesizer.name)
    try:
        await asyncio.gather(transcriber.warm_up(), synthesizer.warm_up())
        await session.prepare()
    except ImportError as exc:
        console.print(Text(_missing_voice_message(exc), style="yellow"))
        return 2
    except Exception as exc:
        log.exception("speech model warm-up failed")
        console.print(Text(f"Could not load the speech models: {exc}", style="red"))
        return 1

    wake = "" if not settings.require_wake_word else f" Say \"{settings.wake_word}\" to talk to me."
    console.print(Text(f"Voice mode is live.{wake} Ctrl+C to quit.", style="cyan"))
    watcher = ReminderWatcher(app.store, session.say, session.is_idle)
    background = Background(None, None)
    task = asyncio.current_task()
    try:
        with Live(display, console=console, refresh_per_second=12, transient=False):
            watcher.start()
            background = start_background(
                app, session.say, session.is_idle, spoken=True, output_mode="voice",
                turn_active=lambda: session.state is VoiceState.THINKING, say_busy=session.say,
            )
            await session.run(greeting=GREETING)
    except (KeyboardInterrupt, asyncio.CancelledError):
        if task is not None:
            task.uncancel()  # asyncio.run turns Ctrl+C into a cancel of the main task
    finally:
        await background.stop()
        await watcher.stop()
    console.print(Text("Goodbye.", style="dim"))
    return 0
