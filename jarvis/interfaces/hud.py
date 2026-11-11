"""HUD frontend: a translucent always-on-top window driven by the session's events.

Qt and asyncio share one thread through qasync's QEventLoop, so session events update
widgets directly and nothing here may block.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from typing import Any

from rich.console import Console
from rich.text import Text

from jarvis.interfaces.app import App

log = logging.getLogger(__name__)
GREETING = "Online and ready."


def hud_install_message() -> str:
    return (
        "The HUD needs the optional hud dependencies (PyQt6, qasync).\n"
        "Install them with:\n\n    uv sync --extra hud\n\nthen run this command again."
    )


def hud_available() -> bool:
    try:
        import PyQt6.QtWidgets  # noqa: F401
        import qasync  # noqa: F401
    except ImportError:
        return False
    return True


def run_hud(app: App, console: Console, *, voice: bool = True, demo: bool = False) -> int:
    if not hud_available():
        console.print(Text(hud_install_message(), style="yellow"))
        return 2
    import qasync
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication

    from jarvis.interfaces.cli import _with_cleanup

    qapp = QApplication.instance() or QApplication(sys.argv[:1])
    qapp.setQuitOnLastWindowClosed(False)
    loop = qasync.QEventLoop(qapp)
    asyncio.set_event_loop(loop)
    # Python only runs signal handlers when the interpreter gets control; this timer provides it.
    wake = QTimer()
    wake.start(250)
    try:
        with loop:
            return loop.run_until_complete(_with_cleanup(_hud_main(app, console, voice, demo), app.settings))
    except KeyboardInterrupt:
        return 130
    finally:
        wake.stop()


def active_model_text(app: App) -> str:
    """Description of the model the next turn will use (the local one when it is forced)."""
    from jarvis.llm.providers import describe

    agent = app.agent
    if agent.using_local:
        for client in (agent.primary, agent.fallback):
            if client is not None and client.is_local:
                return describe(client)
    return describe(agent.primary)


def _make_hooks(app: App, stop: Any, on_model_change: Any = None):
    from jarvis.hud.window import HudHooks

    agent = app.agent

    def has_local() -> bool:
        return agent.primary.is_local or (agent.fallback is not None and agent.fallback.is_local)

    def set_local(on: bool) -> None:
        agent.use_local(on)
        if on_model_change is not None:
            on_model_change(active_model_text(app))

    return HudHooks(
        new_conversation=agent.new_conversation,
        set_local=set_local,
        is_local=lambda: agent.using_local or agent.primary.is_local,
        local_available=has_local,
        quit=stop,
    )


async def _build_voice_session(app: App, on_event: Any):
    """Create audio and speech models off the UI thread (they can take seconds), then warm up."""
    from jarvis.voice.audio import open_audio
    from jarvis.voice.session import VoiceSession
    from jarvis.voice.stt import build_transcriber
    from jarvis.voice.tts import build_synthesizer
    from jarvis.voice.vad import Endpointer, build_vad

    s = app.settings

    def build():
        mic, speaker = open_audio(s)
        return (mic, speaker, Endpointer(build_vad(), end_silence_ms=s.end_silence_ms),
                build_transcriber(s), build_synthesizer(s))

    mic, speaker, endpointer, transcriber, synthesizer = await asyncio.to_thread(build)
    session = VoiceSession(app.agent, mic, speaker, endpointer, transcriber, synthesizer, s, on_event)
    await asyncio.gather(transcriber.warm_up(), synthesizer.warm_up())
    await session.prepare()
    return session


async def _make_session(app: App, voice: bool, demo: bool, on_event: Any, console: Console):
    if demo:
        from jarvis.hud.demo import DemoSession

        return DemoSession(on_event)
    if voice:
        try:
            return await _build_voice_session(app, on_event)
        except ImportError as exc:
            from jarvis.interfaces.cli import voice_install_message

            console.print(Text(voice_install_message(exc.name) + "\nContinuing in text-only mode.", style="yellow"))
        except Exception as exc:
            log.exception("voice setup failed; falling back to text")
            console.print(Text(f"Could not start voice ({exc}). Continuing in text-only mode.", style="yellow"))
    from jarvis.voice.text_session import TextSession  # lazy: written alongside the HUD

    return TextSession(app.agent, app.settings, on_event)


async def _hud_main(app: App, console: Console, voice: bool, demo: bool) -> int:
    from PyQt6.QtWidgets import QApplication

    from jarvis.hud.window import HudWindow
    from jarvis.proactive.scheduler import ReminderWatcher

    quit_event = asyncio.Event()
    holder: dict[str, Any] = {}

    def stop() -> None:
        quit_event.set()
        session = holder.get("session")
        if session is not None:
            session.stop()

    def model_changed(text: str) -> None:
        if "window" in holder:
            holder["window"].controller.set_model(text)

    window = HudWindow(app.settings.data_dir, _make_hooks(app, stop, model_changed))
    holder["window"] = window
    window.controller.set_loading(True)
    if not demo:
        # spoken turns use the faster model, exactly as the terminal voice frontend does
        voice_model = app.use_voice_model()
        window.controller.set_model(voice_model or active_model_text(app))
    window.show()
    signal.signal(signal.SIGINT, lambda *_: stop())

    watcher: ReminderWatcher | None = None
    try:
        build = asyncio.ensure_future(_make_session(app, voice, demo, window.controller.handle, console))
        waiter = asyncio.ensure_future(quit_event.wait())
        await asyncio.wait({build, waiter}, return_when=asyncio.FIRST_COMPLETED)
        if quit_event.is_set():
            build.cancel()
            return 0
        waiter.cancel()
        session = build.result()
        holder["session"] = session
        window.controller.session = session
        window.controller.set_loading(False)
        window.set_audio(bool(getattr(session, "has_audio", True)), app.settings.wake_word.title())
        if not demo:
            watcher = ReminderWatcher(app.store, session.say, session.is_idle)
            watcher.start()
        await session.run(greeting=None if demo else GREETING)
        return 0
    finally:
        if watcher is not None:
            await watcher.stop()
        window.close()
        QApplication.instance().processEvents()
