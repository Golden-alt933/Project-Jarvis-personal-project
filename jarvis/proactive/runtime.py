"""Wiring for the autonomy features: what each frontend starts next to its reminder watcher,
the headless daemon (``jarvis --daemon``) and the daily briefing (``jarvis --brief``).

A frontend only hands over how to *say* something and how to tell whether the user is
busy; this module owns everything else, so the three frontends differ by a few lines.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from jarvis.proactive.notify import notify
from jarvis.proactive.tasks import AgentTaskRunner, TaskScheduler, speakable
from jarvis.proactive.watchers import ProactiveRunner, SystemSensors, build_watchers, parse_quiet_hours

if TYPE_CHECKING:
    from jarvis.interfaces.app import App

log = logging.getLogger(__name__)

BRIEF_PROMPT = (
    "Give me a short spoken-style briefing for today, about a hundred words at most. Say the day "
    "and date. If my saved memory has a home location, check the weather there with your tools "
    "and give it in a sentence; if it does not, skip the weather without mentioning it. Then list "
    "today's reminders (list_reminders) and the scheduled tasks that run today "
    "(list_scheduled_tasks). If Chrome is connected you may use it for public pages, but never "
    "for anything that needs a sign-in. Do not ask me questions."
)
DAEMON_TITLE = "Jarvis"


@dataclass
class Background:
    """The scheduler and the proactive watchers of one frontend, started and stopped together."""

    tasks: TaskScheduler | None
    proactive: ProactiveRunner | None

    def start(self) -> None:
        if self.tasks is not None:
            self.tasks.start()
        if self.proactive is not None:
            self.proactive.start()

    async def stop(self) -> None:
        if self.proactive is not None:
            await self.proactive.stop()
        if self.tasks is not None:
            await self.tasks.stop()


def build_background(
    app: "App",
    say: Callable[[str], Awaitable[None]],
    is_idle: Callable[[], bool],
    *,
    spoken: bool = False,
    output_mode: str = "text",
    turn_active: Callable[[], bool] = lambda: False,
    say_busy: Callable[[str], Awaitable[None]] | None = None,
    confirm: Callable[[str], Awaitable[bool]] | None = None,
) -> Background:
    """The scheduler and watchers for a frontend that can ``say`` things.

    spoken: results are read aloud, so long ones are shortened (the full text is stored).
    confirm: how to ask the user to approve a risky step of an unattended task while they
    are present. None declines every such step, which is the safe default.
    say_busy: delivers the "still working" notice during a turn (None: not delivered).
    """
    settings = app.settings

    async def deliver(text: str) -> None:
        await say(speakable(text) if spoken else text)

    runner = AgentTaskRunner(app.agent, settings, confirm=confirm, output_mode=output_mode)
    tasks = TaskScheduler.from_settings(app.store, runner, deliver, is_idle, settings)
    watchers = build_watchers(settings, turn_active)
    proactive = ProactiveRunner(
        watchers, SystemSensors(), say, is_idle,
        deliver_busy=say_busy, quiet=parse_quiet_hours(settings.quiet_hours),
    )
    return Background(tasks, proactive)


def start_background(app: "App", say: Callable[[str], Awaitable[None]], is_idle: Callable[[], bool], **kwargs: Any) -> Background:
    """build_background(...) and start it. Never fatal: if it cannot be built, the frontend
    simply runs without scheduled tasks and alerts (the reason is logged)."""
    try:
        background = build_background(app, say, is_idle, **kwargs)
    except Exception:
        log.exception("scheduled tasks and proactive alerts are unavailable")
        return Background(None, None)
    background.start()
    return background


# -- the headless daemon ------------------------------------------------------------------------
def _stdout_line(text: str) -> None:
    try:
        print(text, flush=True)
    except (OSError, ValueError):
        pass  # no console (started at login): the log file and the notification remain


async def run_daemon(app: "App", console: Any = None, *, notifier: Callable[[str, str], bool] = notify) -> int:
    """Run scheduled tasks and proactive watchers with no window, until stopped.

    Confirmations are denied (nobody is present), results are stored by the scheduler and
    also logged, printed and shown as a desktop notification where the OS supports it.
    """

    async def say(text: str) -> None:
        log.info("daemon says: %s", text)
        _stdout_line(text)
        await asyncio.to_thread(notifier, DAEMON_TITLE, text)

    background = build_background(app, say, lambda: True, spoken=False, output_mode="text")
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    if sys.platform != "win32":
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                loop.add_signal_handler(sig, stop.set)
            except (NotImplementedError, RuntimeError):
                pass
    background.start()
    log.info("daemon started")
    _stdout_line("Jarvis daemon running: scheduled tasks and proactive alerts are active. Ctrl+C to stop.")
    try:
        await stop.wait()
    except (KeyboardInterrupt, asyncio.CancelledError):
        task = asyncio.current_task()
        if task is not None:
            task.uncancel()
    finally:
        await background.stop()
        log.info("daemon stopped")
    return 0


# -- the daily briefing --------------------------------------------------------------------------
async def run_brief(app: "App", console: Any = None, *, prompt: str = BRIEF_PROMPT) -> int:
    """One unattended turn that produces a short spoken-style briefing, printed to the terminal.

    It runs in its own conversation and declines every confirmation. Scheduled with
    "brief me every day at 9", the result is spoken by the voice or HUD frontend instead.
    """
    from jarvis.agent.events import TurnDone

    runner = AgentTaskRunner(app.agent, app.settings, confirm=None, output_mode="voice")
    agent = runner.new_agent()
    done: TurnDone | None = None
    async for event in agent.run_turn(prompt, output_mode="voice"):
        if isinstance(event, TurnDone):
            done = event
    text = (done.text if done else "").strip()
    if not text or (done is not None and done.stopped == "error"):
        _stdout_line(text or "I could not put a briefing together.")
        return 1
    if console is not None:
        console.print(text)
    else:
        _stdout_line(text)
    return 0
