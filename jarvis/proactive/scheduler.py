"""Delivers due reminders through whatever frontend is running.

The watcher knows nothing about voice or terminals: it is given a `deliver`
coroutine (speak it, print it) and an `is_idle` predicate so a reminder never
talks over the user or a running turn.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Any, Awaitable, Callable

from jarvis.tools.builtin.productivity import due_reminders, mark_reminder_fired

log = logging.getLogger(__name__)

POLL_SECONDS = 20.0
IDLE_RECHECK_SECONDS = 0.5


class ReminderWatcher:
    def __init__(
        self,
        store: Any,
        deliver: Callable[[str], Awaitable[None]],
        is_idle: Callable[[], bool] = lambda: True,
        *,
        interval: float = POLL_SECONDS,
        idle_recheck: float = IDLE_RECHECK_SECONDS,
    ) -> None:
        self.store = store
        self.deliver = deliver
        self.is_idle = is_idle
        self.interval = interval
        self.idle_recheck = idle_recheck
        self._task: asyncio.Task[None] | None = None

    async def check(self) -> int:
        """Deliver everything due right now. Returns how many were delivered."""
        delivered = 0
        for row in due_reminders(self.store, datetime.now()):
            while not self.is_idle():
                await asyncio.sleep(self.idle_recheck)
            await self.deliver(f"Reminder: {row['message']}")
            mark_reminder_fired(self.store, row["id"])  # only after delivery: a crash repeats, never loses
            delivered += 1
        return delivered

    async def run(self) -> None:
        while True:
            try:
                await self.check()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("reminder check failed")
            await asyncio.sleep(self.interval)

    def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.ensure_future(self.run())

    async def stop(self) -> None:
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
