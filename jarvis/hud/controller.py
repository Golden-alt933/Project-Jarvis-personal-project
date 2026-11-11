"""Turns session events into widget updates and UI actions into session calls.

No business logic: it only maps. `view` is the HUD window (or anything with the same
widget attributes); `session` is attached once it exists.
"""

from __future__ import annotations

import logging
from typing import Any

from jarvis.voice.events import (
    ConfirmRequested,
    ConfirmResolved,
    Heard,
    Info,
    Levels,
    MicMuted,
    Session,
    Spoke,
    StateChanged,
    ToolActivity,
    TurnFinished,
    UserText,
    VoiceEvent,
    VoiceState,
)

log = logging.getLogger(__name__)

STATE_WORDS = {
    VoiceState.IDLE: "STANDING BY",
    VoiceState.LISTENING: "LISTENING",
    VoiceState.THINKING: "THINKING",
    VoiceState.SPEAKING: "SPEAKING",
}


class HudController:
    def __init__(self, view: Any) -> None:
        self.view = view
        self.session: Session | None = None
        self.state = VoiceState.IDLE
        self.loading = False
        self.muted = False
        self.model = ""
        self.exchange_open = False
        self.spoke_in_exchange = False

    # -- events --------------------------------------------------------------------
    def handle(self, event: VoiceEvent) -> None:
        try:
            self._dispatch(event)
        except Exception:
            log.exception("HUD failed to render %s", type(event).__name__)
        if not isinstance(event, Levels):  # levels stream continuously; they never wake a resting HUD
            kick = getattr(self.view, "kick", None)
            if kick is not None:
                kick()

    def _dispatch(self, event: VoiceEvent) -> None:
        v = self.view
        if isinstance(event, StateChanged):
            self.state = event.state
            v.orb.set_state(event.state)
            v.input.set_busy(event.state is not VoiceState.IDLE)
            self._refresh_word()
        elif isinstance(event, Levels):
            v.orb.set_levels(event.mic, event.speaker)
        elif isinstance(event, Heard):
            if event.addressed:
                self._begin(event.text)
            else:
                v.reply.whisper(event.text)
        elif isinstance(event, UserText):
            self._begin(event.text)
        elif isinstance(event, Spoke):
            if not self.exchange_open:
                self._begin("")
            self.spoke_in_exchange = True
            v.reply.append_reply(event.text)
        elif isinstance(event, ToolActivity):
            self._tool(event)
        elif isinstance(event, Info):
            v.reply.whisper(event.text, 4.0)
        elif isinstance(event, ConfirmRequested):
            v.confirm.request(event.description)
            v.orb.set_confirming(True)
        elif isinstance(event, ConfirmResolved):
            v.confirm.resolve()
            v.orb.set_confirming(False)
        elif isinstance(event, MicMuted):
            self.muted = event.muted
            v.header.mic.set_muted(event.muted)
            v.orb.set_muted(event.muted)
            self._refresh_word()
        elif isinstance(event, TurnFinished):
            self._finished(event)

    def _begin(self, user_text: str) -> None:
        self.view.reply.begin_exchange(user_text)
        self.exchange_open = True
        self.spoke_in_exchange = False
        self.view.ticker.clear()

    def _tool(self, event: ToolActivity) -> None:
        ticker = self.view.ticker
        if event.done:
            ticker.finish(event.name, bool(event.ok), event.seconds)
        else:
            ticker.start(event.name, event.summary)
        self.view.orb.set_tool_running(ticker.running_count() > 0)

    def _finished(self, event: TurnFinished) -> None:
        v = self.view
        if event.text and not self.spoke_in_exchange and event.stopped not in ("cancelled", "error"):
            if not self.exchange_open:
                self._begin("")
            v.reply.append_reply(event.text)
        self.exchange_open = False
        if event.model:
            self.model = event.model
        parts = [self.model, f"{event.seconds:.1f} s"]
        if event.tool_calls:
            parts.append(f"{event.tool_calls} tool{'s' if event.tool_calls != 1 else ''}")
        v.footer.set_text(" · ".join(p for p in parts if p))
        v.orb.set_tool_running(False)
        if event.stopped in ("error", "max_steps", "time_budget"):
            v.orb.flash_error()

    # -- presentation state --------------------------------------------------------
    def set_loading(self, loading: bool) -> None:
        self.loading = loading
        self.view.orb.set_loading(loading)
        self._refresh_word()

    def set_model(self, text: str) -> None:
        self.model = text
        self.view.footer.set_text(text)

    def _refresh_word(self) -> None:
        if self.loading:
            word = "WAKING UP"
        elif self.state is VoiceState.IDLE and self.muted:
            word = "MUTED"
        else:
            word = STATE_WORDS[self.state]
        self.view.state_label.set_text(word)

    # -- UI actions ----------------------------------------------------------------
    def submit(self, text: str) -> None:
        if self.session is not None and text.strip():
            self.session.submit_text(text.strip())

    def cancel(self) -> None:
        if self.session is not None:
            self.session.cancel()

    def answer_confirm(self, approved: bool) -> None:
        if self.session is not None:
            self.session.answer_confirm(approved)

    def toggle_mute(self) -> None:
        if self.session is not None and getattr(self.session, "has_audio", True):
            self.session.set_muted(not self.session.muted)
