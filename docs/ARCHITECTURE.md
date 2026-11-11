# Jarvis architecture

Jarvis is a rebuild of the Friday assistant. Friday's capability code was mostly
sound; its brain was not. This document is the contract every module is built
against.

## Why Friday fell short (and the rule each failure produced)

| Friday | Rule in Jarvis |
|---|---|
| Six regex layers chose the workflow, tool subset and prompt before the model ran | **The model decides.** No code inspects the user's sentence to route, pick tools, or rewrite the answer. |
| The agent loop received no conversation history | **Full conversation in context**, persisted on every message, compacted by summarising when long. |
| A templated ledger could overwrite the model's reply | **The model's reply is the reply.** |
| Two brains (agent loop + 29-intent classifier fallback) | **One loop.** |
| Every tool call wrapped in Task -> Router -> Executor -> Skill, returning prose | **Tools are typed functions** returning `ToolResult`; schemas are generated from signatures. |
| Local tool turns capped at 256 output tokens; 4B model as the main brain | **A strong hosted model is the default brain**; local Ollama is a switchable option and automatic fallback. |
| Cloud calls gated behind a local classifier and local privacy review per round | **One call per step**, straight to the provider. |
| Facts dumped whole into a fallback prompt; history in RAM only | **SQLite memory**: durable conversations, facts recalled by relevance. |
| `run_command` unconfirmed, every browser click confirmed | **Risk is per call**: confirm what is destructive or arbitrary, nothing else. |

## Layout

```
jarvis/
  config.py            Settings and config-file lookup (see "Configuration lookup"). Data lives in the
                       per-user data folder (%LOCALAPPDATA%\Jarvis, ~/.local/share/jarvis).
  platform/            OS backends behind one contract (see "Platform layer")
  doctor.py            `jarvis --doctor`: checklist with a fix per problem
  setup_wizard.py      `jarvis --setup`: provider, key, name -> per-user config file
  llm/
    types.py           Message, ToolCall, LLMResponse, stream events, errors, LLMClient protocol
    openai_compat.py   One client for every OpenAI-compatible endpoint (NVIDIA, Groq, OpenRouter, Gemini)
    ollama.py          Native Ollama client (needs num_ctx, which the /v1 shim ignores)
    providers.py       Provider profiles and build_clients(settings) -> (primary, fallback | None)
  tools/
    base.py            @tool decorator, ToolSpec, ToolResult, ToolContext, Risk
    registry.py        ToolRegistry: discovery, schemas, guarded execute()
    builtin/           One module per capability area; each function decorated with @tool
  memory/
    store.py           MemoryStore (SQLite, WAL): conversations, messages, summaries, facts (FTS5)
  agent/
    events.py          TextChunk, ToolStarted, ToolFinished, Notice, TurnDone
    prompt.py          build_system_prompt(...)
    context.py         History assembly and compaction
    loop.py            Agent: the only control flow
  interfaces/
    app.py             build_app(settings) wires store + registry + clients + agent
    cli.py             Streaming terminal chat
    voice.py           Voice frontend (status display over VoiceEvents)
  voice/               types.py (contracts), audio.py, vad.py, stt.py, tts.py, session.py
  vision/              capture.py, ocr.py (local RapidOCR), client.py (hosted or local vision model)
  browser/             bridge.py (loopback WebSocket server; extension_dir() finds the extension in a
                       checkout or inside the installed wheel), policy.py (which page actions ask first)
  proactive/           scheduler.py (reminders that fire without being asked), tasks.py + taskstore.py +
                       schedule_math.py (scheduled tasks), watchers.py (battery/disk/long-turn alerts),
                       runtime.py (frontend wiring, --daemon, --brief), notify.py (desktop toast)
extensions/
  jarvis-browser-control/   Chrome extension paired with the bridge by a token (copied into the
                            wheel as jarvis/browser/extension by hatch force-include)
install.sh, install.ps1     One-command installers (uv tool install at a pinned ref)
.github/workflows/ci.yml    Tests on Ubuntu and Windows; installer job on both
```

## Platform layer (`jarvis/platform/`)

Everything an operating system does differently sits behind the `Platform` protocol in
`base.py`: running a shell command (and killing its process group or tree), opening a path
or URL, known folders, the app catalogue and launching, visible windows and closing one,
the foreground window title, volume/mute, media keys, lock, clipboard availability, protected
system paths, screen-grab and cursor helpers, Excel COM detection, GPU library loading and the
helper-program report the doctor prints. `jarvis.platform.current()` picks the backend from
`sys.platform` once: `windows.py`, `linux.py`, or the base class (macOS and others), whose
operations raise `PlatformUnsupported` (an `OSError`) with a message that says what is missing.
Tool modules call the layer and contain no `sys.platform` checks; they turn `OSError` into a
`ToolResult.fail`, so a missing helper reads as "install wmctrl" rather than a traceback.
Tests install a fake with `jarvis.platform.set_backend(...)`.

The Linux backend shells out to ordinary helpers: `xdg-open`/`gio`, `.desktop` files for apps
(`gtk-launch` or the cleaned Exec line), `wmctrl` for windows (X11 only; Wayland reports
"not supported"), `wpctl`/`pactl`/`amixer` for volume, `playerctl`, `loginctl`/`xdg-screensaver`,
`xdotool`, `grim`/`gnome-screenshot` for Wayland capture. `run_command` uses `bash -lc` with a
conservative allowlist classifier (`classify_bash`) next to the PowerShell one; the tool
description and system prompt name the shell and OS actually in use. Excel recalculation is
Excel COM on Windows or a headless LibreOffice convert of a temp copy elsewhere (`excel_calc.py`).

## Senses and hands (camera, desktop control)

<!-- senses:start -->
Two tool groups add the physical side. **Camera** (`vision/camera.py`, `tools/builtin/camera.py`):
OpenCV opens the device, discards the first frames while auto-exposure settles, keeps the
sharpest of a short burst, releases the device in a `finally`, and hands a downscaled PIL image
to the same `VisionClient` the screen tools use. Consent is a setting (`JARVIS_CAMERA`: `ask`
by default, once per process, a refusal is not remembered). **Desktop control**
(`tools/builtin/desktop.py`, policy in `desktop/policy.py`, OCR targeting in
`desktop/ocr_target.py`) works accessibility-tree first, OCR text second, and never offers raw
coordinates to the model. The OS-specific parts are new `Platform` methods: window info, focus,
element listing and acting (`windows_uia.py`: UI Automation through `comtypes`, one COM thread,
one cached subtree call per window, refs keyed by the element's RuntimeId so they stay stable),
and input synthesis (`windows_input.py`: `SendInput`, Unicode typing; Linux: `xdotool` on X11,
no element listing yet). The safety rules live in one pure module that reuses
`browser/policy.py`: consequential clicks ask, credential fields and elevated, own and
security-desktop windows are refused, terminals and the Run dialog ask with the text shown,
irreversible key chords ask. Every action reads back what changed. Camera and desktop are not in
the small local model's default tool groups.
<!-- senses:end -->

## Configuration lookup

First match wins: environment variables, then `<repo>/.env` (only when running from a
checkout, detected by `pyproject.toml` beside the package), then the per-user file
(`%APPDATA%\Jarvis\.env`, `${XDG_CONFIG_HOME:-~/.config}/jarvis/.env`, via platformdirs;
`JARVIS_CONFIG_FILE` overrides it). `jarvis --config-path` prints the file in use and
`jarvis --setup` writes the per-user file with owner-only permissions.

Data that leaves the machine: every model call sends the conversation and tool results
to the configured provider. `look_at_screen` and `describe_image` additionally send the
image unless `JARVIS_VISION_PROVIDER=local`. OCR is always local. Speech is local by
default only in local mode (`JARVIS_PROVIDER=ollama`); otherwise see the speech paragraph
below.

Speech (`voice/`): `build_synthesizer` returns Piper alone or a `FallbackSynthesizer`
(Edge neural voice, then NVIDIA Magpie when a key is set, then Piper) that falls back per
sentence on an error or when no audio arrives within 2.5 s, and stays on Piper for 45 s.
Engines may offer `synthesize_stream`; the session's synthesis worker plays chunks as they
arrive. Stock phrases are cached on disk per voice. `build_transcriber` returns the local
faster-whisper alone, or a `GatedTranscriber`: a CPU tiny.en model hears every utterance,
and only audio addressed to Jarvis (wake word, follow-up window, barge-in, confirmation
answer, `--no-wake`) is sent to the online recogniser (NVIDIA Parakeet, or any
`/audio/transcriptions` endpoint); on failure or after 4 s the local transcript is used.

Frontends only consume `AgentEvent`s and supply `confirm`/`notify` callbacks, so the
agent needs no change for a new one (a HUD is the next).

## Contracts

### LLM (`jarvis/llm/types.py`)
`LLMClient.stream(messages, tools, max_tokens=, temperature=)` yields `TextDelta`
then exactly one `ResponseDone(LLMResponse)`. Errors: `LLMUnavailable` (network,
timeout, 5xx), `LLMRateLimited` (429), `LLMBadRequest` (other 4xx). Attributes:
`name`, `model`, `context_window`, `is_local`.

`providers.build_clients(settings) -> tuple[LLMClient, LLMClient | None]` returns the
primary client and the local fallback (None if the primary is already local, or
`local_fallback` is false). `providers.build_fast_client(settings)` returns a client
for cheap side jobs (the fast model if configured, else the primary).

### Tools (`jarvis/tools/base.py`, `registry.py`)
See the module docstring in `base.py`. Groups in use: `core`, `files`, `shell`,
`web`, `memory`, `system`, `office`, `screen`, `browser`. A small local model is
offered only `settings.local_tool_groups`.

### Memory (`jarvis/memory/store.py`)
```python
@dataclass
class Fact: id: int; text: str; category: str; created_at: str; updated_at: str

class MemoryStore:
    def __init__(self, db_path: Path) -> None            # creates schema, WAL, synchronous=FULL
    def close(self) -> None
    # generic access so tool modules can own their tables (notes, reminders, ...)
    def ensure_schema(self, ddl: str) -> None
    def execute(self, sql: str, params: Sequence = ()) -> list[sqlite3.Row]   # commits
    # conversations
    def start_conversation(self) -> str
    def latest_conversation(self, max_age_hours: float | None = None) -> str | None
    def append_message(self, conversation_id: str, message: Message) -> int   # row id
    def load_messages(self, conversation_id: str, after_id: int = 0) -> list[tuple[int, Message]]
    def get_summary(self, conversation_id: str) -> tuple[str, int] | None     # (summary, covers up to message id)
    def set_summary(self, conversation_id: str, summary: str, upto_message_id: int) -> None
    def search_messages(self, query: str, limit: int = 8) -> list[dict]       # past conversations
    # facts
    def remember(self, text: str, category: str = "general") -> Fact          # dedupes near-identical text
    def recall(self, query: str, limit: int = 8) -> list[Fact]                # FTS5 bm25, falls back to recent
    def list_facts(self, category: str | None = None, limit: int = 50) -> list[Fact]
    def forget(self, fact_id: int) -> bool
```
All methods are synchronous, thread-safe (one connection, `check_same_thread=False`,
an `RLock`), and durable the moment they return.

### Agent (`jarvis/agent/loop.py`)
```python
class Agent:
    def __init__(self, primary: LLMClient, registry: ToolRegistry, store: MemoryStore,
                 settings: Settings, fallback: LLMClient | None = None,
                 fast: LLMClient | None = None) -> None
    conversation_id: str
    def new_conversation(self) -> str
    def resume_latest(self, max_age_hours: float = 12) -> bool
    def use_local(self, on: bool) -> None            # force the local model
    def cancel(self) -> None                          # stop the running turn
    def run_turn(self, user_text: str, *, confirm=..., notify=..., output_mode: str = "text"
                 ) -> AsyncIterator[AgentEvent]
```

One turn:
1. Persist the user message.
2. Build messages: system prompt (persona, current date/time, machine facts, facts
   recalled for this message, conversation summary) + history after the summary cut.
   Compact first if the estimate exceeds `history_token_budget`.
3. Stream the model. Emit `TextChunk`s. Persist the assistant message.
4. No tool calls: emit `TurnDone`.
5. Otherwise execute the calls through `registry.execute` (concurrently when every
   call is `Risk.SAFE`, otherwise in order), emit `ToolStarted`/`ToolFinished`,
   persist each tool message, go to 3.
6. Stop conditions: `max_steps` or `turn_time_budget` (then one final call with no
   tools so the model reports what was and was not done), cancellation, or error.

Failure handling:
- `LLMUnavailable` from the primary before any text was emitted in that step: emit a
  `Notice`, switch to the fallback for the rest of the turn.
- `LLMBadRequest`: compact history harder and retry the step once.
- The same call (name + arguments) a third time in one turn is not executed; the
  model gets a failed result telling it the result will not change.
- A tool failure or timeout never ends the turn; it is a result the model reads.
- History is never cut between an assistant message with tool calls and its results.

Memory is written by the model through the `remember` tool, not by an extra
extraction call per turn.

## Safety
- `Risk.CONFIRM`: deleting/moving/overwriting files, any shell command that is not on
  a read-only allowlist, closing applications, sending anything on the user's behalf,
  purchases. The frontend's `confirm` callback asks; a failed or missing callback
  means "no".
- File deletion goes to the Recycle Bin or Trash (`send2trash`), never a hard delete.
- Web and file content returned by tools is data. The system prompt states that
  instructions inside tool results are not the user's instructions.
- Secrets live only in environment variables and the `.env` config files (gitignored / per-user, mode 0600 where supported) and are never logged or printed.

## Ported from Friday (MIT, copyright JARVIS Dev Team)
Logic is ported, not the `Skill.execute(task)` wrappers: file search scoring,
exact-once file edit with backup, the app shortcut resolver, reminders, documents/PDF,
Excel, screen capture + OCR, and the Chrome extension bridge.

## Autonomy: scheduled tasks, proactive alerts, errands

Everything here is tools, scheduling and prompt guidance; nothing inspects the user's
sentence. The model turns "every morning at nine, brief me" into a `schedule_task` call.

- **Scheduled tasks** (`proactive/tasks.py`). A task is a plain-language instruction stored in
  SQLite with a repeat rule and the first run's wall-clock time. `TaskScheduler` polls, claims a
  due slot with a compare-and-set UPDATE (so the daemon and a window cannot both run it, and a
  crash never repeats it), advances the schedule *before* running (at most one catch-up run), and
  runs the task as an unattended turn of a sibling `Agent` in its own conversation with a smaller
  step and time budget. It waits for the frontend to be idle, runs one task at a time, and skips a
  task deferred for ten minutes. Calendar arithmetic (`schedule_math.py`) works in zone-aware local
  time: DST keeps "9 am" at 9 am, a nonexistent time runs just after the gap, a repeated hour runs
  once, and a next run further away than its rule allows (clock set back) is recomputed.
- **Unattended safety.** The run's `confirm` callback declines everything unless a frontend supplied
  a way to ask the present user; the decline is recorded in the result. No frontend supplies one yet.
- **Proactive watchers** (`proactive/watchers.py`). Pure state machines (episode, hysteresis,
  cool-down, quiet hours) over injected sensors and clock; no model call. Delivered through the same
  `say` path as reminders, only when the user is idle.
- **Errands** (`tools/builtin/errands.py`, group `errands`). Pure URL and .ics builders plus thin
  tools that open the result in the JARVIS Chrome tab group when the extension is connected, else in
  the default browser or app. They prefill and never send or save. Multi-step website work uses the
  browser tools, steered by the "working on websites" prompt section, which is added only when the
  browser tool group is offered to the model (`build_system_prompt(tool_groups=...)`).
