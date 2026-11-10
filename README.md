# Jarvis

Jarvis is a personal desktop agent for Windows 11. You talk to it in a terminal; a language model decides what to do and calls typed tools (files, shell, web, system, memory and so on) to do it. One agent loop handles every request, the whole conversation stays in context, and conversations and saved facts are kept in a local SQLite database so they survive restarts. The default brain is a hosted model behind an OpenAI-compatible API; a local Ollama model is available as a switchable option and as an automatic fallback.

Jarvis is a rebuild of an earlier project, Friday. See `docs/ARCHITECTURE.md` for the design and the reasons behind it.

## Requirements

- Windows 11
- Python 3.13 and [uv](https://docs.astral.sh/uv/)
- An API key for one supported provider (NVIDIA, Groq, OpenRouter, Gemini, or any OpenAI-compatible endpoint), or [Ollama](https://ollama.com) for a fully local setup

## Setup

```powershell
uv sync
copy .env.example .env
```

Edit `.env`:

- Set `NVIDIA_API_KEY`, or set `JARVIS_PROVIDER` to another provider and fill in its key (`GROQ_API_KEY`, `OPENROUTER_API_KEY`, `GEMINI_API_KEY`, or `CUSTOM_BASE_URL` + `CUSTOM_API_KEY`).
- For the local option, install Ollama and run `ollama pull qwen3:4b-instruct`.

`.env` is gitignored. Keys are only read from there and are never logged.

## Running

```powershell
uv run jarvis                       # interactive; resumes a conversation from the last 12 hours
uv run jarvis --new                 # start a fresh conversation
uv run jarvis --local               # use the local Ollama model
uv run jarvis --once "What time is it?"   # one turn, print the reply, exit
python -m jarvis                    # same as `jarvis`
```

| Flag | Meaning |
|---|---|
| `--local` | Use the local Ollama model as the main model |
| `--new` | Do not resume the latest conversation |
| `--once "text"` | Run a single turn non-interactively. Confirmations are denied unless `--yes` is given |
| `--yes` | Auto-approve confirmations (only honoured with `--once`) |
| `--provider NAME` | Override `JARVIS_PROVIDER` for this run |
| `--model ID` | Override `JARVIS_MODEL` for this run |
| `--verbose` | Show full tool arguments and a preview of each result |
| `--voice` | Voice mode (see below) |
| `--devices` | List audio input/output devices and exit |
| `--no-wake` | Voice mode: every utterance is addressed to Jarvis, no wake word needed |

If the configuration is invalid, Jarvis says which variable to set in `.env` and exits with code 2.

### Slash commands

| Command | Action |
|---|---|
| `/help` | List commands |
| `/new` | Start a new conversation |
| `/local`, `/cloud` | Switch to the local model, or back to the configured cloud model |
| `/model` | Show the active model |
| `/tools` | List loaded tools by group, with their risk |
| `/memory` | List saved facts |
| `/forget <id>` | Delete a saved fact |
| `/exit` | Quit (also `exit`, `quit`, Ctrl+D) |

Ctrl+C during a reply cancels it and returns to the prompt. Alt+Enter inserts a newline; pasted multi-line text is sent as one message.

Tools marked `confirm` (deleting or overwriting files, most shell commands, and similar) show what they are about to do and ask `Allow? [y/N]`. The default is No.

## Voice mode

Install the optional dependencies (microphone and speaker access, speech recognition, speech synthesis), then start it:

```powershell
uv sync --extra voice
uv run jarvis --voice          # add --no-wake to skip the wake word
uv run jarvis --devices        # find device names for JARVIS_INPUT_DEVICE / JARVIS_OUTPUT_DEVICE
```

If the extra is not installed, `--voice` prints this install command and exits with code 2.

The microphone stays open for the whole session. Speech is detected locally, transcribed locally, and the transcript goes to the same agent as typed text; replies are spoken sentence by sentence while the model is still writing the rest. The terminal shows the state (idle, listening, thinking, speaking), live mic and output level meters, what was heard (dimmed when it was not meant for Jarvis), the reply as it is spoken, and tool activity. Ctrl+C quits.

- **Wake word.** With `JARVIS_REQUIRE_WAKE_WORD=true` (the default) an utterance is only acted on if it contains the wake word, for example "Jarvis, what time is it?". Common mis-hearings of "jarvis" ("travis", "jervis") are accepted at the start of a sentence. Saying only "Jarvis" gets a "Yes?" and opens the follow-up window.
- **Follow-up window.** For `JARVIS_FOLLOWUP_SECONDS` after Jarvis finishes speaking (or after your last command) you can answer without the wake word.
- **Stopping.** "Stop", "cancel", "never mind", "be quiet" or "shut up" cancels the current reply silently.
- **Confirmations.** Risky actions are asked aloud ("Shall I go ahead and ...? Say yes or no."). Only a clear yes approves; anything else, silence for 15 seconds, or an unclear transcript means no.
- **Reminders.** Reminders set through Jarvis are spoken when they fall due (checked every 20 seconds), once Jarvis is idle. In the text interface they are printed.

### Barge-in

`JARVIS_BARGE_IN` decides what happens when you speak while Jarvis is thinking or talking:

| Mode | Behaviour |
|---|---|
| `vad` (default) | Speaking over Jarvis interrupts him immediately; what you say becomes the next request without the wake word. Mic input must be louder than the echo expected from the speakers. |
| `wake` | Only an utterance containing the wake word interrupts. |
| `off` | The microphone is ignored while Jarvis works or speaks, except for stop phrases. |

`vad` mode works best with headphones. On laptop speakers Jarvis's own voice leaks into the microphone; the echo gate copes with moderate leakage, but if Jarvis keeps cutting himself off or mistaking his voice for yours, use `JARVIS_BARGE_IN=wake`. A false trigger just returns Jarvis to idle; he does not resume the old reply.

### Voice settings

| Variable | Default | Meaning |
|---|---|---|
| `JARVIS_WAKE_WORD` | `jarvis` | The word that addresses Jarvis |
| `JARVIS_REQUIRE_WAKE_WORD` | `true` | `false` treats every utterance as addressed (same as `--no-wake`) |
| `JARVIS_FOLLOWUP_SECONDS` | `10` | Follow-up window length |
| `JARVIS_END_SILENCE_MS` | `650` | Silence that ends an utterance |
| `JARVIS_BARGE_IN` | `vad` | `vad`, `wake` or `off` |
| `JARVIS_STT_MODEL` | `auto` | Whisper model: `auto`, `tiny.en`, `base.en`, `small.en`, ... |
| `JARVIS_STT_DEVICE` | `auto` | `auto`, `cuda` or `cpu` |
| `JARVIS_TTS_VOICE` | `en_GB-alan-medium` | Piper voice |
| `JARVIS_TTS_SPEED` | `1.0` | Speech speed multiplier |
| `JARVIS_INPUT_DEVICE`, `JARVIS_OUTPUT_DEVICE` | empty | Device name or index; empty is the system default |

## Choosing the provider and model

Set `JARVIS_PROVIDER` (`nvidia`, `groq`, `openrouter`, `gemini`, `ollama` or `custom`) and optionally `JARVIS_MODEL` in `.env`. If `JARVIS_MODEL` is empty, the provider's default is used; the default model is set in `jarvis/llm/providers.py`. `JARVIS_FAST_MODEL` can name a smaller model on the same provider for background jobs such as summarising old context. A provider with no default model (and no `JARVIS_MODEL`) is reported as a configuration error.

## Local fallback

When the provider is a cloud one and `JARVIS_LOCAL_FALLBACK` is true (the default), Jarvis switches to the local Ollama model for the rest of a turn if the cloud provider cannot be reached before any text arrived, and tells you it did. A small local model is only offered the tool groups in `JARVIS_LOCAL_TOOL_GROUPS` (default `core,files,web,memory`). Set `JARVIS_LOCAL_FALLBACK=false` to disable this. At startup Jarvis warns if Ollama is not running or the model is not pulled.

## Where data is stored

By default in `%LOCALAPPDATA%\Jarvis` (override with `JARVIS_DATA_DIR`):

- `jarvis.db` - conversations, summaries and saved facts (SQLite)
- `history.txt` - prompt input history
- `logs/jarvis.log` - rotating log file; nothing is logged to the console

## Adding a tool

A tool is a typed function in `jarvis/tools/builtin/`. The JSON schema the model sees is generated from the signature and docstring. Every module in that folder is loaded at startup.

```python
# jarvis/tools/builtin/greeting.py
from jarvis.tools.base import Risk, ToolContext, ToolResult, tool


@tool(group="core")
async def word_count(text: str) -> ToolResult:
    """Count the words in a piece of text.

    Args:
        text: The text to count words in.
    """
    return ToolResult.success(f"{len(text.split())} words")


@tool(group="files", risk=Risk.CONFIRM)
async def create_note(title: str, ctx: ToolContext) -> ToolResult:
    """Create an empty note file in the data directory.

    Args:
        title: File name without extension.
    """
    path = ctx.settings.data_dir / f"{title}.txt"
    path.touch()
    return ToolResult.success(f"Created {path}")
```

`ctx` is injected and hidden from the model. Use `Risk.CONFIRM` for anything destructive or arbitrary. Return `ToolResult.fail(error, hint=...)` for expected failures instead of raising. See the docstring of `jarvis/tools/base.py` for the full rules.

## Project layout

```
jarvis/
  config.py        Settings (env-driven)
  llm/             Provider clients: OpenAI-compatible and native Ollama
  tools/           @tool decorator, registry, builtin/ tool modules
  memory/          SQLite store: conversations, summaries, facts
  agent/           The agent loop, prompt, context compaction, events
  interfaces/      app.py (wiring), cli.py (terminal frontend), voice.py (voice frontend)
  voice/           Audio I/O, VAD, speech-to-text, text-to-speech, the voice session
  proactive/       Reminder watcher (Jarvis speaking first)
docs/              Architecture notes
tests/             pytest suite (no network needed)
```

Run the tests with `uv run pytest`.

## Chrome control

Jarvis can browse in your own Chrome, signed in as you, through the Jarvis Browser Control extension. It can open pages, read them, list their links, buttons and fields, click, type, fill forms, scroll, take screenshots and attach files. It only ever touches tabs in a cyan tab group named **JARVIS**; your other tabs are never read or changed.

Set up once:

1. Open `chrome://extensions`, turn on **Developer mode**, click **Load unpacked** and choose the `extensions/jarvis-browser-control` folder.
2. Run `jarvis --browser-token`. It prints the pairing token and the folder path.
3. Click the extension's toolbar icon, paste the token, leave the port at `8765` (match `JARVIS_BROWSER_PORT` if you changed it) and click **Save and connect**.

Jarvis starts listening (on `127.0.0.1` only, for `chrome-extension://` origins, with the token required) the first time it uses a browser tool, and the extension reconnects on its own after that. Ask Jarvis "is Chrome connected?" to check; if it is not, it tells you what to do. If the port is taken, change `JARVIS_BROWSER_PORT` and the port in the popup.

Safety policy: reading, navigating, scrolling, clicking and typing run without asking. Jarvis asks you first before anything consequential: buying or paying, sending or posting, deleting, signing in or authorising, account or security changes, accepting terms, and uploading a file. It will not type into password, card number, CVV, one-time-code or similar fields at all; you enter those yourself. Only `http` and `https` pages are opened. Page text is treated as data, never as instructions.

While Jarvis clicks or types, Chrome shows a "Jarvis Browser Control started debugging this browser" banner; it disappears when the action finishes. That is how real mouse and keyboard events are sent.

## Documents, spreadsheets and the screen

Installed with `uv sync --all-extras` (or `--extra office --extra screen`).

- **Documents:** read PDF, Word, Markdown and HTML; create Markdown, HTML, PDF or Word documents; merge, split, extract, rotate and fill PDFs. New files only: nothing existing is overwritten.
- **Excel:** overview, read ranges, write cells and formulas, append rows, create workbooks, column totals and lookups. Writes to an existing workbook ask first and keep a `.bak`. openpyxl cannot calculate formulas, so a formula's value appears when the file is next opened in Excel.
- **Screen:** `read_screen_text` runs OCR locally (about 5 s for a 1080p screen). `look_at_screen` and `describe_image` send the image to a vision model. By default that is a hosted model on the configured key, falling back to the local Ollama vision model; set `JARVIS_VISION_PROVIDER=local` to keep images on this machine.

## Roadmap

- A HUD (it will consume the `VoiceEvent`s the voice session already emits)
- Keyboard and window control for desktop apps
- Email and calendar

## Credits

Parts of the capability code are ported from the MIT-licensed Friday project (copyright JARVIS Dev Team).
