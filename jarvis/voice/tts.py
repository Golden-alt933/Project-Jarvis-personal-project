"""Text-to-speech: Piper synthesis, text cleanup for speech, and sentence chunking.

The agent streams text deltas; ``SentenceChunker`` turns them into speakable chunks
as early as possible so the first sentence is already playing while the rest of the
answer is still being generated.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from jarvis.config import Settings

log = logging.getLogger(__name__)

DEFAULT_VOICE = "en_GB-alan-medium"
FALLBACK_VOICE = "en_US-lessac-medium"
# Friday (the previous assistant) shipped this voice; used only if the download fails.
_FRIDAY_VOICE_DIR = Path(r"C:\Users\HP\Desktop\friday-agent-main\speech\text_to_speech\models")


# --------------------------------------------------------------------------- cleanup

_FENCE_RE = re.compile(r"```.*?(?:```|\Z)", re.DOTALL)
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_URL_RE = re.compile(r"(?:https?://|www\.)([^/\s?#]+)[^\s]*", re.IGNORECASE)
_URL_TRAIL = ".,;:!?)\"'"
_TABLE_SEP_RE = re.compile(r"^\s*\|?[\s:|-]*-[\s:|-]*\|?\s*$")
_HR_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
_BULLET_RE = re.compile(r"^\s*(?:[-*+\u2022]\s+)")
_EMOJI_RE = re.compile("[\U0001F000-\U0001FAFF\u2600-\u27BF\uFE0F\u200d]")
_BOLD_RE = re.compile(r"(\*\*|__)(?=\S)(.+?)(?<=\S)\1")
_ITALIC_STAR_RE = re.compile(r"(?<![\w*])\*(?=[^\s*])(.+?)(?<=[^\s*])\*(?![\w*])")
_ITALIC_UNDER_RE = re.compile(r"(?<![\w_])_(?=[^\s_])(.+?)(?<=[^\s_])_(?![\w_])")
_STRIKE_RE = re.compile(r"~~(.+?)~~")

_SYMBOLS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\u00b0\s*C\b"), " degrees Celsius"),
    (re.compile(r"\u00b0\s*F\b"), " degrees Fahrenheit"),
    (re.compile(r"\u00b0"), " degrees"),
    (re.compile(r"\s*%"), " percent"),
    (re.compile(r"\s*&\s*"), " and "),
    (re.compile(r"\s*(?:\u2192|->|=>|\u21d2)\s*"), " to "),
    (re.compile(r"\s*(?:\u2190|<-)\s*"), " from "),
    (re.compile(r"\s*\u2248\s*"), " about "),
    (re.compile(r"\s*\u2265\s*"), " at least "),
    (re.compile(r"\s*\u2264\s*"), " at most "),
    (re.compile(r"\s*\u00d7\s*"), " times "),
    (re.compile(r"\s*\u2260\s*"), " not equal to "),
]


def _domain(match: re.Match[str]) -> str:
    domain = match.group(1).lower().removeprefix("www.").rstrip(_URL_TRAIL)
    end = match.group(0)[-1]
    return domain + (end if end in ".,;:!?" else "")


def _clean_line(line: str) -> str:
    if _HR_RE.match(line) or _TABLE_SEP_RE.match(line) and "|" in line:
        return ""
    line = re.sub(r"^\s*#{1,6}\s+", "", line)
    line = re.sub(r"^\s*>+\s?", "", line)
    line = _BULLET_RE.sub("", line)
    if "|" in line:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        line = ", ".join(c for c in cells if c)
    return line.strip()


def clean_for_speech(text: str) -> str:
    """Make model output speakable: no Markdown, code, URLs or symbols TTS mangles."""
    text = _FENCE_RE.sub("\n(code omitted)\n", text)
    text = _IMAGE_RE.sub(r"\1", text)
    text = _LINK_RE.sub(r"\1", text)
    text = _URL_RE.sub(_domain, text)
    text = re.sub(r"`+([^`\n]*)`+", r"\1", text)
    text = text.replace("`", "")

    lines = [_clean_line(line) for line in text.splitlines()]
    lines = [line for line in lines if line]
    if len(lines) > 1:  # keep list items and headings from running together
        lines = [line if line[-1] in ".!?:;," else line + "." for line in lines]
    text = " ".join(lines)

    text = _BOLD_RE.sub(r"\2", text)
    text = _STRIKE_RE.sub(r"\1", text)
    text = _ITALIC_STAR_RE.sub(r"\1", text)
    text = _ITALIC_UNDER_RE.sub(r"\1", text)
    text = text.replace("*", "")
    text = _EMOJI_RE.sub("", text)
    for pattern, replacement in _SYMBOLS:
        text = pattern.sub(replacement, text)
    return re.sub(r"\s+", " ", text).strip()


# ------------------------------------------------------------------------- chunking

_ABBREVIATIONS = frozenset(
    "dr mr mrs ms prof sr jr st vs v e.g i.e u.s u.k a.m p.m inc ltd co corp approx "
    "fig vol mt gen sgt capt lt col rev hon dept est al cf ca".split()
)
_BOUNDARY_RE = re.compile(
    r"""(?P<sent>[.!?\u2026]+["')\]]*\s)
       |(?P<nl>\n)
       |(?P<clause>[,;:]["')\]]*\s|\s[\u2014\u2013-]\s)""",
    re.VERBOSE,
)
_CLAUSE_RE = re.compile(r"""[,;:]["')\]]*\s|\s[\u2014\u2013-]\s""")


class SentenceChunker:
    """Turns a stream of text deltas into speakable chunks.

    Decisions depend only on the text seen so far (a boundary needs the whitespace
    after it), so feeding character by character yields the same chunks as feeding
    everything at once.
    """

    def __init__(self, min_chars: int = 25, max_chars: int = 240, first_clause_chars: int = 60):
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.first_clause_chars = first_clause_chars
        self._buf = ""
        self._first = True

    def feed(self, text_delta: str) -> list[str]:
        self._buf += text_delta
        return self._drain(final=False)

    def flush(self) -> list[str]:
        out = self._drain(final=True)
        self._buf = ""
        return out

    # -- internals

    def _drain(self, final: bool) -> list[str]:
        out: list[str] = []
        while self._buf:
            fence = self._buf.find("```")
            if fence == 0:
                close = self._buf.find("```", 3)
                if close < 0:
                    if final:
                        self._emit(out, self._buf)  # unterminated block: marker only
                        self._buf = ""
                    break
                self._emit(out, self._buf[: close + 3])
                self._buf = self._buf[close + 3 :]
                continue
            if fence > 0:
                # A code block starts here: everything before it is complete.
                head, self._buf = self._buf[:fence], self._buf[fence:]
                self._split_all(out, head)
                continue
            split = self._find_split(self._buf)
            if split is None:
                break
            self._emit(out, self._buf[:split])
            self._buf = self._buf[split:]
        if final and self._buf:
            self._split_all(out, self._buf)
            self._buf = ""
        return out

    def _split_all(self, out: list[str], text: str) -> None:
        while text:
            split = self._find_split(text)
            if split is None:
                self._emit(out, text)
                return
            self._emit(out, text[:split])
            text = text[split:]

    def _emit(self, out: list[str], raw: str) -> None:
        cleaned = clean_for_speech(raw)
        if cleaned:
            out.append(cleaned)
            self._first = False

    def _find_split(self, buf: str) -> int | None:
        window = buf[: self.max_chars + 1]
        for match in _BOUNDARY_RE.finditer(window):
            end = match.end()
            length = len(window[:end].strip())
            kind = match.lastgroup
            if kind == "sent":
                if length >= self.min_chars and not self._is_non_terminal(window, match):
                    return end
            elif kind == "nl":
                if length >= self.min_chars:
                    return end
            elif self._first and length >= self.first_clause_chars:
                return end
        if len(buf) > self.max_chars:
            return self._force_split(window)
        return None

    def _force_split(self, window: str) -> int:
        clause_ends = [m.end() for m in _CLAUSE_RE.finditer(window) if m.end() >= self.min_chars]
        if clause_ends:
            return clause_ends[-1]
        space = window.rfind(" ", self.min_chars)
        return space + 1 if space > 0 else self.max_chars

    @staticmethod
    def _is_non_terminal(window: str, match: re.Match[str]) -> bool:
        """True for periods that do not end a sentence (abbreviations, ellipses, initials, list numbers)."""
        marks = match.group(0).rstrip(" \n\t\r").rstrip("\"')]")
        if not marks.endswith("."):
            return False
        if len(marks) > 1 and marks[-2] == ".":
            return True  # ellipsis
        start = match.start()
        token_start = max(window.rfind(" ", 0, start), window.rfind("\n", 0, start)) + 1
        token = window[token_start:start]
        bare = token.lstrip("(\"'[").lower()
        if bare in _ABBREVIATIONS:
            return True
        if re.fullmatch(r"(?:[a-z]\.)+[a-z]", bare):
            return True  # e.g. "U.S", "i.e"
        if len(token) == 1 and token.isupper() and token not in "IA":
            return True  # initial, as in "J. Smith"
        if token.isdigit() and (token_start == 0 or window[token_start - 1] == "\n"):
            return True  # numbered list marker
        return False


# --------------------------------------------------------------------------- Piper


class PiperSynthesizer:
    """Synthesizer backed by a local Piper voice."""

    def __init__(self, voice: str = DEFAULT_VOICE, voices_dir: Path | None = None, speed: float = 1.0):
        self.voice = voice
        self.voices_dir = Path(voices_dir) if voices_dir else Path.cwd()
        self.speed = speed if speed > 0 else 1.0
        self.name = f"piper {voice}"
        self.sample_rate = 22050
        self._engine = None
        self._config = None
        self._lock = threading.Lock()  # one onnx session; serialise concurrent callers
        self._read_sample_rate()

    @property
    def model_path(self) -> Path:
        return self.voices_dir / f"{self.voice}.onnx"

    def _read_sample_rate(self) -> None:
        config = Path(f"{self.model_path}.json")
        try:
            self.sample_rate = int(json.loads(config.read_text(encoding="utf-8"))["audio"]["sample_rate"])
        except (OSError, KeyError, ValueError):
            pass

    def ensure_voice(self) -> Path:
        """Make sure the voice files exist locally; download or fall back. Blocking."""
        model = self.model_path
        config = Path(f"{model}.json")
        if model.exists() and config.exists():
            return model
        self.voices_dir.mkdir(parents=True, exist_ok=True)
        try:
            from piper.download_voices import download_voice

            log.info("Downloading Piper voice %s to %s", self.voice, self.voices_dir)
            download_voice(self.voice, self.voices_dir)
        except Exception as exc:  # network, DNS, bad voice name...
            log.warning("Voice download failed (%s); trying local fallback voice", exc)
            for leftover in (model, config):  # a half-written file must not look complete
                leftover.unlink(missing_ok=True)
            self._copy_fallback()
        if not (model.exists() and config.exists()):
            raise RuntimeError(f"Piper voice {self.voice} is not available")
        self._read_sample_rate()
        return model

    def _copy_fallback(self) -> None:
        src = _FRIDAY_VOICE_DIR / f"{FALLBACK_VOICE}.onnx"
        if not src.exists():
            log.error("No fallback voice found at %s", src)
            return
        shutil.copyfile(src, self.model_path)
        shutil.copyfile(f"{src}.json", f"{self.model_path}.json")
        log.warning("Using Friday's %s as %s", FALLBACK_VOICE, self.voice)

    def _load(self):
        with self._lock:
            if self._engine is None:
                from piper import PiperVoice, SynthesisConfig

                model = self.ensure_voice()
                self._engine = PiperVoice.load(model)
                base = self._engine.config.length_scale or 1.0
                self._config = SynthesisConfig(length_scale=base / self.speed)
                self.sample_rate = self._engine.config.sample_rate
            return self._engine

    def _synthesize_blocking(self, text: str) -> np.ndarray:
        engine = self._load()
        with self._lock:
            chunks = [c.audio_float_array for c in engine.synthesize(text, self._config)]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks).astype(np.float32, copy=False)

    async def warm_up(self) -> None:
        await asyncio.to_thread(self._synthesize_blocking, "Ready.")

    async def synthesize(self, text: str) -> np.ndarray:
        text = clean_for_speech(text)
        if not text:
            return np.zeros(0, dtype=np.float32)
        return await asyncio.to_thread(self._synthesize_blocking, text)


def build_synthesizer(settings: Settings) -> PiperSynthesizer:
    return PiperSynthesizer(settings.tts_voice, settings.voices_dir, settings.tts_speed)
