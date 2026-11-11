"""Text-to-speech: Piper synthesis, text cleanup for speech, and sentence chunking.

The agent streams text deltas; ``SentenceChunker`` turns them into speakable chunks
as early as possible so the first sentence is already playing while the rest of the
answer is still being generated.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import shutil
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from jarvis.config import Settings

log = logging.getLogger(__name__)

DEFAULT_VOICE = "en_GB-alan-medium"
# Friday (the previous assistant) shipped this voice; used only if the download fails.


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
    text = _FENCE_RE.sub("\n", text)  # code is shown on screen (CodeShown), never read out
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
            log.warning("Voice download failed (%s); looking for an already installed voice", exc)
            for leftover in (model, config):  # a half-written file must not look complete
                leftover.unlink(missing_ok=True)
            self._copy_fallback()
        if not (model.exists() and config.exists()):
            raise RuntimeError(f"Piper voice {self.voice} is not available")
        self._read_sample_rate()
        return model

    def _copy_fallback(self) -> None:
        """Offline and the wanted voice is missing: reuse any voice already on disk, so
        Jarvis can still speak (in the wrong voice) rather than not at all."""
        for candidate in sorted(self.voices_dir.glob("*.onnx")):
            config = Path(f"{candidate}.json")
            if candidate != self.model_path and config.exists():
                shutil.copyfile(candidate, self.model_path)
                shutil.copyfile(config, f"{self.model_path}.json")
                log.warning("Using the installed voice %s in place of %s", candidate.stem, self.voice)
                return
        log.error("No voice is installed in %s and the download failed", self.voices_dir)

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


    @property
    def cache_key(self) -> str:
        return f"piper:{self.voice}:{self.speed:g}"


# --------------------------------------------------------------- online + fallback

OUTPUT_RATE = 24000  # FallbackSynthesizer always speaks at this rate, whichever engine answered


def resample_audio(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    """Resample float32 mono with PyAV (declared dependency of the voice extra)."""
    if src == dst or len(audio) == 0:
        return audio
    import av

    resampler = av.AudioResampler(format="flt", layout="mono", rate=dst)
    frame = av.AudioFrame.from_ndarray(np.asarray(audio, dtype=np.float32).reshape(1, -1), format="flt", layout="mono")
    frame.sample_rate = src
    parts = [f.to_ndarray().reshape(-1) for f in [*resampler.resample(frame), *resampler.resample(None)]]
    return np.concatenate(parts).astype(np.float32, copy=False) if parts else np.zeros(0, dtype=np.float32)


class FallbackSynthesizer:
    """Tries engines in order, per sentence; the last one (Piper) is the offline floor.

    An engine that raises, or has not produced its first audio within ``deadline`` seconds,
    is skipped for ``cooloff`` seconds (one warning is logged), and the sentence is spoken
    by the next engine. If an engine fails after it already produced audio the rest of
    that sentence is dropped rather than repeated from the start by another voice.
    Engines may offer ``synthesize_stream`` (an async iterator of arrays); this class
    offers it too, so playback can begin before a sentence is complete.
    """

    def __init__(
        self,
        engines: list,
        *,
        deadline: float = 2.5,
        chunk_deadline: float = 6.0,
        cooloff: float = 45.0,
        clock=time.monotonic,
        sample_rate: int = OUTPUT_RATE,
    ) -> None:
        if not engines:
            raise ValueError("FallbackSynthesizer needs at least one engine")
        self.engines = list(engines)
        self.deadline = deadline
        self.chunk_deadline = chunk_deadline
        self.cooloff = cooloff
        self.clock = clock
        self.sample_rate = sample_rate
        self._down_until = [0.0] * len(self.engines)
        self._was_down = [False] * len(self.engines)
        self._last = 0

    # -- state
    def _cooling(self, index: int) -> bool:
        return index < len(self.engines) - 1 and self.clock() < self._down_until[index]

    def _active(self) -> int:
        return next((i for i in range(len(self.engines)) if not self._cooling(i)), len(self.engines) - 1)

    @property
    def name(self) -> str:
        return self.engines[self._active()].name

    @property
    def active_engine(self):
        return self.engines[self._active()]

    def stock_key(self) -> str:
        """Identity of the voice that would speak right now (stock phrases are cached per voice)."""
        return getattr(self.engines[self._active()], "cache_key", self.engines[self._active()].name)

    @property
    def last_key(self) -> str:
        engine = self.engines[self._last]
        return getattr(engine, "cache_key", engine.name)

    def _mark_down(self, index: int, reason: object) -> None:
        self._down_until[index] = self.clock() + self.cooloff
        if not self._was_down[index]:
            log.warning(
                "Speech engine %s is unavailable (%s); using %s for the next %.0f s",
                self.engines[index].name, reason or "timeout", self.engines[-1].name, self.cooloff,
            )
        self._was_down[index] = True

    def _mark_ok(self, index: int) -> None:
        if self._was_down[index]:
            log.info("Speech engine %s is back", self.engines[index].name)
        self._was_down[index] = False
        self._last = index

    # -- engines
    async def _chunks(self, engine, text: str, deadline: float | None):
        if hasattr(engine, "synthesize_stream"):
            stream = engine.synthesize_stream(text)
            async with contextlib.aclosing(stream):
                limit = deadline
                while True:
                    try:
                        chunk = await asyncio.wait_for(stream.__anext__(), limit)
                    except StopAsyncIteration:
                        return
                    limit = None if deadline is None else self.chunk_deadline
                    yield resample_audio(chunk, engine.sample_rate, self.sample_rate)
        else:
            audio = await asyncio.wait_for(engine.synthesize(text), deadline)
            yield resample_audio(audio, engine.sample_rate, self.sample_rate)

    async def synthesize_stream(self, text: str):
        last_index = len(self.engines) - 1
        for index, engine in enumerate(self.engines):
            if self._cooling(index):
                continue
            deadline = None if index == last_index else self.deadline
            started = False
            try:
                async with contextlib.aclosing(self._chunks(engine, text, deadline)) as chunks:
                    async for chunk in chunks:
                        if len(chunk):
                            if not started:
                                started = True
                                self._mark_ok(index)
                            yield chunk
                if not started and index != last_index and clean_for_speech(text):
                    raise RuntimeError("no audio")
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if index == last_index:
                    raise
                self._mark_down(index, str(exc) or type(exc).__name__)
                if started:
                    return  # part of the sentence was already played

    async def synthesize(self, text: str) -> np.ndarray:
        parts = [chunk async for chunk in self.synthesize_stream(text)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    async def warm_up(self) -> None:
        last_index = len(self.engines) - 1
        for index, engine in enumerate(self.engines):
            if self._cooling(index):
                continue
            try:
                await asyncio.wait_for(engine.warm_up(), None if index == last_index else self.deadline * 2)
                self._last = index
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if index == last_index:
                    raise
                self._mark_down(index, str(exc) or type(exc).__name__)


def resolve_tts_provider(settings: Settings) -> str:
    """The provider order `auto` stands for, or the explicit choice.

    auto: Edge (British voice, no key), then NVIDIA when a key is set, then Piper. In local
    mode (provider ollama) nothing leaves the machine, so auto means Piper only.
    """
    choice = (settings.tts_provider or "auto").strip().lower()
    if choice not in {"auto", "edge", "nvidia", "piper"}:
        log.warning("Unknown JARVIS_TTS_PROVIDER %r; using auto", choice)
        choice = "auto"
    return choice


def build_synthesizer(settings: Settings):
    """Piper alone, or online engines in front of it (see FallbackSynthesizer)."""
    piper = PiperSynthesizer(settings.tts_voice, settings.voices_dir, settings.tts_speed)
    choice = resolve_tts_provider(settings)
    wanted: list[str]
    if choice == "piper":
        return piper
    if choice == "auto":
        if settings.provider.strip().lower() == "ollama":
            return piper
        wanted = ["edge"] + (["nvidia"] if settings.nvidia_api_key else [])
    else:
        wanted = [choice]
    engines: list = []
    for provider in wanted:
        try:
            if provider == "edge":
                from jarvis.voice.tts_edge import DEFAULT_ONLINE_VOICE, EdgeSynthesizer

                engines.append(EdgeSynthesizer(
                    settings.tts_online_voice or DEFAULT_ONLINE_VOICE, settings.tts_rate, settings.tts_pitch,
                ))
            else:
                from jarvis.voice.tts_nvidia import DEFAULT_VOICE as NV_VOICE, NvidiaSynthesizer

                if not settings.nvidia_api_key:
                    log.warning("JARVIS_TTS_PROVIDER=nvidia needs NVIDIA_API_KEY; skipping it")
                    continue
                voice = settings.tts_online_voice if choice == "nvidia" and settings.tts_online_voice else NV_VOICE
                engines.append(NvidiaSynthesizer(settings.nvidia_api_key, voice))
        except ImportError as exc:
            log.warning("Online speech engine %s is not installed (%s)", provider, exc)
    if not engines:
        return piper
    return FallbackSynthesizer([*engines, piper])
