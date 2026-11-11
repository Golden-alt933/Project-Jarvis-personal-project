"""Speech layer tests. Fast tests need no model; slow ones need JARVIS_SLOW_TESTS=1."""

from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from jarvis.voice import stt
from jarvis.voice.stt import WhisperTranscriber, filter_hallucination, score_segments
from jarvis.voice.tts import SentenceChunker, clean_for_speech

slow = pytest.mark.skipif(os.environ.get("JARVIS_SLOW_TESTS") != "1", reason="set JARVIS_SLOW_TESTS=1")


def chunks_whole(text: str, **kw) -> list[str]:
    c = SentenceChunker(**kw)
    return c.feed(text) + c.flush()


def chunks_streamed(text: str, **kw) -> list[str]:
    c = SentenceChunker(**kw)
    out: list[str] = []
    for ch in text:
        out += c.feed(ch)
    return out + c.flush()


# ------------------------------------------------------------------ clean_for_speech

CLEAN_CASES = [
    ("**bold** and *italic*", "bold and italic"),
    ("__bold__ and _italic_", "bold and italic"),
    ("snake_case_name stays", "snake_case_name stays"),
    ("# Heading", "Heading"),
    ("- one\n- two\n- three", "one. two. three."),
    ("* item\n+ other", "item. other."),
    ("Use `ls -la` now", "Use ls -la now"),
    ("See [the docs](https://example.com/a/b) please", "See the docs please"),
    ("Visit https://www.example.com/path?q=1 today", "Visit example.com today"),
    ("Go to https://docs.python.org/3/.", "Go to docs.python.org."),
    ("It is 21 °C outside", "It is 21 degrees Celsius outside"),
    ("Battery at 50%", "Battery at 50 percent"),
    ("Tom & Jerry", "Tom and Jerry"),
    ("A → B", "A to B"),
    ("a -> b", "a to b"),
    ("| a | b |\n|---|---|\n| 1 | 2 |", "a, b. 1, 2."),
    ("before\n```python\nprint(1)\n```\nafter", "before. after."),
    ("~~gone~~ text", "gone text"),
    ("> quoted line", "quoted line"),
    ("Pi is 3.14 and v1.2.3", "Pi is 3.14 and v1.2.3"),
    ("  lots   of   space  ", "lots of space"),
    ("Intro:\n\n\nBody text", "Intro: Body text."),
    ("Nice \U0001F600 job", "Nice job"),
    ("", ""),
]


@pytest.mark.parametrize("raw,expected", CLEAN_CASES)
def test_clean_for_speech(raw, expected):
    assert clean_for_speech(raw) == expected


# ------------------------------------------------------------------ SentenceChunker

STREAM_TEXTS = [
    "Hello there. How are you doing today? I am fine, thanks for asking! Lovely weather.",
    "Dr. Smith met Mr. Jones in the U.S. last week, e.g. on Monday. They agreed to meet at 10.30 again.",
    "Pi is 3.14 and the version is 2.0.1 so we upgrade. Well... that was unexpected for everyone here.",
    "Intro line that is long enough.\n```python\nx = 1. 2\n```\nAfter the code block we continue speaking.",
    "- first item in the list\n- second item in the list\n- third item in the list\n",
    "word " * 120,
    "A very long clause, " * 30 + "end.",
]


@pytest.mark.parametrize("text", STREAM_TEXTS)
def test_streaming_equals_whole(text):
    assert chunks_streamed(text) == chunks_whole(text)


def test_sentences_split_when_long_enough():
    out = chunks_whole("The quick brown fox jumps. Then it rests under a tree. Done now, truly.")
    assert out[0] == "The quick brown fox jumps."
    assert out[1] == "Then it rests under a tree."


def test_short_sentences_merge_until_min_chars():
    assert chunks_whole("Yes. I did it.") == ["Yes. I did it."]


def test_first_chunk_released_early_at_clause_boundary():
    c = SentenceChunker()
    text = "Well, when you consider all the moving parts involved in this thing, it gets complicated and long"
    out: list[str] = []
    for i, ch in enumerate(text):
        out += c.feed(ch)
        if out:
            break
    assert out == ["Well, when you consider all the moving parts involved in this thing,"]
    assert i < 70


def test_clause_split_only_applies_to_first_chunk():
    c = SentenceChunker()
    first = c.feed("This opening clause is quite long enough to count, and more words follow here. ")
    assert first == ["This opening clause is quite long enough to count, and more words follow here."]
    later = c.feed("Another clause that is long enough to count, then the rest follows on. ")
    assert later == ["Another clause that is long enough to count, then the rest follows on."]


@pytest.mark.parametrize(
    "text",
    [
        "Dr. Smith will see you now, please come in and sit down.",
        "Pi is roughly 3.14 and that is a famous number in maths.",
        "Hmm... I was thinking about that for quite a long while.",
        "We met on Tuesday e.g. at the office and talked a while.",
        "Meet at 10.30 in the lobby of the building downtown today.",
        "Upgrade to version 2.0.1 before you run the migration step.",
        "He lives in the U.S. now and works for a large company.",
    ],
)
def test_no_split_inside_abbreviations_numbers_ellipses(text):
    assert chunks_whole(text) == [text]


def test_code_fence_never_split_or_spoken():
    out = chunks_whole("Here it is. Look:\n```\nfoo. bar. baz.\nqux\n```\nThat was the code, nice and short.")
    assert not any("foo" in c for c in out)
    assert "omitted" not in " ".join(out) and out[-1].startswith("That was the code")


def test_open_fence_waits_for_close():
    c = SentenceChunker()
    assert c.feed("Intro text that is long enough to go.\n```py\nx = 1. y") == ["Intro text that is long enough to go."]
    assert c.feed("= 2\n") == []
    out = c.feed("```\nDone talking about it.\n") + c.flush()
    assert out == ["Done talking about it."]


def test_long_sentence_split_near_max_chars():
    text = ", ".join(f"item number {i}" for i in range(40)) + "."
    out = chunks_whole(text, max_chars=100)
    assert len(out) > 1
    assert all(len(c) <= 101 for c in out)
    assert " ".join(out).replace(",", "") == text.replace(",", "")


def test_unbroken_text_hard_split():
    out = chunks_whole("x" * 500, max_chars=100)
    assert all(len(c) <= 100 for c in out) and "".join(out) == "x" * 500


def test_flush_returns_remainder_and_resets():
    c = SentenceChunker()
    assert c.feed("This is a finished sentence. And a trailing bit") == ["This is a finished sentence."]
    assert c.flush() == ["And a trailing bit"]
    assert c.flush() == []


def test_empty_and_markup_only_chunks_dropped():
    assert chunks_whole("---\n\n**") == []


# ------------------------------------------------------------------ Whisper logic


def seg(text, no_speech=0.0, logprob=-0.2, compression=1.2):
    return SimpleNamespace(text=text, no_speech_prob=no_speech, avg_logprob=logprob, compression_ratio=compression)


def test_confident_speech_kept():
    s = score_segments([seg(" Open the notes folder.", 0.01, -0.15)])
    assert s.confidence > 0.8
    assert filter_hallucination(s, rms=0.05) == "Open the notes folder."


def test_non_speech_segment_returns_empty():
    s = score_segments([seg(" Thanks for watching!", no_speech=0.9, logprob=-1.6)])
    assert s.non_speech
    assert filter_hallucination(s, rms=0.05) == ""


def test_repetition_loop_dropped():
    s = score_segments([seg(" la la la la la la la", compression=3.5)])
    assert filter_hallucination(s, rms=0.05) == ""


def test_hallucination_phrase_dropped_when_quiet():
    s = score_segments([seg(" Thank you.", 0.2, -0.3)])
    assert filter_hallucination(s, rms=0.004) == ""


def test_hallucination_phrase_dropped_when_unsure():
    s = score_segments([seg(" you", 0.4, -0.9)])
    assert s.confidence < stt.POOR_CONFIDENCE
    assert filter_hallucination(s, rms=0.1) == ""


def test_clear_thank_you_kept():
    s = score_segments([seg(" Thank you.", 0.02, -0.1)])
    assert filter_hallucination(s, rms=0.08) == "Thank you."


@pytest.mark.parametrize("word", ["Yes.", "No.", "Stop.", "Okay."])
def test_short_real_replies_never_dropped(word):
    s = score_segments([seg(f" {word}", 0.05, -0.25)])
    assert filter_hallucination(s, rms=0.006) == word


def test_no_segments_is_empty():
    s = score_segments([])
    assert s.text == "" and filter_hallucination(s, 0.1) == ""


def test_confidence_is_length_weighted():
    s = score_segments([seg(" a", 0.0, -2.0), seg(" long confident segment here", 0.0, -0.1)])
    assert 0.6 < s.confidence < 1.0


# ---------------------------------------------------------- device/model selection


def test_auto_device_cuda_when_vram_free(monkeypatch):
    monkeypatch.setattr(stt, "_cuda_device_count", lambda: 1)
    monkeypatch.setattr(stt, "_free_vram_mb", lambda: 3000)
    assert stt.resolve_device("auto") == "cuda"


def test_auto_device_cpu_when_vram_low(monkeypatch):
    monkeypatch.setattr(stt, "_cuda_device_count", lambda: 1)
    monkeypatch.setattr(stt, "_free_vram_mb", lambda: 800)
    assert stt.resolve_device("auto") == "cpu"


def test_auto_device_cpu_without_cuda_or_smi(monkeypatch):
    monkeypatch.setattr(stt, "_cuda_device_count", lambda: 0)
    assert stt.resolve_device("auto") == "cpu"
    monkeypatch.setattr(stt, "_cuda_device_count", lambda: 1)
    monkeypatch.setattr(stt, "_free_vram_mb", lambda: None)
    assert stt.resolve_device("auto") == "cpu"


def test_explicit_device_and_model_respected(monkeypatch):
    monkeypatch.setattr(stt, "_cuda_device_count", lambda: 0)
    t = WhisperTranscriber("tiny.en", "cpu")
    assert (t.model_name, t.device) == ("tiny.en", "cpu")
    assert "tiny.en" in t.name and "cpu" in t.name


def test_auto_model_depends_on_device():
    assert stt.resolve_model("auto", "cuda") == stt.AUTO_MODEL_CUDA
    assert stt.resolve_model("auto", "cpu") == stt.AUTO_MODEL_CPU
    assert stt.resolve_model("small.en", "cpu") == "small.en"


class FakeModel:
    def __init__(self, fail: bool):
        self.fail = fail

    def transcribe(self, pcm, **kwargs):
        if self.fail:
            raise RuntimeError("Library cublas64_12.dll is not found")
        return iter([seg(" hello there", 0.01, -0.1)]), None


async def test_cuda_failure_at_load_falls_back_to_cpu(monkeypatch):
    t = WhisperTranscriber("auto", "cuda")

    def build(device, compute_type):
        if device == "cuda":
            raise RuntimeError("no cuda")
        return FakeModel(fail=False)

    monkeypatch.setattr(t, "_build", build)
    result = await t.transcribe(np.full(16000, 0.1, dtype=np.float32))
    assert result.text == "hello there"
    assert t.device == "cpu" and "cpu" in t.name


async def test_cuda_failure_at_inference_falls_back_to_cpu(monkeypatch):
    t = WhisperTranscriber("auto", "cuda")
    t._load = lambda: None  # model "loaded" fine on CUDA ...
    t._model = FakeModel(fail=True)  # ... but inference blows up

    monkeypatch.setattr(t, "_build", lambda device, ct: FakeModel(fail=False))
    result = await t.transcribe(np.full(16000, 0.1, dtype=np.float32))
    assert result.text == "hello there"
    assert t.device == "cpu"


async def test_silence_skips_model():
    t = WhisperTranscriber("tiny.en", "cpu")
    t._build = lambda *a: pytest.fail("model must not load for silence")
    result = await t.transcribe(np.zeros(16000, dtype=np.float32))
    assert result.text == "" and result.confidence == 0.0


# ---------------------------------------------------------------- slow (real models)


@slow
async def test_round_trip_synthesize_then_transcribe():
    from jarvis.config import Settings
    from jarvis.voice.tts import build_synthesizer

    synth = build_synthesizer(Settings())
    audio = await synth.synthesize("Jarvis, please open the notes folder.")
    assert audio.dtype == np.float32 and audio.size > synth.sample_rate

    n = int(len(audio) * 16000 / synth.sample_rate)
    pcm = np.interp(np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio).astype(np.float32)
    transcriber = WhisperTranscriber("auto", "cpu")
    result = await transcriber.transcribe(pcm, hint="Jarvis")
    text = result.text.lower()
    assert "jarvis" in text and "notes folder" in text


def test_initial_prompt_wraps_bare_name_only():
    assert stt.initial_prompt("jarvis") == "Hey Jarvis, can you help me?"
    assert stt.initial_prompt("Project names: Jarvis") == "Project names: Jarvis"
    assert stt.initial_prompt(None) is None and stt.initial_prompt("  ") is None


def test_code_blocks_are_extracted_as_written():
    from jarvis.voice.events import code_blocks

    reply = (
        "Here is the fix.\n```python\ndef two_sum(nums, target):\n    seen = {}\n"
        "    for i, n in enumerate(nums):\n        if target - n in seen:\n"
        "            return [seen[target - n], i]\n        seen[n] = i\n```\nAnd a shell line:\n"
        "```\npytest -q\n```\nThat is all."
    )
    blocks = code_blocks(reply)
    assert [b.language for b in blocks] == ["python", ""]
    assert blocks[0].code.splitlines()[1] == "    seen = {}"  # indentation survives
    assert blocks[0].code.endswith("        seen[n] = i") and blocks[1].code == "pytest -q"
    assert code_blocks("no code here") == [] and code_blocks("```\n\n```") == []
    assert code_blocks("cut off\n```js\nlet a = 1")[0].code == "let a = 1"  # unterminated block
