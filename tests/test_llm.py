"""LLM layer tests. No network: chunks are faked and clients are never called."""

from __future__ import annotations

import asyncio
import types

import httpx
import openai
import pytest

from jarvis.config import Settings
from jarvis.llm import ollama as ollama_mod
from jarvis.llm import providers as providers_mod
from jarvis.llm import types as llm_types
from jarvis.llm.ollama import NDJSONAssembler, OllamaClient, message_to_ollama
from jarvis.llm.openai_compat import (
    ChunkAssembler,
    OpenAICompatClient,
    ThinkStripper,
    map_openai_error,
    parse_arguments,
)
from jarvis.llm.providers import (
    ConfigError,
    LIGHTNING_MODEL,
    MODEL_OPTIONS,
    NVIDIA_DEFAULT_MODEL,
    PROFILES,
    build_clients,
    build_fast_client,
    build_voice_client,
    describe,
)
from jarvis.llm.types import (
    LLMBadRequest,
    LLMClient,
    LLMRateLimited,
    LLMUnavailable,
    Message,
    ResponseDone,
    TextDelta,
    ToolCall,
)


def chunk(content=None, tool_calls=None, finish=None, usage=None, **delta_extra):
    delta = {"content": content, "tool_calls": tool_calls, **delta_extra}
    choices = [{"delta": delta, "finish_reason": finish}] if (delta or finish) else []
    out = {"choices": choices}
    if usage:
        out["usage"] = usage
    return out


def tc(index, id=None, name=None, args=None):
    fn = {}
    if name is not None:
        fn["name"] = name
    if args is not None:
        fn["arguments"] = args
    d = {"index": index, "function": fn}
    if id:
        d["id"] = id
    return d


def run(chunks):
    a = ChunkAssembler("m")
    deltas = [t for c in chunks for t in a.feed(c)]
    tail, resp = a.finish()
    return deltas + tail, resp


# ---- stream assembly -------------------------------------------------------

def test_text_only_with_usage():
    deltas, resp = run([
        chunk("Hel"), chunk("lo"), chunk(finish="stop"),
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}},
    ])
    assert deltas == ["Hel", "lo"]
    assert resp.content == "Hello" and resp.tool_calls == []
    assert resp.finish_reason == "stop"
    assert (resp.usage.prompt_tokens, resp.usage.completion_tokens) == (5, 2)


def test_tool_call_split_across_chunks():
    pieces = ['{"ci', 'ty": "Par', 'is", "n":', ' 2}']
    chunks = [chunk(tool_calls=[tc(0, "call_1", "get_weather", "")])]
    chunks += [chunk(tool_calls=[tc(0, args=p)]) for p in pieces]
    chunks.append(chunk(finish="tool_calls"))
    deltas, resp = run(chunks)
    assert deltas == []
    assert resp.tool_calls == [ToolCall("call_1", "get_weather", {"city": "Paris", "n": 2})]
    assert resp.finish_reason == "tool_calls"


def test_two_parallel_tool_calls_interleaved():
    _, resp = run([
        chunk(tool_calls=[tc(0, "a", "one", '{"x":'), tc(1, "b", "two", '{"y":')]),
        chunk(tool_calls=[tc(1, args="2}")]),
        chunk(tool_calls=[tc(0, args="1}")]),
        chunk(finish="tool_calls"),
    ])
    assert [(c.id, c.name, c.arguments) for c in resp.tool_calls] == [
        ("a", "one", {"x": 1}), ("b", "two", {"y": 2}),
    ]


def test_missing_ids_generated_and_stop_becomes_tool_calls():
    _, resp = run([
        chunk(tool_calls=[tc(0, None, "one", "{}"), tc(1, None, "two", "")]),
        chunk(finish="stop"),
    ])
    ids = [c.id for c in resp.tool_calls]
    assert all(ids) and len(set(ids)) == 2
    assert resp.tool_calls[1].arguments == {}
    assert resp.finish_reason == "tool_calls"


def test_repeated_name_not_duplicated():
    _, resp = run([
        chunk(tool_calls=[tc(0, "a", "one", "{")]),
        chunk(tool_calls=[tc(0, None, "one", "}")]),
    ])
    assert resp.tool_calls[0].name == "one"


def test_invalid_json_arguments():
    assert parse_arguments("") == {}
    assert parse_arguments('```json\n{"a": 1,}\n```') == {"a": 1}
    assert parse_arguments("{not json") == {"_raw": "{not json"}
    assert parse_arguments("[1, 2]") == {"_raw": "[1, 2]"}
    _, resp = run([chunk(tool_calls=[tc(0, "a", "one", "{oops")])])
    assert resp.tool_calls[0].arguments == {"_raw": "{oops"}


def test_reasoning_fields_ignored():
    deltas, resp = run([
        chunk(None, reasoning_content="thinking hard"),
        chunk(None, reasoning="more"),
        chunk("answer"),
    ])
    assert deltas == ["answer"] and resp.content == "answer"


def test_think_tags_across_chunk_boundaries():
    deltas, resp = run([
        chunk("Hi <thi"), chunk("nk>secret"), chunk(" stuff</th"), chunk("ink> there"),
    ])
    assert "".join(deltas) == "Hi  there"
    assert resp.content == "Hi  there"


def test_think_stripper_flush_behaviour():
    s = ThinkStripper()
    assert s.feed("a <") == "a "
    assert s.flush() == "<"  # a lone '<' was ordinary text
    s = ThinkStripper()
    assert s.feed("x<think>unterminated") == "x"
    assert s.flush() == ""


# ---- Ollama ----------------------------------------------------------------

def test_ollama_ndjson_tool_call():
    a = NDJSONAssembler("q")
    lines = [
        {"message": {"role": "assistant", "content": ""}, "done": False},
        {"message": {"role": "assistant", "content": "", "tool_calls": [
            {"function": {"name": "get_time", "arguments": {"tz": "UTC"}}}]}, "done": False},
        {"message": {"content": ""}, "done": True, "done_reason": "stop",
         "prompt_eval_count": 10, "eval_count": 4, "model": "q"},
    ]
    deltas = [t for ln in lines for t in a.feed(ln)]
    tail, resp = a.finish()
    assert deltas == [] and tail == []
    assert resp.tool_calls[0].name == "get_time"
    assert resp.tool_calls[0].arguments == {"tz": "UTC"}
    assert resp.tool_calls[0].id
    assert resp.finish_reason == "tool_calls"
    assert resp.usage.prompt_tokens == 10 and resp.usage.completion_tokens == 4


def test_ollama_text_and_think_stripping():
    a = NDJSONAssembler()
    out = []
    for c in ["<think>x", "</think>Hel", "lo"]:
        out += a.feed({"message": {"content": c}, "done": False})
    a.feed({"message": {"content": ""}, "done": True})
    _, resp = a.finish()
    assert "".join(out) == "Hello" and resp.content == "Hello"


def test_ollama_error_object():
    with pytest.raises(LLMBadRequest):
        NDJSONAssembler().feed({"error": "boom"})


def test_ollama_string_arguments_parsed():
    a = NDJSONAssembler()
    a.feed({"message": {"tool_calls": [{"function": {"name": "f", "arguments": '{"a": 1}'}}]}})
    _, resp = a.finish()
    assert resp.tool_calls[0].arguments == {"a": 1}


# ---- message conversion ----------------------------------------------------

def conversation():
    call = ToolCall("call_1", "get_time", {"tz": "UTC"})
    return [
        Message("system", "sys"),
        Message("user", "time?"),
        Message("assistant", "", tool_calls=[call]),
        Message("tool", "12:00", tool_call_id="call_1", name="get_time"),
    ]


def test_openai_payload_conversion():
    msgs = conversation()
    client = OpenAICompatClient("nvidia", "http://x/v1", "k", "m", 1000)
    req = client._request(msgs, None, None, None)
    assert "tools" not in req and "max_tokens" not in req and "temperature" not in req
    assert req["messages"][2]["tool_calls"][0]["function"]["arguments"] == '{"tz": "UTC"}'
    assert req["messages"][3]["tool_call_id"] == "call_1"
    assert "tools" not in client._request(msgs, [], None, None)
    full = client._request(msgs, [{"type": "function"}], 50, 0.1)
    assert full["tools"] and full["max_tokens"] == 50 and full["temperature"] == 0.1


def test_ollama_payload_conversion():
    msgs = conversation()
    assert message_to_ollama(msgs[2])["tool_calls"] == [
        {"function": {"name": "get_time", "arguments": {"tz": "UTC"}}}
    ]
    assert message_to_ollama(msgs[3]) == {"role": "tool", "content": "12:00", "tool_name": "get_time"}
    client = OllamaClient("http://localhost:11434/", "qwen", 8192)
    body = client._body(msgs, [{"type": "function"}], 100, 0.2)
    assert body["options"] == {"num_ctx": 8192, "temperature": 0.2, "num_predict": 100}
    assert body["keep_alive"] == "30m" and body["stream"] is True and body["tools"]
    assert "tools" not in client._body(msgs, None, None, None)
    assert client.base_url == "http://localhost:11434"


# ---- errors ----------------------------------------------------------------

def _status_exc(cls, code, msg="nope"):
    req = httpx.Request("POST", "https://x/v1/chat/completions")
    return cls(msg, response=httpx.Response(code, request=req), body=None)


def test_error_mapping():
    req = httpx.Request("POST", "https://x")
    assert isinstance(map_openai_error(_status_exc(openai.RateLimitError, 429)), LLMRateLimited)
    assert isinstance(map_openai_error(_status_exc(openai.InternalServerError, 503)), LLMUnavailable)
    assert isinstance(map_openai_error(openai.APIConnectionError(request=req)), LLMUnavailable)
    assert isinstance(map_openai_error(openai.APITimeoutError(request=req)), LLMUnavailable)
    assert isinstance(map_openai_error(_status_exc(openai.BadRequestError, 400)), LLMBadRequest)
    assert isinstance(map_openai_error(_status_exc(openai.AuthenticationError, 401)), LLMBadRequest)


def test_error_message_hides_api_key():
    err = map_openai_error(_status_exc(openai.BadRequestError, 400, "bad key sk-SECRET123"), "sk-SECRET123")
    assert "sk-SECRET123" not in str(err)


async def test_ollama_not_running_is_unavailable():
    client = OllamaClient("http://127.0.0.1:1", "qwen", 4096, request_timeout=2, autostart=False)
    with pytest.raises(LLMUnavailable, match="not running"):
        await client.complete([Message("user", "hi")])
    assert await client.is_available() is False


# ---- providers -------------------------------------------------------------

def make_settings(**kw) -> Settings:
    base = {"JARVIS_PROVIDER": "nvidia", "NVIDIA_API_KEY": "k"}
    base.update(kw)
    return Settings(_env_file=None, **base)


def test_profiles_present():
    assert {"nvidia", "groq", "openrouter", "gemini", "custom"} <= set(PROFILES)


def test_primary_and_fallback():
    primary, fallback = build_clients(make_settings())
    assert primary.name == "nvidia" and not primary.is_local
    assert fallback is not None and fallback.is_local and fallback.name == "ollama"
    assert isinstance(primary, LLMClient) and isinstance(fallback, LLMClient)
    assert describe(primary).startswith("nvidia · ")


def test_fallback_disabled():
    _, fallback = build_clients(make_settings(JARVIS_LOCAL_FALLBACK="false"))
    assert fallback is None


def test_ollama_provider():
    primary, fallback = build_clients(make_settings(JARVIS_PROVIDER="ollama", NVIDIA_API_KEY=""))
    assert primary.is_local and fallback is None


def test_missing_key_names_env_var():
    with pytest.raises(ConfigError, match=r"NVIDIA_API_KEY.*JARVIS_PROVIDER=ollama"):
        build_clients(make_settings(NVIDIA_API_KEY=""))


def test_provider_without_default_model():
    with pytest.raises(ConfigError, match="JARVIS_MODEL"):
        build_clients(make_settings(JARVIS_PROVIDER="groq", GROQ_API_KEY="k"))
    primary, _ = build_clients(make_settings(JARVIS_PROVIDER="groq", GROQ_API_KEY="k", JARVIS_MODEL="m"))
    assert primary.model == "m"


def test_custom_and_unknown_provider():
    with pytest.raises(ConfigError, match="CUSTOM_BASE_URL"):
        build_clients(make_settings(JARVIS_PROVIDER="custom", JARVIS_MODEL="m"))
    primary, _ = build_clients(
        make_settings(JARVIS_PROVIDER="custom", JARVIS_MODEL="m", CUSTOM_BASE_URL="http://h/v1")
    )
    assert primary.name == "custom"
    with pytest.raises(ConfigError, match="Unknown"):
        build_clients(make_settings(JARVIS_PROVIDER="nope"))


def test_fast_client():
    assert build_fast_client(make_settings(JARVIS_FAST_MODEL="small")).model == "small"
    assert build_fast_client(make_settings()).model == LIGHTNING_MODEL  # nvidia default
    local = build_fast_client(make_settings(JARVIS_PROVIDER="ollama"))
    assert local.is_local


# ---- request extras, non-streaming mode, reasoning -------------------------

class FakeStream:
    """Async-iterable standing in for the SDK stream; `steps` are chunks, exceptions or delays."""

    def __init__(self, steps):
        self._steps = list(steps)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._steps:
            raise StopAsyncIteration
        step = self._steps.pop(0)
        if isinstance(step, (int, float)):
            await asyncio.sleep(step)
            return await self.__anext__()
        if isinstance(step, Exception):
            raise step
        return step

    async def close(self):
        self.closed = True


class FakeCompletion:
    def __init__(self, message, finish="stop", usage=None):
        self._data = {
            "model": "m",
            "choices": [{"message": message, "finish_reason": finish}],
            "usage": usage,
        }

    def model_dump(self):
        return self._data


def make_client(create, **kw):
    client = OpenAICompatClient("nvidia", "http://x/v1", "k", "m", 1000, **kw)
    calls = []

    async def fake_create(**req):
        calls.append(req)
        return await create(len(calls), req)

    client._client = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=fake_create))
    )
    return client, calls


async def collect(client, **kw):
    return [e async for e in client.stream([Message("user", "hi")], **kw)]


def test_extras_merged_into_request():
    client = OpenAICompatClient(
        "nvidia", "http://x/v1", "k", "m", 1000,
        extra_body={"reasoning_effort": "none"}, extra_params={"top_p": 0.5},
    )
    req = client._request([Message("user", "hi")], None, None, None)
    assert req["extra_body"] == {"reasoning_effort": "none"}
    assert req["top_p"] == 0.5 and req["stream"] is True
    plain = OpenAICompatClient("nvidia", "http://x/v1", "k", "m", 1000)
    assert "extra_body" not in plain._request([Message("user", "hi")], None, None, None)


async def test_extras_reach_the_sdk_call():
    async def create(n, req):
        return FakeStream([chunk("ok", finish="stop")])

    client, calls = make_client(create, extra_body={"reasoning_effort": "none"})
    await collect(client)
    assert calls[0]["extra_body"] == {"reasoning_effort": "none"}
    assert calls[0]["stream_options"] == {"include_usage": True}


async def test_non_streaming_text_is_one_delta_then_done():
    msg = {"role": "assistant", "content": "Hello there", "tool_calls": None}

    async def create(n, req):
        return FakeCompletion(msg, usage={"prompt_tokens": 3, "completion_tokens": 2})

    client, calls = make_client(create, streaming=False)
    events = await collect(client)
    assert calls[0]["stream"] is False and "stream_options" not in calls[0]
    assert [type(e) for e in events] == [TextDelta, ResponseDone]
    assert events[0].text == "Hello there"
    done = events[1].response
    assert done.content == "Hello there" and done.finish_reason == "stop"
    assert (done.usage.prompt_tokens, done.usage.completion_tokens) == (3, 2)


async def test_non_streaming_tool_calls():
    msg = {
        "content": None,
        "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "get_time", "arguments": "{}"}},
            {"id": "c2", "type": "function", "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'}},
        ],
    }

    async def create(n, req):
        return FakeCompletion(msg, finish="tool_calls")

    client, _ = make_client(create, streaming=False)
    events = await collect(client, tools=[{"type": "function"}])
    assert [type(e) for e in events] == [ResponseDone]  # no text, so no delta
    calls = events[0].response.tool_calls
    assert [(c.id, c.name, c.arguments) for c in calls] == [
        ("c1", "get_time", {}), ("c2", "get_weather", {"location": "Paris"}),
    ]
    assert events[0].response.finish_reason == "tool_calls"


async def test_non_streaming_strips_think_and_maps_errors():
    async def create(n, req):
        return FakeCompletion({"content": "<think>hmm</think>Answer"})

    client, _ = make_client(create, streaming=False)
    events = await collect(client)
    assert events[0].text == "Answer"

    async def boom(n, req):
        raise _status_exc(openai.InternalServerError, 503)

    client, _ = make_client(boom, streaming=False)
    with pytest.raises(LLMUnavailable):
        await collect(client)


async def test_reasoning_never_becomes_text_streaming_and_not():
    async def create(n, req):
        return FakeStream([
            chunk(None, reasoning_content="step one "),
            chunk(None, reasoning="step two"),
            chunk("Final", finish="stop"),
        ])

    client, _ = make_client(create)
    events = await collect(client)
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["Final"]
    assert events[-1].response.content == "Final"

    async def create_plain(n, req):
        return FakeCompletion({"content": "Final", "reasoning_content": "secret thoughts"})

    client, _ = make_client(create_plain, streaming=False)
    events = await collect(client)
    assert events[0].text == "Final" and events[-1].response.content == "Final"


async def test_reasoning_only_reply_is_empty_not_the_reasoning():
    a = ChunkAssembler("m")
    assert a.feed(chunk(None, reasoning_content="abcd")) == []
    _, resp = a.finish()
    assert resp.content == "" and a.reasoning_chars == 4

    async def create(n, req):
        return FakeStream([chunk(None, reasoning_content="thinking"), chunk(finish="length")])

    client, _ = make_client(create)
    events = await collect(client)
    assert events[-1].response.content == "" and events[-1].response.finish_reason == "length"


# ---- first-byte timeout ----------------------------------------------------

async def test_first_byte_timeout_retries_once_then_succeeds():
    async def create(n, req):
        if n == 1:
            await asyncio.sleep(5)  # never starts
        return FakeStream([chunk("Hi"), chunk(finish="stop")])

    client, calls = make_client(create, first_byte_timeout=0.05)
    events = await collect(client)
    assert len(calls) == 2
    assert [e.text for e in events if isinstance(e, TextDelta)] == ["Hi"]  # not duplicated


async def test_first_byte_timeout_stalled_stream_is_closed_and_retried():
    streams = []

    async def create(n, req):
        s = FakeStream([5, chunk("late")] if n == 1 else [chunk("ok", finish="stop")])
        streams.append(s)
        return s

    client, calls = make_client(create, first_byte_timeout=0.05)
    events = await collect(client)
    assert len(calls) == 2 and streams[0].closed
    assert events[-1].response.content == "ok"


async def test_first_byte_timeout_twice_raises_unavailable():
    async def create(n, req):
        await asyncio.sleep(5)

    client, calls = make_client(create, first_byte_timeout=0.05)
    with pytest.raises(LLMUnavailable, match="did not respond"):
        await collect(client)
    assert len(calls) == 2


async def test_no_retry_once_output_has_begun():
    req = httpx.Request("POST", "https://x")

    async def create(n, r):
        return FakeStream([chunk("partial"), openai.APIConnectionError(request=req)])

    client, calls = make_client(create, first_byte_timeout=0.05)
    seen = []
    with pytest.raises(LLMUnavailable):
        async for e in client.stream([Message("user", "hi")]):
            seen.append(e)
    assert len(calls) == 1 and [e.text for e in seen] == ["partial"]


async def test_timeout_only_covers_the_first_chunk():
    async def create(n, r):
        return FakeStream([chunk("a"), 0.15, chunk("b", finish="stop")])

    client, calls = make_client(create, first_byte_timeout=0.1)
    events = await collect(client)
    assert len(calls) == 1 and events[-1].response.content == "ab"


async def test_first_byte_timeout_disabled_with_zero():
    async def create(n, r):
        await asyncio.sleep(0.1)
        return FakeStream([chunk("x", finish="stop")])

    client, calls = make_client(create, first_byte_timeout=0)
    events = await collect(client)
    assert len(calls) == 1 and events[-1].response.content == "x"


# ---- per-model options and factories --------------------------------------

def test_model_options_applied_automatically():
    assert LIGHTNING_MODEL in MODEL_OPTIONS
    primary, _ = build_clients(make_settings(JARVIS_MODEL=LIGHTNING_MODEL))
    assert primary._extra_body == MODEL_OPTIONS[LIGHTNING_MODEL].extra_body
    assert primary.streaming is True
    ultra, _ = build_clients(make_settings())
    assert ultra.model == NVIDIA_DEFAULT_MODEL and ultra._extra_body == {}
    other, _ = build_clients(make_settings(JARVIS_MODEL="Some/Other-Model"))
    assert other._extra_body == {}


def test_model_options_lookup_is_case_insensitive(monkeypatch):
    monkeypatch.setitem(
        MODEL_OPTIONS, "vendor/slow-model", providers_mod.ModelOptions(streaming=False, extra_params={"top_p": 0.1})
    )
    primary, _ = build_clients(make_settings(JARVIS_MODEL="Vendor/Slow-Model"))
    assert primary.streaming is False and primary._extra_params == {"top_p": 0.1}


def test_first_byte_timeout_setting_reaches_client():
    assert build_clients(make_settings())[0]._first_byte_timeout == 25.0
    assert build_clients(make_settings(JARVIS_FIRST_BYTE_TIMEOUT="7"))[0]._first_byte_timeout == 7.0


def test_default_fast_model_is_lightning_with_its_options():
    fast = build_fast_client(make_settings())
    assert fast.model == LIGHTNING_MODEL and fast._extra_body == {"reasoning_effort": "none"}
    assert build_fast_client(make_settings(JARVIS_FAST_MODEL="mine")).model == "mine"
    # a provider without a default fast model falls back to the primary model
    groq = build_fast_client(make_settings(JARVIS_PROVIDER="groq", GROQ_API_KEY="k", JARVIS_MODEL="g"))
    assert groq.model == "g"


def test_voice_client_factory():
    # On NVIDIA with no explicit choice, spoken turns default to the fast model.
    default = build_voice_client(make_settings())
    assert default.model == LIGHTNING_MODEL and default._extra_body == {"reasoning_effort": "none"}
    # An explicit main model is respected: no silent switch to another model.
    assert build_voice_client(make_settings(JARVIS_MODEL="some/model")) is None
    # Other providers have no measured fast default.
    assert build_voice_client(make_settings(JARVIS_PROVIDER="groq", GROQ_API_KEY="k", JARVIS_MODEL="m")) is None
    voice = build_voice_client(make_settings(JARVIS_VOICE_MODEL="fast/model"))
    assert voice.model == "fast/model" and voice.name == "nvidia" and not voice.is_local
    light = build_voice_client(make_settings(JARVIS_VOICE_MODEL=LIGHTNING_MODEL))
    assert light._extra_body == {"reasoning_effort": "none"}
    local = build_voice_client(make_settings(JARVIS_PROVIDER="ollama", JARVIS_VOICE_MODEL="tiny"))
    assert local.is_local and local.model == "tiny"


def test_voice_client_falls_back_to_none_when_unbuildable():
    assert build_voice_client(make_settings(JARVIS_VOICE_MODEL="x", NVIDIA_API_KEY="")) is None
    assert build_voice_client(make_settings(JARVIS_VOICE_MODEL="  ", JARVIS_MODEL="some/model")) is None
    assert build_voice_client(make_settings(NVIDIA_API_KEY="")) is None


def test_config_error_lives_in_types_and_is_reexported():
    assert providers_mod.ConfigError is llm_types.ConfigError


# ---- Ollama auto-start -----------------------------------------------------

@pytest.fixture(autouse=True)
def _no_real_ollama(monkeypatch):
    """Nothing in this module may start a real server; tests override the pieces they need."""
    state = ollama_mod._AutostartState()
    monkeypatch.setattr(ollama_mod, "_autostart", state)
    launches: list[str] = []
    monkeypatch.setattr(ollama_mod, "find_ollama_exe", lambda: "C:/fake/ollama.exe")
    monkeypatch.setattr(ollama_mod, "launch_ollama_serve", launches.append)

    async def up(base_url, timeout=1.5):
        return True

    monkeypatch.setattr(ollama_mod, "server_responds", up)
    monkeypatch.setattr(ollama_mod, "_STARTUP_WAIT", 0.3)
    return launches


@pytest.fixture
def launches(_no_real_ollama):
    return _no_real_ollama


def ndjson_ok(text="pong"):
    body = (
        '{"message": {"role": "assistant", "content": "%s"}, "done": false}\n'
        '{"message": {"content": ""}, "done": true, "done_reason": "stop"}\n' % text
    )
    return httpx.Response(200, content=body.encode())


def refusing_then_ok(counter):
    def handler(request: httpx.Request) -> httpx.Response:
        counter.append(1)
        if len(counter) == 1:
            raise httpx.ConnectError("refused", request=request)
        return ndjson_ok()

    return httpx.MockTransport(handler)


def always_refusing(counter):
    def handler(request: httpx.Request) -> httpx.Response:
        counter.append(1)
        raise httpx.ConnectError("refused", request=request)

    return httpx.MockTransport(handler)


def test_loopback_detection():
    for url in ("http://localhost:11434", "http://127.0.0.1:11434", "http://[::1]:11434", "http://127.0.0.2:1"):
        assert ollama_mod.is_loopback_url(url), url
    for url in ("http://192.168.1.5:11434", "http://ollama.example.com", "https://10.0.0.1:11434"):
        assert not ollama_mod.is_loopback_url(url), url


def test_find_exe_prefers_path_then_local_appdata(monkeypatch, tmp_path):
    monkeypatch.undo()  # drop the autouse stubs for this pure-function test
    monkeypatch.setattr(ollama_mod.shutil, "which", lambda name: "C:/bin/ollama.exe")
    assert ollama_mod.find_ollama_exe() == "C:/bin/ollama.exe"
    monkeypatch.setattr(ollama_mod.shutil, "which", lambda name: None)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert ollama_mod.find_ollama_exe() is None
    exe = tmp_path / "Programs" / "Ollama" / "ollama.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("")
    assert ollama_mod.find_ollama_exe() == str(exe)


async def test_autostart_then_retry_succeeds(launches):
    attempts: list[int] = []
    client = OllamaClient("http://localhost:11434", "q", 4096)
    client._transport = refusing_then_ok(attempts)
    reply = await client.complete([Message("user", "ping")])
    assert reply.content == "pong"
    assert launches == ["C:/fake/ollama.exe"] and len(attempts) == 2


async def test_autostart_happens_once_per_process(launches):
    attempts: list[int] = []
    first = OllamaClient("http://localhost:11434", "q", 4096)
    first._transport = always_refusing(attempts)
    # the server "comes up" (probe answers) but requests keep being refused
    with pytest.raises(LLMUnavailable):
        await first.complete([Message("user", "a")])
    second = OllamaClient("http://127.0.0.1:11434", "q", 4096)
    second._transport = always_refusing(attempts)
    with pytest.raises(LLMUnavailable):
        await second.complete([Message("user", "b")])
    assert len(launches) == 1


async def test_autostart_never_for_non_loopback(launches):
    client = OllamaClient("http://192.168.1.5:11434", "q", 4096)
    client._transport = always_refusing([])
    with pytest.raises(LLMUnavailable, match="not running"):
        await client.complete([Message("user", "a")])
    assert launches == []
    assert await ollama_mod.ensure_ollama_running("http://192.168.1.5:11434") is False


async def test_autostart_disabled_by_setting(launches):
    client = OllamaClient("http://localhost:11434", "q", 4096, autostart=False)
    client._transport = always_refusing([])
    with pytest.raises(LLMUnavailable, match="Start it with `ollama serve`"):
        await client.complete([Message("user", "a")])
    assert launches == []
    assert build_clients(make_settings(JARVIS_OLLAMA_AUTOSTART="false"))[1]._autostart is False
    assert build_clients(make_settings(JARVIS_OLLAMA_AUTOSTART="true"))[1]._autostart is True


async def test_autostart_failure_messages(monkeypatch, launches):
    monkeypatch.setattr(ollama_mod, "find_ollama_exe", lambda: None)
    client = OllamaClient("http://localhost:11434", "q", 4096)
    client._transport = always_refusing([])
    with pytest.raises(LLMUnavailable, match="could not be started.*not found"):
        await client.complete([Message("user", "a")])
    assert launches == []
    # the attempt is spent: a later request reports the same reason without trying again
    with pytest.raises(LLMUnavailable, match="could not be started"):
        await client.complete([Message("user", "b")])


async def test_autostart_launch_error_and_slow_server(monkeypatch):
    def broken(exe):
        raise OSError("denied")

    monkeypatch.setattr(ollama_mod, "launch_ollama_serve", broken)
    client = OllamaClient("http://localhost:11434", "q", 4096)
    client._transport = always_refusing([])
    with pytest.raises(LLMUnavailable, match="launching it failed"):
        await client.complete([Message("user", "a")])

    monkeypatch.setattr(ollama_mod, "_autostart", ollama_mod._AutostartState())
    monkeypatch.setattr(ollama_mod, "launch_ollama_serve", lambda exe: None)

    async def down(base_url, timeout=1.5):
        return False

    monkeypatch.setattr(ollama_mod, "server_responds", down)
    with pytest.raises(LLMUnavailable, match="did not answer"):
        await client.complete([Message("user", "a")])


async def test_is_available_autostarts_too(launches):
    seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(1)
        if len(seen) == 1:
            raise httpx.ConnectError("refused", request=request)
        return httpx.Response(200, json={"models": [{"name": "q:latest"}]})

    client = OllamaClient("http://localhost:11434", "q", 4096)
    client._transport = httpx.MockTransport(handler)
    assert await client.is_available() is True
    assert launches == ["C:/fake/ollama.exe"]
    off = OllamaClient("http://localhost:11434", "q", 4096, autostart=False)
    off._transport = always_refusing([])
    assert await off.is_available() is False
    assert len(launches) == 1


async def test_non_connection_errors_do_not_trigger_autostart(launches):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    client = OllamaClient("http://localhost:11434", "q", 4096)
    client._transport = httpx.MockTransport(handler)
    with pytest.raises(LLMUnavailable, match="server error 500"):
        await client.complete([Message("user", "a")])
    assert launches == []
