"""LLM layer tests. No network: chunks are faked and clients are never called."""

from __future__ import annotations

import httpx
import openai
import pytest

from jarvis.config import Settings
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
    PROFILES,
    build_clients,
    build_fast_client,
    describe,
)
from jarvis.llm.types import (
    LLMBadRequest,
    LLMClient,
    LLMRateLimited,
    LLMUnavailable,
    Message,
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
    client = OllamaClient("http://127.0.0.1:1", "qwen", 4096, request_timeout=2)
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
    assert build_fast_client(make_settings()).model == build_clients(make_settings())[0].model
    local = build_fast_client(make_settings(JARVIS_PROVIDER="ollama"))
    assert local.is_local
