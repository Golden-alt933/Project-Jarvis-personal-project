from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from jarvis.config import Settings
from jarvis.llm.types import Message, ToolCall
from jarvis.memory.store import MemoryStore
from jarvis.tools.base import ToolContext
from jarvis.tools.builtin import memory_tools
from jarvis.tools.registry import ToolRegistry


@pytest.fixture
def store(tmp_path: Path):
    s = MemoryStore(tmp_path / "jarvis.db")
    yield s
    s.close()


def test_message_round_trip_with_tool_calls(store: MemoryStore):
    cid = store.start_conversation()
    msgs = [
        Message("user", "open notepad"),
        Message("assistant", "", tool_calls=[ToolCall("c1", "open_app", {"name": "notepad", "n": [1, 2]})]),
        Message("tool", "opened", tool_call_id="c1", name="open_app"),
        Message("assistant", "Done."),
    ]
    ids = [store.append_message(cid, m) for m in msgs]
    assert ids == sorted(ids)
    loaded = store.load_messages(cid)
    assert [m for _, m in loaded] == msgs
    assert [i for i, _ in loaded] == ids
    assert [m for _, m in store.load_messages(cid, after_id=ids[1])] == msgs[2:]


def test_durable_without_close(tmp_path: Path):
    path = tmp_path / "d.db"
    a = MemoryStore(path)
    cid = a.start_conversation()
    a.append_message(cid, Message("user", "hello there"))
    a.remember("The user likes tea", "preference")
    a.set_summary(cid, "greeting", 1)
    b = MemoryStore(path)  # a is never closed: simulates a kill
    assert [m.content for _, m in b.load_messages(cid)] == ["hello there"]
    assert b.recall("tea")[0].text == "The user likes tea"
    assert b.get_summary(cid) == ("greeting", 1)
    b.close()
    a.close()


def test_summary_get_set(store: MemoryStore):
    cid = store.start_conversation()
    assert store.get_summary(cid) is None
    store.set_summary(cid, "first", 5)
    store.set_summary(cid, "second", 9)
    assert store.get_summary(cid) == ("second", 9)


def test_recall_ranking_and_fallback(store: MemoryStore):
    store.remember("Maya is the user's sister")
    store.remember("The user deploys Jarvis on Windows")
    store.remember("Favourite colour is green")
    assert store.recall("sister")[0].text == "Maya is the user's sister"
    assert store.recall("where is the kubernetes cluster", limit=2)  # no hits: recent
    assert len(store.recall("nothing matches zzzz", limit=2)) == 2


@pytest.mark.parametrize(
    "query", ["the a an of", "", "what's \"my\" (name)?", "AND OR NOT *", 'foo" OR "bar', "NEAR(a b) ^ -x:y", "100% _x"]
)
def test_recall_safe_queries(store: MemoryStore, query: str):
    store.remember("The user's name is Ravi", "profile")
    store.remember("Likes jazz")
    result = store.recall(query)
    assert result and result[0].text == "The user's name is Ravi"
    store.search_messages(query)  # must not raise


def test_punctuation_query_still_matches(store: MemoryStore):
    store.remember("The user's name is Ravi")
    assert store.recall("what's \"my\" (name)?")[0].text == "The user's name is Ravi"


def test_profile_always_first_but_capped(store: MemoryStore):
    for i in range(6):
        store.remember(f"profile detail {i}", "profile")
    store.remember("Loves hiking in the mountains")
    store.remember("Hiking boots are size nine")
    result = store.recall("hiking", limit=5)
    assert len(result) == 5
    assert all(f.category == "profile" for f in result[:3])
    assert {f.text for f in result[3:]} == {"Loves hiking in the mountains", "Hiking boots are size nine"}


def test_dedupe(store: MemoryStore):
    a = store.remember("Likes   Jazz ")
    b = store.remember("likes jazz")
    assert a.id == b.id
    assert b.updated_at >= a.updated_at
    assert len(store.list_facts()) == 1


def test_forget(store: MemoryStore):
    f = store.remember("temporary")
    assert store.forget(f.id) is True
    assert store.forget(f.id) is False
    assert store.recall("temporary", limit=5) == []
    assert store.list_facts() == []


def test_list_facts_category(store: MemoryStore):
    store.remember("a1", "project")
    store.remember("b1", "person")
    assert [f.text for f in store.list_facts("project")] == ["a1"]


def test_latest_conversation_age(store: MemoryStore):
    assert store.latest_conversation() is None
    old = store.start_conversation()
    new = store.start_conversation()
    store.append_message(new, Message("user", "hi"))
    assert store.latest_conversation() == new
    assert store.latest_conversation(max_age_hours=1) == new
    past = (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat(timespec="microseconds")
    store.execute("UPDATE conversations SET updated_at=?", (past,))
    assert store.latest_conversation(max_age_hours=12) is None
    assert store.latest_conversation() in (old, new)


def test_search_messages_only_user_assistant(store: MemoryStore):
    cid = store.start_conversation()
    store.append_message(cid, Message("user", "Please summarise the quarterly budget"))
    store.append_message(cid, Message("assistant", "The budget grew by ten percent"))
    store.append_message(cid, Message("tool", "budget secret dump", tool_call_id="x", name="t"))
    hits = store.search_messages("budget")
    assert {h["role"] for h in hits} == {"user", "assistant"}
    assert all(h["conversation_id"] == cid and h["created_at"] for h in hits)
    assert store.search_messages("quarterly")[0]["text"].startswith("Please summarise")


def test_corrupt_file_quarantined(tmp_path: Path):
    path = tmp_path / "jarvis.db"
    path.write_bytes(b"this is definitely not a sqlite database" * 50)
    s = MemoryStore(path)
    try:
        assert s.list_facts() == []
        s.remember("works after recovery")
        assert len(list(tmp_path.glob("jarvis.db.corrupt-*"))) == 1
    finally:
        s.close()


def test_concurrent_writes(store: MemoryStore):
    cid = store.start_conversation()

    def work(n: int) -> None:
        for i in range(25):
            store.append_message(cid, Message("user", f"t{n} m{i}"))
            store.remember(f"fact {n}-{i}")

    threads = [threading.Thread(target=work, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(store.load_messages(cid)) == 150
    assert len(store.list_facts(limit=1000)) == 150


def test_ensure_schema_and_execute(store: MemoryStore, tmp_path: Path):
    ddl = "CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY, body TEXT)"
    store.ensure_schema(ddl)
    store.ensure_schema(ddl)  # idempotent
    store.execute("INSERT INTO notes(body) VALUES (?)", ("hi",))
    rows = store.execute("SELECT body FROM notes")
    assert rows[0]["body"] == "hi"
    other = MemoryStore(tmp_path / "jarvis.db")  # committed, visible to a second connection
    assert other.execute("SELECT count(*) AS c FROM notes")[0]["c"] == 1
    other.close()


def test_schema_version_set(store: MemoryStore):
    assert store.execute("PRAGMA user_version")[0][0] >= 1


# -- tools ------------------------------------------------------------------


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()
    assert reg.register_module(memory_tools) == 4
    return reg


def _call(name: str, **args) -> ToolCall:
    return ToolCall(id="1", name=name, arguments=args)


async def test_tools_end_to_end(registry: ToolRegistry, store: MemoryStore):
    ctx = ToolContext(settings=Settings(), memory=store)
    r = await registry.execute(_call("remember", fact="The user's dog is called Bruno", category="person"), ctx)
    assert r.ok and "Bruno" in r.content
    fact_id = r.data["id"]

    r = await registry.execute(_call("recall", query="dog"), ctx)
    assert r.ok and f"[{fact_id}]" in r.content and "Bruno" in r.content

    cid = store.start_conversation()
    store.append_message(cid, Message("user", "let's plan the Lisbon trip"))
    r = await registry.execute(_call("search_past_conversations", query="Lisbon"), ctx)
    assert r.ok and "Lisbon" in r.content

    r = await registry.execute(_call("forget", fact_id=fact_id), ctx)
    assert r.ok
    r = await registry.execute(_call("forget", fact_id=fact_id), ctx)
    assert not r.ok and "recall" in r.content


async def test_tools_reject_bad_category(registry: ToolRegistry, store: MemoryStore):
    ctx = ToolContext(settings=Settings(), memory=store)
    r = await registry.execute(_call("remember", fact="x", category="bogus"), ctx)
    assert not r.ok


async def test_tools_without_memory(registry: ToolRegistry):
    ctx = ToolContext(settings=Settings(), memory=None)
    for call in (
        _call("remember", fact="x"),
        _call("recall", query="x"),
        _call("forget", fact_id=1),
        _call("search_past_conversations", query="x"),
    ):
        r = await registry.execute(call, ctx)
        assert not r.ok and "not available" in r.content


def test_tool_schemas(registry: ToolRegistry):
    specs = {s.name: s for s in registry.specs(["memory"])}
    assert set(specs) == {"remember", "recall", "forget", "search_past_conversations"}
    props = specs["remember"].parameters["properties"]
    assert "ctx" not in props
    assert props["category"]["enum"] == ["profile", "preference", "project", "person", "general"]
