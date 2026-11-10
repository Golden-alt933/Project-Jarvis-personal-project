"""SQLite-backed memory: durable conversations, summaries and facts.

Durability is the point. Every write commits before returning, the journal is
WAL with ``synchronous=FULL``, and there is no save-on-exit: killing the process
at any moment loses nothing that a method already returned from.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

from jarvis.llm.types import Message

log = logging.getLogger(__name__)

PROFILE_CATEGORY = "profile"
_PROFILE_CAP = 3

_STOP_WORDS = frozenset(
    """a about after all am an and any are as at be been but by can could did do does
    for from had has have he her him his how i if in into is it its me my of on or our
    she so than that the their them then there these they this to was we were what when
    where which who whom why will with would you your tell know please""".split()
)
_TOKEN = re.compile(r"\w+", re.UNICODE)
_WS = re.compile(r"\s+")


@dataclass
class Fact:
    id: int
    text: str
    category: str
    created_at: str
    updated_at: str


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _normalise(text: str) -> str:
    return _WS.sub(" ", text).strip()


def _tokens(query: str) -> list[str]:
    seen: list[str] = []
    for tok in _TOKEN.findall(query.lower()):
        if len(tok) >= 2 and tok not in _STOP_WORDS and tok not in seen:
            seen.append(tok)
    return seen


def _fts_query(tokens: list[str]) -> str:
    # Tokens are \w+ only, so quoting is sufficient to neutralise FTS5 syntax.
    return " OR ".join(f'"{t}"' for t in tokens)


def _like_escape(token: str) -> str:
    return token.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _fts5_available() -> bool:
    try:
        probe = sqlite3.connect(":memory:")
        try:
            probe.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        finally:
            probe.close()
        return True
    except sqlite3.Error:
        return False


# -- schema -----------------------------------------------------------------

_BASE_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    summary TEXT,
    summary_upto INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    payload TEXT NOT NULL,
    text TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);
CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at);
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    category TEXT NOT NULL DEFAULT 'general',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_facts_category ON facts(category);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts USING fts5(
    text, content='facts', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO facts_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    text, content='messages', content_rowid='id');
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, text) VALUES ('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, text) VALUES ('delete', old.id, old.text);
    INSERT INTO messages_fts(rowid, text) VALUES (new.id, new.text);
END;
"""


def _migrate_v1(conn: sqlite3.Connection, fts: bool) -> None:
    conn.executescript(_BASE_SCHEMA)
    if fts:
        conn.executescript(_FTS_SCHEMA)


def _migrate_v2(conn: sqlite3.Connection, fts: bool) -> None:
    """Rebuild the facts index with stemming, so 'temperature' finds 'temperatures'."""
    if not fts:
        return
    conn.executescript(
        """
        DROP TABLE IF EXISTS facts_fts;
        CREATE VIRTUAL TABLE facts_fts USING fts5(
            text, content='facts', content_rowid='id', tokenize='porter unicode61');
        INSERT INTO facts_fts(facts_fts) VALUES ('rebuild');
        """
    )


# Each entry upgrades the schema from (index) to (index + 1). Append, never edit.
_MIGRATIONS: list[Callable[[sqlite3.Connection, bool], None]] = [_migrate_v1, _migrate_v2]
SCHEMA_VERSION = len(_MIGRATIONS)


class MemoryStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
        self._fts = _fts5_available()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = self._open_with_recovery()

    # -- lifecycle ----------------------------------------------------------
    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=FULL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._migrate(conn)
        except BaseException:
            conn.close()
            raise
        return conn

    def _open_with_recovery(self) -> sqlite3.Connection:
        try:
            return self._connect()
        except sqlite3.DatabaseError as exc:
            if not self.db_path.exists():
                raise
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
            quarantine = self.db_path.with_name(f"{self.db_path.name}.corrupt-{stamp}")
            log.warning(
                "memory database %s is unreadable (%s); moving it to %s and starting fresh",
                self.db_path, exc, quarantine,
            )
            self.db_path.replace(quarantine)
            for suffix in ("-wal", "-shm"):
                side = self.db_path.with_name(self.db_path.name + suffix)
                if side.exists():
                    side.replace(quarantine.with_name(quarantine.name + suffix))
            return self._connect()

    def _migrate(self, conn: sqlite3.Connection) -> None:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            log.warning("database schema v%s is newer than this app (v%s)", version, SCHEMA_VERSION)
            return
        for target in range(version, SCHEMA_VERSION):
            _MIGRATIONS[target](conn, self._fts)
            conn.execute(f"PRAGMA user_version={target + 1}")
            conn.commit()

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                log.exception("closing memory database failed")

    # -- generic access -----------------------------------------------------
    def ensure_schema(self, ddl: str) -> None:
        """Run idempotent DDL (CREATE TABLE IF NOT EXISTS ...) for a tool-owned table."""
        with self._lock:
            self._conn.executescript(ddl)
            self._conn.commit()

    def execute(self, sql: str, params: Sequence = ()) -> list[sqlite3.Row]:
        with self._lock:
            try:
                rows = self._conn.execute(sql, params).fetchall()
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
            return rows

    # -- conversations ------------------------------------------------------
    def start_conversation(self) -> str:
        cid = str(uuid.uuid4())
        now = _now()
        self.execute(
            "INSERT INTO conversations(id, started_at, updated_at) VALUES (?, ?, ?)",
            (cid, now, now),
        )
        return cid

    def latest_conversation(self, max_age_hours: float | None = None) -> str | None:
        rows = self.execute(
            # Empty conversations are skipped: every Agent starts one at construction,
            # and resuming that instead of the last real one would lose the context.
            "SELECT id, updated_at FROM conversations c"
            " WHERE EXISTS (SELECT 1 FROM messages m WHERE m.conversation_id = c.id)"
            " ORDER BY updated_at DESC LIMIT 1"
        )
        if not rows:
            return None
        if max_age_hours is not None:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
            if datetime.fromisoformat(rows[0]["updated_at"]) < cutoff:
                return None
        return rows[0]["id"]

    def append_message(self, conversation_id: str, message: Message) -> int:
        now = _now()
        text = message.content if message.role in ("user", "assistant") else ""
        payload = json.dumps(message.to_dict(), ensure_ascii=False)
        with self._lock:
            try:
                cur = self._conn.execute(
                    "INSERT INTO messages(conversation_id, role, payload, text, created_at)"
                    " VALUES (?, ?, ?, ?, ?)",
                    (conversation_id, message.role, payload, text, now),
                )
                self._conn.execute(
                    "UPDATE conversations SET updated_at=? WHERE id=?", (now, conversation_id)
                )
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
            return int(cur.lastrowid)

    def load_messages(
        self, conversation_id: str, after_id: int = 0
    ) -> list[tuple[int, Message]]:
        rows = self.execute(
            "SELECT id, payload FROM messages WHERE conversation_id=? AND id>? ORDER BY id",
            (conversation_id, after_id),
        )
        return [(r["id"], Message.from_dict(json.loads(r["payload"]))) for r in rows]

    def get_summary(self, conversation_id: str) -> tuple[str, int] | None:
        rows = self.execute(
            "SELECT summary, summary_upto FROM conversations WHERE id=?", (conversation_id,)
        )
        if not rows or rows[0]["summary"] is None:
            return None
        return rows[0]["summary"], rows[0]["summary_upto"]

    def set_summary(self, conversation_id: str, summary: str, upto_message_id: int) -> None:
        self.execute(
            "UPDATE conversations SET summary=?, summary_upto=? WHERE id=?",
            (summary, upto_message_id, conversation_id),
        )

    def search_messages(self, query: str, limit: int = 8) -> list[dict[str, Any]]:
        tokens = _tokens(query)
        if not tokens:
            return []
        if self._fts:
            sql = (
                "SELECT m.conversation_id, m.role, m.text, m.created_at FROM messages_fts f"
                " JOIN messages m ON m.id = f.rowid"
                " WHERE messages_fts MATCH ? AND m.role IN ('user','assistant') AND m.text != ''"
                " ORDER BY bm25(messages_fts) LIMIT ?"
            )
            params: list[Any] = [_fts_query(tokens), limit]
        else:
            clause = " OR ".join("m.text LIKE ? ESCAPE '\\'" for _ in tokens)
            sql = (
                "SELECT m.conversation_id, m.role, m.text, m.created_at FROM messages m"
                f" WHERE m.role IN ('user','assistant') AND ({clause})"
                " ORDER BY m.id DESC LIMIT ?"
            )
            params = [f"%{_like_escape(t)}%" for t in tokens] + [limit]
        return [
            {
                "conversation_id": r["conversation_id"],
                "role": r["role"],
                "text": r["text"] if len(r["text"]) <= 300 else r["text"][:299] + "…",
                "created_at": r["created_at"],
            }
            for r in self.execute(sql, params)
        ]

    # -- facts --------------------------------------------------------------
    @staticmethod
    def _fact(row: sqlite3.Row) -> Fact:
        return Fact(row["id"], row["text"], row["category"], row["created_at"], row["updated_at"])

    def remember(self, text: str, category: str = "general") -> Fact:
        clean = _normalise(text)
        if not clean:
            raise ValueError("cannot remember an empty fact")
        now = _now()
        with self._lock:
            try:
                row = self._conn.execute(
                    "SELECT * FROM facts WHERE lower(text)=?", (clean.lower(),)
                ).fetchone()
                if row is not None:
                    self._conn.execute("UPDATE facts SET updated_at=? WHERE id=?", (now, row["id"]))
                else:
                    cur = self._conn.execute(
                        "INSERT INTO facts(text, category, created_at, updated_at)"
                        " VALUES (?, ?, ?, ?)",
                        (clean, category, now, now),
                    )
                    row = self._conn.execute(
                        "SELECT * FROM facts WHERE id=?", (cur.lastrowid,)
                    ).fetchone()
                self._conn.commit()
                fresh = self._conn.execute("SELECT * FROM facts WHERE id=?", (row["id"],)).fetchone()
            except BaseException:
                self._conn.rollback()
                raise
            return self._fact(fresh)

    def _matching_facts(self, tokens: list[str], limit: int) -> list[Fact]:
        if self._fts:
            rows = self.execute(
                "SELECT f.* FROM facts_fts JOIN facts f ON f.id = facts_fts.rowid"
                " WHERE facts_fts MATCH ? ORDER BY bm25(facts_fts) LIMIT ?",
                (_fts_query(tokens), limit),
            )
        else:
            clause = " OR ".join("text LIKE ? ESCAPE '\\'" for _ in tokens)
            rows = self.execute(
                f"SELECT * FROM facts WHERE {clause} ORDER BY updated_at DESC LIMIT ?",
                [f"%{_like_escape(t)}%" for t in tokens] + [limit],
            )
        return [self._fact(r) for r in rows]

    def recall(self, query: str, limit: int = 8) -> list[Fact]:
        if limit <= 0:
            return []
        profile = self.list_facts(PROFILE_CATEGORY, limit=min(_PROFILE_CAP, limit))
        result = list(profile)
        seen = {f.id for f in result}
        tokens = _tokens(query)
        found = self._matching_facts(tokens, limit + len(seen)) if tokens else []
        for fact in found:
            if len(result) >= limit:
                break
            if fact.id not in seen:
                result.append(fact)
                seen.add(fact.id)
        # Always top up with recent facts: keyword matching misses paraphrases, and a
        # few extra lines in the prompt cost far less than forgetting something.
        if len(result) < limit:
            for fact in self.list_facts(limit=limit + len(seen)):
                if len(result) >= limit:
                    break
                if fact.id not in seen:
                    result.append(fact)
                    seen.add(fact.id)
        return result

    def list_facts(self, category: str | None = None, limit: int = 50) -> list[Fact]:
        if category is None:
            rows = self.execute("SELECT * FROM facts ORDER BY updated_at DESC, id DESC LIMIT ?", (limit,))
        else:
            rows = self.execute(
                "SELECT * FROM facts WHERE category=? ORDER BY updated_at DESC, id DESC LIMIT ?",
                (category, limit),
            )
        return [self._fact(r) for r in rows]

    def forget(self, fact_id: int) -> bool:
        with self._lock:
            try:
                cur = self._conn.execute("DELETE FROM facts WHERE id=?", (fact_id,))
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
            return cur.rowcount > 0
