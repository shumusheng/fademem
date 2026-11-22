"""SQLite 记忆库：落盘、去重巩固、归档、遗忘整理。

一张表存记忆，一张表记录"哪些对话轮已经处理过"——后者用于在客户端每次都
把完整历史重发一遍的情况下，只对新增的那几轮做记忆抽取。
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from . import bm25, decay

__all__ = ["Memory", "Candidate", "Store"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session      TEXT    NOT NULL,
    kind         TEXT    NOT NULL,
    subject      TEXT    NOT NULL DEFAULT '',
    text         TEXT    NOT NULL,
    canonical    TEXT    NOT NULL,
    importance   REAL    NOT NULL,
    salience     REAL    NOT NULL DEFAULT 0,
    stability    REAL    NOT NULL,
    created_at   REAL    NOT NULL,
    updated_at   REAL    NOT NULL,
    access_count INTEGER NOT NULL DEFAULT 0,
    archived     INTEGER NOT NULL DEFAULT 0,
    source       TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_mem_session ON memories(session, archived);
CREATE INDEX IF NOT EXISTS idx_mem_canon   ON memories(session, canonical);

CREATE TABLE IF NOT EXISTS seen_turns (
    session TEXT NOT NULL,
    digest  TEXT NOT NULL,
    ts      REAL NOT NULL,
    PRIMARY KEY (session, digest)
);

CREATE TABLE IF NOT EXISTS meta (
    k TEXT PRIMARY KEY,
    v TEXT
);
"""


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()


def _now() -> float:
    return time.time()


@dataclass
class Candidate:
    """一条待写入的记忆候选。"""

    kind: str
    text: str
    subject: str = ""
    importance: float = 0.4
    salience: float = 0.0
    source: str = ""

    def canonical(self) -> str:
        if self.subject:
            return f"{self.kind}:{self.subject.strip().lower()}"
        return f"{self.kind}:#{_sha1(self.text)[:12]}"


@dataclass
class Memory:
    id: int
    session: str
    kind: str
    subject: str
    text: str
    canonical: str
    importance: float
    salience: float
    stability: float
    created_at: float
    updated_at: float
    access_count: int
    archived: int = 0
    source: str = ""

    # ---- 遗忘曲线相关的派生量 ------------------------------------------
    def retrievability(self, now: float | None = None) -> float:
        now = _now() if now is None else now
        return decay.retrievability(
            decay.days_between(now, self.updated_at), self.stability
        )

    def state(self, now: float | None = None) -> str:
        return decay.decay_state(self.retrievability(now))

    def half_life_days(self) -> float:
        return decay.half_life(self.stability)

    def as_dict(self, now: float | None = None) -> dict:
        now = _now() if now is None else now
        r = self.retrievability(now)
        return {
            "id": self.id,
            "session": self.session,
            "kind": self.kind,
            "subject": self.subject,
            "text": self.text,
            "importance": round(self.importance, 3),
            "salience": round(self.salience, 3),
            "stability_days": round(self.stability, 3),
            "retrievability": round(r, 3),
            "state": decay.decay_state(r),
            "half_life_days": round(decay.half_life(self.stability), 2),
            "age_days": round(decay.days_between(now, self.updated_at), 3),
            "access_count": self.access_count,
            "archived": bool(self.archived),
        }


@dataclass
class Store:
    """线程安全的记忆存储。代理服务是多线程的，所以连接上加锁。"""

    path: str
    base_stability: float = 6.0
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    _conn: sqlite3.Connection | None = field(default=None, repr=False)
    _index_cache: tuple | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        parent = os.path.dirname(os.path.abspath(self.path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(_SCHEMA)
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.Error:  # pragma: no cover
                pass
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    # ---- 内部 -----------------------------------------------------------
    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("Store 已关闭")
        return self._conn

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"],
            session=row["session"],
            kind=row["kind"],
            subject=row["subject"],
            text=row["text"],
            canonical=row["canonical"],
            importance=row["importance"],
            salience=row["salience"],
            stability=row["stability"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            access_count=row["access_count"],
            archived=row["archived"],
            source=row["source"],
        )

    def _invalidate_index(self) -> None:
        self._index_cache = None

    # ---- 写入 -----------------------------------------------------------
    def add(self, session: str, cand: Candidate, now: float | None = None) -> tuple[Memory, bool]:
        """写入一条记忆。同 subject 已存在时走"巩固"而不是"新增"。

        返回 ``(memory, created)``；``created=False`` 表示这是对旧记忆的强化。
        """
        now = _now() if now is None else now
        canonical = cand.canonical()
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM memories WHERE session=? AND canonical=? AND archived=0 "
                "ORDER BY id DESC LIMIT 1",
                (session, canonical),
            ).fetchone()

            if row is not None:
                mem = self._row_to_memory(row)
                r = mem.retrievability(now)
                new_stability = decay.reinforce(mem.stability, r)
                # 再次被提起，说明这条对"人"是重要的
                importance = min(1.0, max(mem.importance, cand.importance) + 0.04)
                text = cand.text if len(cand.text) >= len(mem.text) * 0.9 else mem.text
                self.conn.execute(
                    "UPDATE memories SET text=?, importance=?, stability=?, updated_at=?, "
                    "access_count=access_count+1 WHERE id=?",
                    (text, importance, new_stability, now, mem.id),
                )
                self.conn.commit()
                self._invalidate_index()
                refreshed = self.get(mem.id)
                assert refreshed is not None
                return refreshed, False

            stability = decay.initial_stability(
                cand.importance, cand.salience, base=self.base_stability
            )
            cur = self.conn.execute(
                "INSERT INTO memories (session, kind, subject, text, canonical, importance,"
                " salience, stability, created_at, updated_at, access_count, archived, source)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,0,0,?)",
                (
                    session,
                    cand.kind,
                    cand.subject,
                    cand.text,
                    canonical,
                    cand.importance,
                    cand.salience,
                    stability,
                    now,
                    now,
                    cand.source,
                ),
            )
            self.conn.commit()
            self._invalidate_index()
            fresh = self.get(int(cur.lastrowid))
            assert fresh is not None
            return fresh, True

    def reinforce(self, memory_id: int, now: float | None = None) -> Memory | None:
        """外部主动"想起来一次"。"""
        now = _now() if now is None else now
        with self._lock:
            mem = self.get(memory_id)
            if mem is None:
                return None
            r = mem.retrievability(now)
            self.conn.execute(
                "UPDATE memories SET stability=?, updated_at=?, access_count=access_count+1 "
                "WHERE id=?",
                (decay.reinforce(mem.stability, r), now, memory_id),
            )
            self.conn.commit()
            self._invalidate_index()
            return self.get(memory_id)

    def forget(self, memory_id: int) -> bool:
        """主动忘掉（软删除，保留可追溯）。"""
        with self._lock:
            cur = self.conn.execute(
                "UPDATE memories SET archived=1 WHERE id=?", (memory_id,)
            )
            self.conn.commit()
            self._invalidate_index()
            return cur.rowcount > 0

    def restore(self, memory_id: int) -> bool:
        with self._lock:
            cur = self.conn.execute(
                "UPDATE memories SET archived=0 WHERE id=?", (memory_id,)
            )
            self.conn.commit()
            self._invalidate_index()
            return cur.rowcount > 0

    # ---- 读取 -----------------------------------------------------------
    def get(self, memory_id: int) -> Memory | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM memories WHERE id=?", (memory_id,)
            ).fetchone()
        return self._row_to_memory(row) if row else None

    def list(
        self,
        session: str | None = None,
        include_archived: bool = False,
        limit: int = 1000,
    ) -> list[Memory]:
        sql = "SELECT * FROM memories WHERE 1=1"
        args: list[object] = []
        if session is not None:
            sql += " AND session=?"
            args.append(session)
        if not include_archived:
            sql += " AND archived=0"
        sql += " ORDER BY updated_at DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self.conn.execute(sql, args).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def sessions(self) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT session, COUNT(*) AS n, SUM(archived) AS dead, MAX(updated_at) AS last "
                "FROM memories GROUP BY session ORDER BY last DESC"
            ).fetchall()
        return [
            {
                "session": r["session"],
                "count": r["n"],
                "archived": r["dead"] or 0,
                "last_seen": r["last"],
            }
            for r in rows
        ]

    def stats(self) -> dict:
        with self._lock:
            row = self.conn.execute(
                "SELECT COUNT(*) AS total, SUM(archived) AS dead, "
                "COUNT(DISTINCT session) AS sess FROM memories"
            ).fetchone()
        total = row["total"] or 0
        return {
            "memories": total,
            "archived": row["dead"] or 0,
            "sessions": row["sess"] or 0,
            "db": os.path.abspath(self.path),
        }

    # ---- 检索 -----------------------------------------------------------
    def search(
        self,
        session: str,
        query: str,
        limit: int = 8,
        now: float | None = None,
    ) -> list[tuple[Memory, float]]:
        """BM25 召回，返回 ``[(memory, bm25_score)]``，已按分数降序。"""
        now = _now() if now is None else now
        docs = self.list(session, include_archived=False, limit=5000)
        if not docs:
            return []

        cache_key = (session, len(docs), max(d.id for d in docs))
        if self._index_cache is None or self._index_cache[0] != cache_key:
            tokens = [bm25.tokenize(f"{d.subject} {d.text}") for d in docs]
            self._index_cache = (cache_key, docs, bm25.BM25(tokens))
        _, cached_docs, index = self._index_cache

        raw = index.scores(bm25.tokenize(query))
        pairs = [(doc, score) for doc, score in zip(cached_docs, raw) if score > 0]
        pairs.sort(key=lambda p: p[1], reverse=True)
        return pairs[:limit]

    # ---- 遗忘整理 -------------------------------------------------------
    def sweep(self, session: str | None = None, forget_below: float = 0.06,
              now: float | None = None) -> int:
        """把已经彻底想不起来的记忆归档，返回归档条数。"""
        now = _now() if now is None else now
        dead: list[int] = []
        for mem in self.list(session, include_archived=False, limit=100000):
            if mem.retrievability(now) < forget_below:
                dead.append(mem.id)
        if not dead:
            return 0
        with self._lock:
            self.conn.executemany(
                "UPDATE memories SET archived=1 WHERE id=?", [(i,) for i in dead]
            )
            self.conn.commit()
        self._invalidate_index()
        return len(dead)

    def consolidate(self, session: str | None = None, threshold: float = 0.82) -> dict:
        """睡眠式整理：合并近义记忆、吸收弱的到强的。

        人睡觉时海马体会重放白天的经历，把重复的、相关的记忆压成一条。
        """
        merged = 0
        for sess in {m.session for m in self.list(session, limit=100000)}:
            mems = self.list(sess, limit=100000)
            alive: list[Memory] = []
            for mem in sorted(mems, key=lambda m: (m.access_count, m.stability), reverse=True):
                tokens = set(bm25.tokenize(mem.text))
                absorb = None
                for kept in alive:
                    if kept.kind != mem.kind:
                        continue
                    other = set(bm25.tokenize(kept.text))
                    if not tokens or not other:
                        continue
                    jaccard = len(tokens & other) / len(tokens | other)
                    if jaccard >= threshold:
                        absorb = kept
                        break
                if absorb is None:
                    alive.append(mem)
                    continue
                with self._lock:
                    self.conn.execute(
                        "UPDATE memories SET stability=?, access_count=access_count+?, "
                        "importance=? WHERE id=?",
                        (
                            max(absorb.stability, mem.stability) * 1.1,
                            mem.access_count,
                            max(absorb.importance, mem.importance),
                            absorb.id,
                        ),
                    )
                    self.conn.execute(
                        "UPDATE memories SET archived=1 WHERE id=?", (mem.id,)
                    )
                    self.conn.commit()
                merged += 1
        if merged:
            self._invalidate_index()
        return {"merged": merged}

    # ---- 轮次去重 -------------------------------------------------------
    def seen(self, session: str, digest: str) -> bool:
        with self._lock:
            row = self.conn.execute(
                "SELECT 1 FROM seen_turns WHERE session=? AND digest=?",
                (session, digest),
            ).fetchone()
        return row is not None

    def mark_seen(self, session: str, digest: str, now: float | None = None) -> None:
        now = _now() if now is None else now
        with self._lock:
            self.conn.execute(
                "INSERT OR IGNORE INTO seen_turns (session, digest, ts) VALUES (?,?,?)",
                (session, digest, now),
            )
            self.conn.commit()

    def prune_seen(self, keep: int = 4000) -> int:
        with self._lock:
            cur = self.conn.execute(
                "DELETE FROM seen_turns WHERE rowid NOT IN "
                "(SELECT rowid FROM seen_turns ORDER BY ts DESC LIMIT ?)",
                (keep,),
            )
            self.conn.commit()
        return cur.rowcount

    # ---- 迁移 -----------------------------------------------------------
    def export(self, session: str | None = None) -> str:
        data = [m.as_dict() for m in self.list(session, include_archived=True)]
        return json.dumps(data, ensure_ascii=False, indent=2)
