"""记忆检索：BM25 命中 × 遗忘曲线 × 重要度。

单靠关键词命中是不够的——人会突然想起一件跟当下无关、但很重要的事。
所以这里除了"相关记忆"，还会补一点**背景记忆**和偶发的**闪回**。
这三样加起来，角色才像"一直记得你"，而不是"每轮现查数据库"。
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from . import decay
from .store import Memory, Store

__all__ = ["Hit", "recall", "ScoringConfig"]


@dataclass
class ScoringConfig:
    alpha: float = 1.4          # 遗忘曲线的惩罚指数：越大越"健忘"
    beta: float = 0.9           # 重要度的奖励指数
    max_items: int = 8
    min_score: float = 0.02
    forget_below: float = 0.06  # 低于此值直接归档，不给模型看
    ambient_items: int = 2      # 补几条"没有直接关系但我记得"的背景记忆
    ambient_min_importance: float = 0.55
    flashback_rate: float = 0.05
    fuzz_below: float = 0.55    # 可提取度低于此值开始磨花细节
    reinforce_on_recall: bool = True


@dataclass
class Hit:
    memory: Memory
    score: float
    r: float
    state: str
    text: str
    spontaneous: bool = False

    def as_dict(self) -> dict:
        data = self.memory.as_dict()
        data.update(
            {
                "score": round(self.score, 4),
                "rendered": self.text,
                "spontaneous": self.spontaneous,
            }
        )
        return data


def recall(
    store: Store,
    session: str,
    query: str,
    cfg: ScoringConfig | None = None,
    now: float | None = None,
    rng: random.Random | None = None,
) -> list[Hit]:
    """按当前这句话，捞出这个"人"此刻会想起的东西。"""
    cfg = cfg or ScoringConfig()
    rng = rng or random.Random()
    import time

    now = time.time() if now is None else now

    candidates = store.list(session, include_archived=False, limit=5000)
    if not candidates:
        return []

    alive: list[tuple[Memory, float]] = []
    for mem in candidates:
        r = mem.retrievability(now)
        if r < cfg.forget_below:
            continue  # 已经想不起来了，交给 sweep 归档
        alive.append((mem, r))
    if not alive:
        return []

    by_id = {mem.id: (mem, r) for mem, r in alive}
    matches = store.search(session, query, limit=40, now=now)
    top = matches[0][1] if matches else 1.0

    scored: list[tuple[Memory, float, float]] = []
    for mem, bm in matches:
        if mem.id not in by_id:
            continue
        _, r = by_id[mem.id]
        rel = bm / top if top > 0 else 0.0
        score = (rel ** 1.2) * (r**cfg.alpha) * ((0.35 + mem.importance) ** cfg.beta)
        if score >= cfg.min_score:
            scored.append((mem, score, r))

    scored.sort(key=lambda x: x[1], reverse=True)

    # 背景记忆：跟这句话无关，但这个人本来就一直记着
    if cfg.ambient_items > 0:
        chosen = {m.id for m, _, _ in scored}
        spare = [
            (mem, r)
            for mem, r in alive
            if mem.id not in chosen and mem.importance >= cfg.ambient_min_importance
        ]
        spare.sort(key=lambda p: p[0].importance * (p[1] ** 0.5), reverse=True)
        for mem, r in spare[: cfg.ambient_items]:
            scored.append((mem, cfg.min_score * 0.6, r))

    hits: list[Hit] = []
    for mem, score, r in scored[: cfg.max_items]:
        state = decay.decay_state(r)
        hits.append(
            Hit(
                memory=mem,
                score=score,
                r=r,
                state=state,
                text=decay.fuzz_text(mem.text, r, seed=mem.id),
            )
        )

    # 闪回：一件久未想起、但分量不小的事，忽然冒出来
    if cfg.flashback_rate > 0 and rng.random() < cfg.flashback_rate:
        used = {h.memory.id for h in hits}
        pool = [
            (mem, r)
            for mem, r in alive
            if mem.id not in used and 0.10 <= r <= 0.48 and mem.importance >= 0.62
        ]
        if pool:
            mem, r = rng.choice(pool)
            hits.append(
                Hit(
                    memory=mem,
                    score=cfg.min_score * 0.5,
                    r=r,
                    state=decay.decay_state(r),
                    text=decay.fuzz_text(mem.text, r, seed=mem.id),
                    spontaneous=True,
                )
            )

    if cfg.reinforce_on_recall:
        # 想起来了，就会更牢一点。R 越低的记忆被想起时增幅越大（间隔效应）
        for hit in hits[:4]:
            store.reinforce(hit.memory.id, now)

    return hits
