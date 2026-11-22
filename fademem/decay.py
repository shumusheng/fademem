"""艾宾浩斯遗忘曲线：记忆稳定性、可提取度、复习增益、细节模糊化。

核心模型（指数遗忘）::

    R(t) = exp(-Δt / S)

``R`` 是可提取度（retrievability，0~1），``Δt`` 是距上次强化经过的天数，
``S`` 是记忆稳定性（stability，单位天）。``S`` 越大，忘得越慢。

复习的间隔效应：在 ``R`` 已经掉得比较低的时候复习，稳定性增幅最大。这里用

    S' = S * (1 + k * (1 - R)^1.5)

近似。刚想起就立刻再想一遍几乎不会加深印象，隔一阵子再想起来才记得牢。
这跟人的实际体验一致，也是"背单词要拉开间隔"的数学解释。

新记忆的初始稳定性由重要度和情绪显著度决定：

    S0 = base * (0.35 + importance)^γ * (1 + salience_gain * salience)

情绪越强的经历一开始就记得越牢，这是 Ebbinghaus 之后一百多年的共识。
"""

from __future__ import annotations

import math
import random
import re

SECONDS_PER_DAY = 86400.0

#: 记忆状态，从清晰到彻底忘掉
STATE_ORDER = ("vivid", "clear", "hazy", "faint", "gone")

#: 每种状态的渲染前缀，供提示词使用
STATE_LABEL = {
    "vivid": "很清楚",
    "clear": "记得",
    "hazy": "模糊地记得",
    "faint": "只剩一点印象",
    "gone": "想不起来了",
}

__all__ = [
    "SECONDS_PER_DAY",
    "STATE_ORDER",
    "STATE_LABEL",
    "days_between",
    "retrievability",
    "initial_stability",
    "half_life",
    "reinforce",
    "decay_state",
    "fuzz_text",
]


def days_between(now: float, then: float) -> float:
    """两个时间戳之间相差多少天（负数按 0 处理）。"""
    return max(0.0, (now - then) / SECONDS_PER_DAY)


def retrievability(elapsed_days: float, stability_days: float) -> float:
    """当前可提取度 R ∈ [0, 1]。"""
    if stability_days <= 1e-9:
        return 0.0
    if elapsed_days <= 0:
        return 1.0
    try:
        return math.exp(-elapsed_days / stability_days)
    except OverflowError:  # pragma: no cover - 极端参数保护
        return 0.0


def initial_stability(
    importance: float,
    salience: float = 0.0,
    base: float = 6.0,
    gamma: float = 1.9,
    salience_gain: float = 2.0,
) -> float:
    """新记忆的初始稳定性（天）。"""
    imp = min(1.0, max(0.0, float(importance)))
    sal = min(1.0, max(0.0, float(salience)))
    return base * (0.35 + imp) ** gamma * (1.0 + salience_gain * sal)


def half_life(stability_days: float) -> float:
    """稳定度为 S 时，记忆衰减到一半需要多少天。"""
    return max(0.0, stability_days) * math.log(2.0)


def reinforce(stability: float, r_now: float, k: float = 1.5, cap: float = 3650.0) -> float:
    """复习后的新稳定性。``r_now`` 越低（越模糊），增幅越大。"""
    r = min(1.0, max(0.0, float(r_now)))
    current = max(0.05, float(stability))
    return min(cap, current * (1.0 + k * (1.0 - r) ** 1.5))


def decay_state(r: float) -> str:
    """把可提取度映射成离散的记忆状态。"""
    if r >= 0.75:
        return "vivid"
    if r >= 0.50:
        return "clear"
    if r >= 0.25:
        return "hazy"
    if r >= 0.08:
        return "faint"
    return "gone"


_PAREN = re.compile(r"[（(][^）)]{1,60}[）)]")
_NUM = re.compile(r"\d+")
_YEAR = re.compile(r"\d{4}\s*年")
_MONTH = re.compile(r"\d{1,2}\s*月")
_DAY = re.compile(r"\d{1,2}\s*[日号]")
_QUOTED = re.compile(r"[“\"「『][^”\"」』]{1,40}[”\"」』]")

_VAGUE_COUNT = ("几个", "好些", "不少", "好几")
_VAGUE_YEAR = ("有一年", "那年", "以前", "有些年")
_VAGUE_MONTH = ("有个月", "某个月", "那阵子")
_VAGUE_DAY = ("某天", "有一天", "那天")
_CLAUSE_BREAKS = ("，", ",", "；", ";", "、")


def _maybe(rng: random.Random, p: float, keep: str, alternatives: tuple[str, ...]) -> str:
    """按概率 p 把具体信息换成模糊说法。p 越大越模糊。"""
    return rng.choice(alternatives) if rng.random() < p else keep


def fuzz_text(text: str, r: float, seed: object = 0) -> str:
    """按可提取度把一段记忆的文字"磨花"。

    人会遗忘细节，但不会遗忘主干。所以这里保留骨架，只动细节：
    括号里的补充、具体年月日、数字、引用的原话。同样的 ``seed`` 与 ``r`` 会
    得到同样的结果，方便复现和测试。
    """
    if not text:
        return ""
    if r >= 0.55:
        return text

    loss = min(1.0, (0.55 - max(0.0, r)) / 0.55)
    rng = random.Random(f"{seed}|{r:.3f}")
    # 越模糊，细节丢失越接近必然
    p = min(1.0, loss * 1.6)

    out = _PAREN.sub("", text)
    out = _YEAR.sub(lambda m: _maybe(rng, p, m.group(0), _VAGUE_YEAR), out)
    out = _MONTH.sub(lambda m: _maybe(rng, p, m.group(0), _VAGUE_MONTH), out)
    out = _DAY.sub(lambda m: _maybe(rng, p, m.group(0), _VAGUE_DAY), out)
    out = _NUM.sub(lambda m: _maybe(rng, p, m.group(0), _VAGUE_COUNT), out)
    out = _QUOTED.sub(lambda m: _maybe(rng, p, m.group(0), ("说了句什么来着",)), out)

    out = re.sub(r"[，,、]\s*[，,、]", "，", out).strip()
    out = re.sub(r"\s{2,}", " ", out)

    # 掉得太厉害时，主干也保不住了，只剩一个模糊的印象。
    # 切在分句边界上，读起来才像人说话，而不是像被截断的字符串。
    if loss > 0.55 and len(out) > 16:
        target = max(8, int(len(out) * (1.0 - loss * 0.5)))
        window = out[:target]
        cut = max((window.rfind(ch) for ch in _CLAUSE_BREAKS), default=-1)
        if cut > 6:
            out = window[:cut] + "，后面记不清了"
        else:
            out = window.rstrip("，,、；; ") + "……"

    return out
