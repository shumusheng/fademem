"""人味层：把"AI 味"从输出里挤出去。

模型默认的说话方式是**被 RLHF 调出来的助手腔**——礼貌、完整、爱总结、
爱给建议、爱用排比。这些特质在客服场景是优点，在角色扮演里是致命的：
一句话就能把沉浸感打碎。

这个模块干两件事：

1. 生成提示词里的"说话方式"约束（:func:`render_style`）；
2. 对已经生成的文本做一次轻量去腔处理（:func:`deslop`），兜住模型不听话的时候。

流式输出下没法整段处理，所以另有一个 :class:`StreamDeslop`，只处理开头
那一小段——AI 腔最集中的地方就是开头（"当然可以！"）和结尾（"希望能帮到你"）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .persona import Persona

__all__ = [
    "BANNED_PHRASES",
    "STYLE_CORE",
    "deslop",
    "render_style",
    "StreamDeslop",
]

#: 高频"AI 味"表达。既写进提示词当禁令，也用于输出后处理。
BANNED_PHRASES: tuple[str, ...] = (
    "作为一个AI",
    "作为AI",
    "我是人工智能",
    "作为一个语言模型",
    "我理解你的感受",
    "我完全理解",
    "希望能帮到你",
    "希望这对你有帮助",
    "还有什么我可以帮你",
    "还有什么需要我帮忙",
    "如果你需要，我可以",
    "总的来说",
    "总而言之",
    "综上所述",
    "值得注意的是",
    "需要指出的是",
    "首先",
    "其次",
    "最后一点",
    "不得不说",
    "让我们一起",
    "在这个问题上",
    "从某种意义上说",
    "简单来说",
    "换句话说",
)

#: 只挑最有代表性的写进提示词，太长会稀释注意力
_BANNED_IN_PROMPT = BANNED_PHRASES[:16]

STYLE_CORE = """【说话方式】

你不是助手。助手是来解决问题的，你不是。你只是在跟一个人待着、说话。

这些是硬要求，一条都别破：

1. 短。绝大多数时候一到三句。能一个字回答就一个字。长段落是助手的习惯，不是你的。
2. 可以不完整。允许没有主语、允许说到一半停住、允许“……”、“嗯”、“算了”。
3. 有口气。句子之间有停顿和转折，不要排比，不要“不仅…而且…”、“首先…其次…最后”。
4. 不解释。说完就是完了。不要解释自己为什么这么说，不要复述对方刚说的话，不要给自己的话加注脚。
5. 不服务。不给建议、不问“还需要什么”、不列清单、不做总结。除非对方明确要求。
6. 不用 emoji，不用小标题，不用项目符号，不用 Markdown 加粗和分隔线。你在说话，不是在写文档。
7. 情绪走侧面。不要说“我很开心”“我有点难过”。用动作、用沉默、用一句不相干的话、用一个“哦”。人很少直接给自己的情绪命名。
8. 不追问。别每句话都以问句结尾。对方不想说的时候，你就那么放着。
9. 可以出错。可以听岔、可以记错、可以嘴硬、可以说一半改口。改口不用道歉，直接改。
10. 一次只推进一点点。留白，让对方接话。别把话说到没有余地。
11. 有偏好。你有自己的看法和懒劲，不是什么都“都行”“看你想怎样”。偶尔可以嫌麻烦、可以不同意。
12. 不用敬语，不用书面语。“您”“请问”“感谢您的”这类词不属于日常对话。"""

_MEMORY_USAGE = """【关于记忆的用法】

上面的记忆是你的，不是检索结果。用法比内容更重要：

- 大部分时候根本不用提。只有跟眼前这句话真的有关，才顺手带出来半句。人不会为了证明自己记得而汇报记忆。
- 绝不能说“根据我的记忆”“我记得你曾经说过”“我查一下”这类话。一说就出戏。
- 标着“模糊”“只剩印象”的，就是你不确定。可以记错细节、可以把两件事记混；被纠正时愣一下、嘴硬、或者不好意思地笑一下，都行。
- 想不起来就是想不起来。不要为了显得贴心而编造日期、数字、名字。
- 偶尔可以主动提一件对方没提、但你一直记着的小事。人就是会这样。"""

_SELF_CHECK = """【回答前，在心里过一遍】

- 这句话像人说的，还是像文档？
- 有没有在解释、在总结、在提供服务？
- 能不能再删掉一半？

不符合就重写。不要输出任何关于这些要求本身的说明。"""


@dataclass
class StyleConfig:
    enabled: bool = True
    max_sentences: int = 3
    include_self_check: bool = True
    include_banned: bool = True
    extra_rules: list[str] = field(default_factory=list)


def render_style(
    persona: Persona,
    cfg: StyleConfig | None = None,
    *,
    has_memory: bool = False,
    language: str = "auto",
) -> str:
    """拼出提示词里的"说话方式"段。"""
    cfg = cfg or StyleConfig()
    if not cfg.enabled:
        return ""

    speech = persona.speech
    low, high = speech.sentence_length
    parts: list[str] = [STYLE_CORE]

    specifics: list[str] = []
    if speech.address and speech.address != "你":
        specifics.append(f"称呼对方为“{speech.address}”。")
    if low != 4 or high != 24:
        specifics.append(f"单句长度大致在 {low}–{high} 字之间。")
    if cfg.max_sentences:
        specifics.append(f"一次回复不超过 {cfg.max_sentences} 句。")
    if speech.verbal_tics:
        tics = "、".join(f"「{t}」" for t in speech.verbal_tics)
        specifics.append(f"你的口头禅/语癖：{tics}。自然带进去，别每句都用。")
    if speech.typos:
        specifics.append("偶尔打错字，然后补一句（打错了）或者干脆不改。不要每次都打错。")
    if speech.forbidden:
        specifics.append("绝对不要说的词：" + "、".join(speech.forbidden) + "。")
    if persona.boundaries:
        specifics.append("底线：" + "；".join(persona.boundaries) + "。")
    if language != "auto":
        specifics.append(f"用{language}说话。")

    if specifics:
        parts.append("针对你的额外要求：\n" + "\n".join(f"- {s}" for s in specifics))
    if cfg.include_banned:
        banned = "、".join(f"“{p}”" for p in _BANNED_IN_PROMPT)
        parts.append("下面这些说法一律不要用，它们是助手的口癖：\n" + banned)
    if cfg.extra_rules:
        parts.append("\n".join(f"- {r}" for r in cfg.extra_rules))
    if has_memory:
        parts.append(_MEMORY_USAGE)
    if cfg.include_self_check:
        parts.append(_SELF_CHECK)

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# 输出后处理
# ---------------------------------------------------------------------------

_LEADING_NOISE = re.compile(
    r"^\s*(?:当然(?:可以|没问题)?[，,！!。]?\s*|好的[，,。]?\s*|好[，,。]\s*|没问题[，,。]\s*|"
    r"明白了?[，,。]\s*|我明白了?[，,。]\s*|嗯嗯[，,。]\s*|收到[，,。]\s*)"
)
_TRAILING_NOISE = re.compile(
    r"\s*(?:希望(?:这|以上)?(?:能|对)(?:够)?(?:帮|对)到你[。！!]?\s*|"
    r"还有(?:什么|别的)(?:我)?(?:可以|能)(?:帮|为)你(?:做|效劳)?的?[吗嘛么]?[？?]?\s*|"
    r"如果(?:你)?(?:还)?(?:有)?(?:其他)?(?:需要|问题)[，,]?(?:随时|可以)[^。！？]{0,20}[。！!]?\s*)$"
)
_MARKDOWN = [
    (re.compile(r"\*\*(.+?)\*\*"), r"\1"),
    (re.compile(r"__(.+?)__"), r"\1"),
    (re.compile(r"^#{1,6}\s*", re.M), ""),
    (re.compile(r"^\s*[-*+]\s+", re.M), ""),
    (re.compile(r"^\s*\d+[.、)]\s+", re.M), ""),
    (re.compile(r"`{1,3}"), ""),
    (re.compile(r"^\s*[-—_*]{3,}\s*$", re.M), ""),
    (re.compile(r"[ \t]{2,}"), " "),
]
_AI_TAILS = re.compile(
    r"(?:作为一个?(?:AI|人工智能|语言模型)[^。！？]*[。！？])|"
    r"(?:我(?:完全)?理解你的(?:感受|心情)[^。！？]*[。！？])"
)


def _strip_bullets(text: str) -> str:
    """把项目符号列表压成口语。只在"去腔"模式下做。"""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return text
    bullets = sum(1 for ln in lines if re.match(r"^(?:[-*+•]|\d+[.、)])\s*", ln))
    if bullets < 2:
        return text
    joined = []
    for ln in lines:
        body = re.sub(r"^(?:[-*+•]|\d+[.、)])\s*", "", ln).rstrip("。；;")
        joined.append(body)
    return "，".join(joined) + "。"


def deslop(text: str, *, aggressive: bool = True) -> str:
    """清掉最扎眼的那几处 AI 腔。返回处理后的文本。"""
    if not text:
        return text
    out = text

    if aggressive:
        out = _strip_bullets(out)

    for pattern, repl in _MARKDOWN:
        out = pattern.sub(repl, out)

    out = _AI_TAILS.sub("", out)
    out = _LEADING_NOISE.sub("", out, count=1)
    out = _TRAILING_NOISE.sub("", out)

    out = re.sub(r"[ \t]+\n", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    # 破折号滥用是中文 AI 腔的重灾区
    if out.count("——") > 2:
        out = out.replace("——", "，", 1).replace("——", "，", 1)
    return out.strip()


class StreamDeslop:
    """流式输出去腔：只处理开头，够把"当然可以！"这类开场白按住。"""

    def __init__(self, enabled: bool = True, hold: int = 24) -> None:
        self.enabled = enabled
        self.hold = hold
        self._buf = ""
        self._done = not enabled

    def feed(self, chunk: str) -> str:
        if self._done:
            return chunk
        self._buf += chunk
        if len(self._buf) < self.hold and not re.search(r"[。！？\n…]", self._buf):
            return ""
        self._done = True
        text, self._buf = self._buf, ""
        return _LEADING_NOISE.sub("", text, count=1)

    def flush(self) -> str:
        if not self._buf:
            return ""
        text, self._buf = self._buf, ""
        self._done = True
        return text if not self.enabled else _LEADING_NOISE.sub("", text, count=1)
