"""记忆抽取：把一句人话拆成可长期保存的事实。

默认走**纯规则**，不调用任何模型。理由很实际：如果每轮对话都额外发一次
抽取请求，延迟和费用都会翻倍，而角色扮演里真正值得记的东西其实高度集中在
少数句式上——自我介绍、喜好、关系、习惯、最近发生的事。

需要更强抽取能力时可以换成 LLM 模式（见 :func:`llm_extract`），接口一致。

每条记忆存的都是**谓语短语**（"住在城西"、"喜欢黑咖啡"），不带人称代词。
这样渲染进提示词时不会出现"他/她/你"打架的问题。
"""

from __future__ import annotations

import re
from typing import Callable, Iterable, Sequence

from .store import Candidate

__all__ = ["extract", "redact", "EMOTION_WORDS", "llm_extract"]

#: 情绪词，用于计算记忆的"显著度"——带情绪的经历天生记得牢
EMOTION_WORDS = (
    "爱", "喜欢", "讨厌", "恨", "怕", "害怕", "恐惧", "难过", "伤心", "开心", "高兴",
    "委屈", "生气", "愤怒", "羞", "尴尬", "哭", "笑", "孤独", "焦虑", "压力", "心疼",
    "心动", "失望", "期待", "紧张", "崩溃", "感动", "后悔", "想", "念", "恶心", "安心",
    "睡不着", "失眠", "生病", "发烧", "疼", "累", "烦",
)

_INTENSIFIERS = (
    "特别", "非常", "超级", "最", "永远", "第一次", "从不", "再也不", "一直", "真的很",
    "巨", "贼", "死", "极",
)

_HEDGE = re.compile(r"^(?:其实|老实说|说实话|怎么说呢|嗯+|呃+|那个)[，,]?\s*")


class _Rule:
    __slots__ = ("kind", "pattern", "fmt", "importance", "needles")

    def __init__(
        self,
        kind: str,
        pattern: str,
        fmt: str,
        importance: float,
        needles: tuple[str, ...] | None = None,
    ):
        self.kind = kind
        self.pattern = re.compile(pattern)
        self.fmt = fmt
        self.importance = importance
        self.needles = needles


# 顺序有讲究：具体的先说，泛化的兜底放最后
_RULES: tuple[_Rule, ...] = (
    _Rule(
        "identity",
        r"(?:我叫|我的名字(?:是|叫)?|你可以叫我|就叫我|叫我|大家都叫我)\s*(?P<x>[A-Za-z\u4e00-\u9fff·•]{1,8})",
        "叫{x}",
        0.92,
    ),
    _Rule("identity", r"我(?:今年)?(?P<x>\d{1,2})\s*岁", "{x}岁", 0.8),
    _Rule(
        "identity",
        r"我(?:的)?生日(?:是|在)?\s*(?P<x>[^\s，。！？,!?]{2,14})",
        "生日是{x}",
        0.7,
    ),
    _Rule(
        "location",
        r"我(?:现在|最近)?(?:住在|家在|定居在|搬到了|搬到|回了)\s*(?P<x>[^\s，。！？,!?]{1,15})",
        "住在{x}",
        0.66,
    ),
    _Rule(
        "occupation",
        r"我(?:在|于)\s*(?P<x>[^\s，。！？,!?]{2,16}?)\s*(?:上班|工作|实习|打工)",
        "在{x}上班",
        0.72,
    ),
    _Rule(
        "occupation",
        r"我(?:在|于)\s*(?P<x>[^\s，。！？,!?]{2,16}?)\s*(?:上学|读书|念书|读研|读博)",
        "在{x}读书",
        0.72,
    ),
    _Rule(
        "occupation",
        r"我(?:的)?(?:工作|职业)(?:是|就是)\s*(?P<x>[^\s，。！？,!?]{1,16})",
        "工作是{x}",
        0.72,
    ),
    _Rule(
        "relation",
        r"我(?:的)?(?P<x>(?:女朋友|男朋友|老婆|老公|妈妈|爸爸|妈|爸|弟弟|妹妹|哥哥|姐姐|"
        r"儿子|女儿|室友|老板|同事|上司|前任|朋友|闺蜜|兄弟|师父|师傅))",
        "有个{x}",
        0.74,
    ),
    _Rule(
        "pet",
        r"我(?:养了|养着|养过|有一只|有只)\s*(?:一只|只|条|个|头)?\s*"
        r"(?P<x>[^\s，。！？,!?]{1,14})",
        "养了{x}",
        0.62,
        needles=("猫", "狗", "仓鼠", "兔子", "鹦鹉", "乌龟", "鱼", "蛇", "蜥蜴", "鸟"),
    ),
    _Rule(
        "promise",
        r"我(?:答应|保证|承诺|说好)(?:了|过)?(?:你)?\s*(?P<x>[^。！？!?]{2,30})",
        "答应{x}",
        0.56,
    ),
    _Rule(
        "preference",
        r"我(?:很|最|特别|超|挺|蛮|真的|有点)?(?:喜欢|爱|迷上了|迷上|沉迷|上瘾|偏爱)"
        r"\s*(?P<x>[^\s。！？!?]{1,24})",
        "喜欢{x}",
        0.6,
    ),
    _Rule(
        "dislike",
        r"我(?:很|最|特别|超|挺|真的|有点)?(?:不喜欢|讨厌|受不了|反感|嫌弃|吃不消)"
        r"\s*(?P<x>[^\s。！？!?]{1,24})",
        "讨厌{x}",
        0.64,
    ),
    _Rule(
        "fear",
        r"我(?:很|最|特别|有点|超)?(?:害怕|怕|恐惧|不敢)\s*(?P<x>[^\s。！？!?]{1,24})",
        "害怕{x}",
        0.72,
    ),
    _Rule(
        "habit",
        r"我(?:每天|每晚|每周|每月|经常|总是|习惯|一般|通常)\s*(?P<x>[^\s。！？!?]{2,26})",
        "{x}",
        0.52,
    ),
    _Rule(
        "event",
        r"(?P<t>今天|昨天|前天|大前天|上周|上个月|去年|前几天|刚才)\s*[，,]?\s*"
        r"(?P<x>[^。！？!?]{4,46})",
        "{t}，{x}",
        0.46,
    ),
    _Rule(
        "boundary",
        r"我(?:不喜欢别人|讨厌别人|别|不要)\s*(?P<x>[^\s。！？!?]{1,24})",
        "不喜欢别人{x}",
        0.68,
    ),
)


def _norm(text: str) -> str:
    """归一化，只用于去重比对。"""
    t = re.sub(r"[\s，。！？、,.!?~…—\-·]+", "", text or "")
    for w in ("我的", "我", "了", "的", "啊", "呀", "呢", "吧", "挺", "很", "特别", "超"):
        t = t.replace(w, "")
    return t.lower()


#: 不该被写进长期记忆的敏感信息
_PHONE = re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)")
_IDCARD = re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)")
_BANK = re.compile(r"(?<!\d)\d{16,19}(?!\d)")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
_SECRET = re.compile(r"(?:密码|口令|验证码|password|passwd|token|密钥)\s*(?:是|为|[:：=])?\s*\S{2,20}", re.I)


def redact(text: str) -> str:
    """把手机号、身份证、银行卡、邮箱、口令替换掉。记忆里不该有这些东西。"""
    if not text:
        return ""
    out = _SECRET.sub("［已隐去］", text)
    out = _IDCARD.sub("［证件号］", out)
    out = _BANK.sub("［长数字］", out)
    out = _PHONE.sub("［手机号］", out)
    out = _EMAIL.sub("［邮箱］", out)
    return out


def _clean(text: str) -> str:
    t = _HEDGE.sub("", (text or "").strip())
    t = re.sub(r"\s+", " ", t)
    return t.strip(" ，,。.、；;~…—-")


def _emotion_score(text: str) -> tuple[int, bool]:
    hits = sum(1 for w in EMOTION_WORDS if w in text)
    intense = any(w in text for w in _INTENSIFIERS)
    return hits, intense


def extract(
    message: str,
    *,
    role: str = "user",
    max_items: int = 6,
    allow_generic: bool = True,
) -> list[Candidate]:
    """从一条消息里抽出记忆候选。"""
    raw = _clean(message)
    if not raw or len(raw) < 3:
        return []
    # 纯提问通常不含可记的事实；但"我该怎么办"这种也不该记
    text = redact(raw)

    emo_hits, intense = _emotion_score(text)
    salience = min(1.0, emo_hits * 0.22 + (0.22 if intense else 0.0))
    boost = min(0.28, emo_hits * 0.055) + (0.07 if intense else 0.0)
    length_boost = min(0.08, len(text) / 600.0)

    out: list[Candidate] = []
    seen_keys: set[tuple[str, str]] = set()

    for rule in _RULES:
        for match in rule.pattern.finditer(text):
            parts = match.groupdict()
            value = (parts.get("x") or "").strip()
            if not value or len(value) < 1:
                continue
            if rule.needles and not any(n in value for n in rule.needles):
                continue
            if rule.kind == "event":
                # 事件规则容易把整句话吞掉，限制一下
                if len(value) < 4:
                    continue
            try:
                phrase = rule.fmt.format(**{k: _clean(v or "") for k, v in parts.items()})
            except (KeyError, IndexError):
                continue
            phrase = _clean(phrase)
            if len(phrase) < 2:
                continue
            key = (rule.kind, _norm(value))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            imp = min(0.99, rule.importance + boost + length_boost)
            if role != "user":
                imp *= 0.6  # 角色自己说的话，记忆权重低一档
            out.append(
                Candidate(
                    kind=rule.kind,
                    text=phrase,
                    subject=_norm(value),
                    importance=round(imp, 3),
                    salience=round(salience, 3),
                    source=role,
                )
            )
            if len(out) >= max_items:
                return out

    if not out and allow_generic and role == "user":
        # 兜底：带"我"的陈述句，权重给低，靠遗忘曲线自然淘汰
        if "我" in text and 5 <= len(text) <= 90 and not text.endswith(("？", "?")):
            if not re.match(r"^(?:我(?:觉得|认为|在想|想问|是不是)|你)", text):
                out.append(
                    Candidate(
                        kind="fact",
                        text=_clean(text)[:80],
                        subject=_norm(text)[:40],
                        importance=round(min(0.4, 0.22 + boost), 3),
                        salience=round(salience, 3),
                        source=role,
                    )
                )

    return out


_EXTRACT_SYSTEM = (
    "你是一个记忆抽取器。阅读对话，抽出**关于用户**的、值得长期记住的事实。\n"
    "只输出 JSON 数组，不要任何解释。每个元素形如：\n"
    '{"kind": "identity|preference|dislike|fear|habit|event|relation|occupation|location|boundary|fact", '
    '"text": "谓语短语，不带人称代词，如『住在城西』", "importance": 0.0-1.0}\n'
    "宁缺勿滥。寒暄、提问、情绪宣泄不要抽。最多 5 条。"
)


def llm_extract(
    message: str,
    call: Callable[[str, str], str],
    *,
    max_items: int = 5,
) -> list[Candidate]:
    """用模型做抽取。``call(system, user) -> str`` 由调用方提供（复用上游配置）。"""
    import json

    try:
        raw = call(_EXTRACT_SYSTEM, redact(message))
        start, end = raw.find("["), raw.rfind("]")
        if start < 0 or end <= start:
            return []
        data = json.loads(raw[start : end + 1])
    except Exception:  # noqa: BLE001 - 抽取失败不能影响主流程
        return []

    out: list[Candidate] = []
    for item in data[:max_items]:
        if not isinstance(item, dict):
            continue
        text = _clean(str(item.get("text") or ""))
        if len(text) < 2:
            continue
        try:
            imp = float(item.get("importance", 0.5))
        except (TypeError, ValueError):
            imp = 0.5
        out.append(
            Candidate(
                kind=str(item.get("kind") or "fact")[:24],
                text=text,
                subject=_norm(text)[:40],
                importance=max(0.05, min(0.99, imp)),
                salience=min(1.0, _emotion_score(message)[0] * 0.22),
                source="llm",
            )
        )
    return out
