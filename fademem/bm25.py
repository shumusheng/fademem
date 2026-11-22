"""纯 Python 的 BM25 检索 + 中英混排分词。零第三方依赖。

为什么不直接用向量检索：一是要跑 embedding 模型或调 API，二是对"角色扮演
里那三五条短记忆"这种小规模语料，BM25 的召回质量并不差，而延迟低一个数量级。
真需要语义召回时，把 :func:`search` 换掉即可，接口只要求返回一个分数列表。

中文分词策略：不引入 jieba 之类的词典。汉字连续串同时产出 **单字** 和
**相邻二元组（bigram）**，二元组承担了大部分语义区分度（"咖啡" vs "咖啡店"），
单字则保证召回下限。这是搜索引擎里常用的 n-gram 兜底做法，对短文本足够。
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Iterable, Sequence

__all__ = ["tokenize", "BM25", "CJK_STOPWORDS"]

# 汉字、日文假名、韩文音节
_CJK = r"\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af"
_TOKEN_RE = re.compile(rf"[{_CJK}]+|[a-z0-9][a-z0-9'_\-]*")

#: 高频虚词，单字形式丢弃（保留其参与的 bigram）
CJK_STOPWORDS = frozenset(
    "的 了 是 我 你 他 她 它 们 在 就 也 都 和 与 有 没 不 这 那 个 上 下 一 呀 吗 呢 吧 啊 嗯 哦 呗 么 着 过 被 把 让 给 又 还 很 太 更 最 会 能 要 想 说 对 好 后 前 里 外 时 再 才 只 但 而 或".split()
)

_SINGLE_KEEP = frozenset("猫 狗 雨 雪 钱 病 酒 烟 茶 书 药 哭 笑 死 家 妈 爸")


def tokenize(text: str) -> list[str]:
    """把一段文本切成检索用的 token 列表。"""
    if not text:
        return []
    norm = unicodedata.normalize("NFKC", str(text)).lower()
    tokens: list[str] = []
    for match in _TOKEN_RE.finditer(norm):
        chunk = match.group(0)
        if chunk[0].isascii():
            if len(chunk) > 1 or chunk.isdigit():
                tokens.append(chunk)
            continue
        # CJK 连续串
        if len(chunk) == 1:
            if chunk not in CJK_STOPWORDS or chunk in _SINGLE_KEEP:
                tokens.append(chunk)
            continue
        for i, ch in enumerate(chunk):
            if ch not in CJK_STOPWORDS or ch in _SINGLE_KEEP:
                tokens.append(ch)
            if i + 1 < len(chunk):
                tokens.append(chunk[i : i + 2])
    return tokens


class BM25:
    """经典 Okapi BM25。语料规模在万条以内时，每次查询重建索引也就几毫秒。"""

    def __init__(
        self,
        docs: Sequence[Sequence[str]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.n = len(docs)
        self.term_freqs: list[Counter[str]] = [Counter(d) for d in docs]
        self.doc_lens = [len(d) for d in docs]
        self.avg_len = (sum(self.doc_lens) / self.n) if self.n else 0.0

        df: Counter[str] = Counter()
        for tf in self.term_freqs:
            for term in tf:
                df[term] += 1
        self.idf = {
            term: math.log(1.0 + (self.n - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }

    def scores(self, query_tokens: Iterable[str]) -> list[float]:
        query = [t for t in query_tokens if t in self.idf]
        if not query or not self.n:
            return [0.0] * self.n
        avg = self.avg_len or 1.0
        out: list[float] = []
        for i in range(self.n):
            dl = self.doc_lens[i] or 1
            norm = self.k1 * (1.0 - self.b + self.b * dl / avg)
            total = 0.0
            tf = self.term_freqs[i]
            for term in query:
                freq = tf.get(term, 0)
                if not freq:
                    continue
                total += self.idf[term] * freq * (self.k1 + 1.0) / (freq + norm)
            out.append(total)
        return out


def build(tokens_per_doc: Sequence[Sequence[str]], **kwargs: float) -> BM25:
    """便捷构造。"""
    return BM25(tokens_per_doc, **kwargs)  # type: ignore[arg-type]
