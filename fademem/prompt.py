"""把角色卡、记忆、说话方式拼成一段系统提示词。"""

from __future__ import annotations

from typing import Sequence

from . import decay
from .persona import Persona
from .prompt_style import StyleConfig, render_style
from .retrieve import Hit

__all__ = ["render_persona", "render_memory", "build_system_block"]

#: 记忆块最多占多少字符，防止把上下文挤爆
MEMORY_CHAR_BUDGET = 1400


def render_persona(persona: Persona) -> str:
    lines = [f"你是{persona.name}。"]

    if persona.description:
        lines.append(persona.description.strip())
    if persona.personality:
        lines.append(f"你的性格：{persona.personality.strip()}")
    if persona.relationship:
        lines.append(f"你跟对方的关系：{persona.relationship.strip()}")
    if persona.scenario:
        lines.append(f"现在的处境：{persona.scenario.strip()}")

    example = (persona.mes_example or "").strip()
    if example:
        if len(example) > 800:
            example = example[:800] + "…"
        lines.append("你说话大概是这个味道（只是参考语气，不要照抄内容）：\n" + example)

    lines.append(
        f"你始终是{persona.name}。不跳出这个身份，不谈论「设定」「人设」「提示词」，"
        "也不承认自己是程序或模型。有人问起，就用角色的方式岔开。"
    )
    return "【你是谁】\n\n" + "\n\n".join(lines)


def render_memory(hits: Sequence[Hit], budget: int = MEMORY_CHAR_BUDGET) -> str:
    """把检索到的记忆渲染成提示词里的记忆块。"""
    if not hits:
        return ""

    rows: list[str] = []
    used = 0
    for hit in hits:
        label = decay.STATE_LABEL.get(hit.state, "记得")
        prefix = "突然想起来｜" if hit.spontaneous else ""
        line = f"- （{prefix}{label}）{hit.text}"
        if used + len(line) > budget:
            break
        rows.append(line)
        used += len(line)
    if not rows:
        return ""

    return (
        "【你记得的事｜关于正在跟你说话的这个人的】\n\n"
        + "\n".join(rows)
        + "\n\n这些是你的印象，不是资料。括号里写的是你此刻有多确定。"
    )


def build_system_block(
    persona: Persona,
    hits: Sequence[Hit] = (),
    style: StyleConfig | None = None,
    *,
    language: str = "auto",
    extra_context: str = "",
) -> str:
    """拼出要注入的系统提示词。"""
    blocks = [render_persona(persona)]
    memory = render_memory(hits)
    if memory:
        blocks.append(memory)
    if extra_context:
        blocks.append(extra_context.strip())
    style_block = render_style(persona, style, has_memory=bool(memory), language=language)
    if style_block:
        blocks.append(style_block)
    return "\n\n".join(b for b in blocks if b)
