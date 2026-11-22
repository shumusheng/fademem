"""角色卡：读 SillyTavern 风格的 JSON，也接受极简写法。

兼容 `spec: "chara_card_v2"` 的字段（name / description / personality /
scenario / first_mes / mes_example），额外支持 fademem 自己的 `speech` 段，
用来约束"这个人怎么说话"。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Persona", "load_persona", "DEFAULT_PERSONA"]

DEFAULT_PERSONA: dict[str, Any] = {
    "name": "未命名",
    "description": "一个普通的、有点自己的脾气的人。",
    "personality": "不太热情，但也不冷漠；说话直接，偶尔嘴硬。",
    "scenario": "和对方在同一个空间里待着，随口聊天。",
    "relationship": "认识了一段时间，还没到无话不谈的地步。",
    "speech": {
        "address": "你",
        "sentence_length": [4, 24],
        "verbal_tics": [],
        "forbidden": [],
        "typos": False,
    },
    "boundaries": [],
    "first_mes": "",
    "mes_example": "",
}


@dataclass
class Speech:
    address: str = "你"
    sentence_length: tuple[int, int] = (4, 24)
    verbal_tics: list[str] = field(default_factory=list)
    forbidden: list[str] = field(default_factory=list)
    typos: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "Speech":
        data = data or {}
        raw_len = data.get("sentence_length") or [4, 24]
        try:
            low, high = int(raw_len[0]), int(raw_len[1])
        except (TypeError, ValueError, IndexError):
            low, high = 4, 24
        return cls(
            address=str(data.get("address") or "你"),
            sentence_length=(max(1, low), max(low + 1, high)),
            verbal_tics=[str(x) for x in (data.get("verbal_tics") or [])][:12],
            forbidden=[str(x) for x in (data.get("forbidden") or [])][:24],
            typos=bool(data.get("typos", False)),
        )


@dataclass
class Persona:
    name: str
    description: str = ""
    personality: str = ""
    scenario: str = ""
    relationship: str = ""
    speech: Speech = field(default_factory=Speech)
    boundaries: list[str] = field(default_factory=list)
    first_mes: str = ""
    mes_example: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Persona":
        # SillyTavern V2 把内容包在 data 里
        if isinstance(data.get("data"), dict):
            outer = {k: v for k, v in data.items() if k != "data"}
            merged = {**outer, **data["data"]}
        else:
            merged = dict(data)

        speech_raw = merged.get("speech")
        if speech_raw is None:
            speech_raw = {
                "address": merged.get("address") or "你",
                "verbal_tics": merged.get("verbal_tics") or [],
            }

        boundaries = merged.get("boundaries") or []
        if isinstance(boundaries, str):
            boundaries = [boundaries]

        return cls(
            name=str(merged.get("name") or "未命名"),
            description=str(merged.get("description") or ""),
            personality=str(merged.get("personality") or ""),
            scenario=str(merged.get("scenario") or ""),
            relationship=str(merged.get("relationship") or ""),
            speech=Speech.from_dict(speech_raw if isinstance(speech_raw, dict) else None),
            boundaries=[str(x) for x in boundaries],
            first_mes=str(merged.get("first_mes") or ""),
            mes_example=str(merged.get("mes_example") or ""),
            raw=merged,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "personality": self.personality,
            "scenario": self.scenario,
            "relationship": self.relationship,
            "speech": {
                "address": self.speech.address,
                "sentence_length": list(self.speech.sentence_length),
                "verbal_tics": self.speech.verbal_tics,
                "forbidden": self.speech.forbidden,
                "typos": self.speech.typos,
            },
            "boundaries": self.boundaries,
            "first_mes": self.first_mes,
            "mes_example": self.mes_example,
        }


def load_persona(path: str | None) -> Persona:
    """从文件读；文件不存在或解析失败时退回默认人设，不抛异常。"""
    if not path:
        return Persona.from_dict(DEFAULT_PERSONA)
    expanded = os.path.expanduser(path)
    if not os.path.exists(expanded):
        return Persona.from_dict(DEFAULT_PERSONA)
    try:
        with open(expanded, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("角色卡必须是一个 JSON 对象")
        return Persona.from_dict(data)
    except (OSError, ValueError, json.JSONDecodeError):
        return Persona.from_dict(DEFAULT_PERSONA)
