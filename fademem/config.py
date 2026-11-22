"""配置：JSON 文件 + 默认值深合并。"""

from __future__ import annotations

import copy
import json
import os
from typing import Any

__all__ = ["DEFAULT_CONFIG", "load_config", "write_default_config", "HOME_DIR", "default_paths"]

HOME_DIR = os.path.join(os.path.expanduser("~"), ".fademem")

DEFAULT_CONFIG: dict[str, Any] = {
    "server": {
        "host": "127.0.0.1",
        "port": 8770,
        "verbose": False,
    },
    "upstream": {
        # 指向任何 OpenAI 兼容服务；末尾要带 /v1
        "base_url": "https://api.openai.com/v1",
        "api_key": "",
        # true 时优先用客户端自己带的 Authorization，方便一个服务接多家
        "passthrough_auth": True,
        "timeout": 180,
        "extra_headers": {},
    },
    "memory": {
        "db": os.path.join(HOME_DIR, "memory.db"),
        "base_stability_days": 6.0,
        "alpha": 1.4,
        "beta": 0.9,
        "max_items": 8,
        "min_score": 0.02,
        "forget_below": 0.06,
        "ambient_items": 2,
        "ambient_min_importance": 0.55,
        "flashback_rate": 0.05,
        "fuzz_below": 0.55,
        "reinforce_on_recall": True,
        "sweep_every": 50,
    },
    "extraction": {
        "mode": "heuristic",   # heuristic | llm | off
        "allow_generic": True,
        "max_items": 6,
    },
    "persona": os.path.join(HOME_DIR, "persona.json"),
    "style": {
        "enabled": True,
        "max_sentences": 3,
        "include_self_check": True,
        "include_banned": True,
        "deslop": True,
        "deslop_stream": True,
        "extra_rules": [],
    },
    "prompt": {
        # 合并进客户端已有的 system 消息，而不是新增一条（兼容更严格的服务商）
        "merge_system": True,
        "language": "auto",
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | None = None) -> dict[str, Any]:
    """读配置。文件不存在就用默认值，不报错。"""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    target = path or os.path.join(HOME_DIR, "config.json")
    target = os.path.expanduser(target)
    if os.path.exists(target):
        try:
            with open(target, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                cfg = _deep_merge(cfg, data)
        except (OSError, json.JSONDecodeError) as exc:
            raise SystemExit(f"配置文件解析失败：{target}\n{exc}") from exc
    cfg["_config_path"] = target
    return cfg


def default_paths() -> dict[str, str]:
    return {
        "home": HOME_DIR,
        "config": os.path.join(HOME_DIR, "config.json"),
        "persona": os.path.join(HOME_DIR, "persona.json"),
        "db": os.path.join(HOME_DIR, "memory.db"),
    }


def write_default_config(path: str, overwrite: bool = False) -> str:
    path = os.path.expanduser(path)
    if os.path.exists(path) and not overwrite:
        return path
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {k: v for k, v in DEFAULT_CONFIG.items() if not k.startswith("_")}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    return path
