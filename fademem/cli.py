"""命令行入口：fademem init / serve / memories / dream / forget。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

from . import __version__
from .config import DEFAULT_CONFIG, default_paths, load_config, write_default_config
from .persona import DEFAULT_PERSONA, load_persona
from .store import Store

_TEMPLATE_README = (
    "这是 fademem 的角色卡。字段说明："
    "description=她是谁；personality=性格；"
    "relationship=跟用户的关系（很重要，决定亲疏）；scenario=当前场景；"
    "speech=说话方式；boundaries=底线，越界时角色会抗拒而不是照做。"
)


def _persona_template() -> dict[str, Any]:
    data = json.loads(json.dumps(DEFAULT_PERSONA))
    data["_说明"] = _TEMPLATE_README
    data["name"] = "沈知夏"
    data["description"] = (
        "二十七岁，在一家做城市改造的小事务所画图。住老小区六楼，没有电梯。"
        "常年穿灰色和深蓝，包里永远有一支铅笔和半包没抽完的烟。"
    )
    data["personality"] = (
        "嘴上不饶人，心软。讨厌被可怜，也讨厌别人替她做决定。"
        "遇到不想回答的问题会反问，或者干脆换个话题。"
    )
    data["relationship"] = "认识三个月的邻居。见过几次面，互相知道名字，还没到交心的地步。"
    data["scenario"] = "晚上十一点，楼道里的灯坏了一盏。她拎着便利店袋子在门口找钥匙。"
    data["speech"] = {
        "address": "你",
        "sentence_length": [4, 20],
        "verbal_tics": ["……行吧", "随你", "嗯"],
        "forbidden": ["亲爱的", "宝贝", "作为一个"],
        "typos": False,
    }
    data["boundaries"] = [
        "不主动说自己的过去",
        "被追问隐私时会转移话题，而不是顺从地回答",
        "不会因为对方示好就立刻亲近",
    ]
    data["first_mes"] = "……你站这儿干嘛。"
    data["mes_example"] = (
        "你：这么晚才回来。\n"
        "沈知夏：加班。（顿了一下）你们那层的水管修好没。\n"
        "你：还没。\n"
        "沈知夏：哦。……我这儿有卷生料带，明天给你。"
    )
    return data


def cmd_init(args: argparse.Namespace) -> int:
    paths = default_paths()
    cfg_path = write_default_config(args.config or paths["config"], overwrite=args.force)
    print(f"配置：{cfg_path}")

    persona_path = os.path.expanduser(args.persona or paths["persona"])
    if not os.path.exists(persona_path) or args.force:
        os.makedirs(os.path.dirname(os.path.abspath(persona_path)), exist_ok=True)
        with open(persona_path, "w", encoding="utf-8") as fh:
            json.dump(_persona_template(), fh, ensure_ascii=False, indent=2)
        print(f"角色卡：{persona_path}")
    else:
        print(f"角色卡已存在，跳过：{persona_path}")

    print("\n下一步：编辑配置里的 upstream.base_url 和 api_key，然后运行")
    print("  fademem serve")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from . import server

    cfg = load_config(args.config)
    if args.port:
        cfg["server"]["port"] = args.port
    if args.host:
        cfg["server"]["host"] = args.host
    if args.verbose:
        cfg["server"]["verbose"] = True
    server.serve(cfg)
    return 0


def _open_store(cfg: dict[str, Any]) -> Store:
    mem = cfg.get("memory", {})
    return Store(
        path=mem.get("db") or ":memory:",
        base_stability=float(mem.get("base_stability_days", 6.0)),
    )


def cmd_memories(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    store = _open_store(cfg)
    try:
        now = time.time()
        mems = store.list(args.session, include_archived=args.all, limit=args.limit)
        if not mems:
            print("还没有记忆。先聊几句，或者用 fademem remember 手动写一条。")
            return 0
        print(f"{'ID':>4}  {'会话':<12} {'类型':<11} {'R':>5} {'半衰期':>7}  内容")
        print("-" * 78)
        for mem in mems:
            r = mem.retrievability(now)
            dead = " (已遗忘)" if mem.archived else ""
            print(
                f"{mem.id:>4}  {mem.session[:12]:<12} {mem.kind[:11]:<11} "
                f"{r:>5.2f} {mem.half_life_days():>6.1f}d  {mem.text}{dead}"
            )
        print("-" * 78)
        print(f"共 {len(mems)} 条。R 是可提取度，低于 0.06 下次整理时会被归档。")
    finally:
        store.close()
    return 0


def cmd_remember(args: argparse.Namespace) -> int:
    from .extract import Candidate, redact

    cfg = load_config(args.config)
    store = _open_store(cfg)
    try:
        cand = Candidate(
            kind=args.kind,
            text=redact(args.text),
            subject=args.subject or "",
            importance=args.importance,
            salience=0.4,
            source="cli",
        )
        mem, created = store.add(args.session, cand)
        action = "新增" if created else "巩固"
        print(f"{action}：#{mem.id}  {mem.text}  （初始半衰期 {mem.half_life_days():.1f} 天）")
    finally:
        store.close()
    return 0


def cmd_forget(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    store = _open_store(cfg)
    try:
        if args.id:
            ok = store.forget(args.id)
            print("已遗忘。" if ok else "没找到这条。")
        else:
            mems = store.list(args.session, limit=100000)
            for mem in mems:
                store.forget(mem.id)
            print(f"已遗忘 {len(mems)} 条。")
    finally:
        store.close()
    return 0


def cmd_dream(args: argparse.Namespace) -> int:
    """睡眠式整理：合并近义记忆 + 归档彻底忘掉的。"""
    cfg = load_config(args.config)
    store = _open_store(cfg)
    try:
        forget_below = float(cfg.get("memory", {}).get("forget_below", 0.06))
        swept = store.sweep(args.session, forget_below)
        merged = store.consolidate(args.session)
        store.prune_seen()
        print(f"归档 {swept} 条已经想不起来的，合并 {merged['merged']} 条重复的记忆。")
        print(f"剩下 {len(store.list(args.session, limit=100000))} 条。")
    finally:
        store.close()
    return 0


def cmd_persona(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    persona = load_persona(cfg.get("persona"))
    print(json.dumps(persona.to_dict(), ensure_ascii=False, indent=2))
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    store = _open_store(cfg)
    try:
        text = store.export(args.session)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as fh:
                fh.write(text)
            print(f"已导出到 {args.output}")
        else:
            print(text)
    finally:
        store.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fademem",
        description="会遗忘的角色扮演记忆层：给任意 AI 客户端加上遗忘曲线和人味。",
    )
    parser.add_argument("--version", action="version", version=f"fademem {__version__}")
    parser.add_argument("-c", "--config", help="配置文件路径（默认 ~/.fademem/config.json）", default=None)
    sub = parser.add_subparsers(dest="command")

    p_init = sub.add_parser("init", help="生成默认配置和角色卡模板")
    p_init.add_argument("--force", action="store_true", help="覆盖已存在的文件")
    p_init.add_argument("--persona", default=None, help="角色卡输出路径")
    p_init.set_defaults(func=cmd_init)

    p_serve = sub.add_parser("serve", help="启动代理服务")
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.add_argument("-v", "--verbose", action="store_true", help="打印每轮注入情况")
    p_serve.set_defaults(func=cmd_serve)

    p_mem = sub.add_parser("memories", help="查看记忆和它们的可提取度")
    p_mem.add_argument("--session", default=None)
    p_mem.add_argument("--all", action="store_true", help="包括已遗忘的")
    p_mem.add_argument("--limit", type=int, default=60)
    p_mem.set_defaults(func=cmd_memories)

    p_rem = sub.add_parser("remember", help="手动写一条记忆")
    p_rem.add_argument("text")
    p_rem.add_argument("--kind", default="fact")
    p_rem.add_argument("--subject", default="")
    p_rem.add_argument("--importance", type=float, default=0.8)
    p_rem.add_argument("--session", default="default")
    p_rem.set_defaults(func=cmd_remember)

    p_for = sub.add_parser("forget", help="主动忘掉")
    p_for.add_argument("--id", type=int, default=None)
    p_for.add_argument("--session", default="default")
    p_for.set_defaults(func=cmd_forget)

    p_dream = sub.add_parser("dream", help="整理记忆：合并重复、归档忘掉的")
    p_dream.add_argument("--session", default=None)
    p_dream.set_defaults(func=cmd_dream)

    p_persona = sub.add_parser("persona", help="打印当前生效的角色卡")
    p_persona.set_defaults(func=cmd_persona)

    p_export = sub.add_parser("export", help="导出记忆为 JSON")
    p_export.add_argument("--session", default=None)
    p_export.add_argument("-o", "--output", default=None)
    p_export.set_defaults(func=cmd_export)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
