"""OpenAI 兼容代理服务。

客户端（SillyTavern / Cherry Studio / Chatbox / 自己的脚本）把 Base URL 指到
``http://127.0.0.1:8770/v1`` 就能用。服务做三件事：

1. 收到请求 → 识别会话 → 对没见过的用户发言做记忆抽取；
2. 按当前这句话召回记忆，把「人设 + 记忆 + 说话方式」注入 system；
3. 转发给上游模型，流式原样回传（只在必要处改写 content 字段，不动事件结构）。
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Iterable, Sequence

from . import __version__, extract, retrieve
from .config import load_config
from .persona import Persona, load_persona
from .prompt import build_system_block
from .prompt_style import StreamDeslop, StyleConfig, deslop
from .store import Store

__all__ = ["FadeMemApp", "serve", "StreamRewriter"]

_LEADING_NOISE = re.compile(
    r"^\s*(?:当然(?:可以|没问题)?[，,！!。]?\s*|好的[，,。]?\s*|好[，,。]\s*|"
    r"没问题[，,。]\s*|明白了?[，,。]\s*|我明白了?[，,。]\s*|嗯嗯[，,。]\s*|收到[，,。]\s*)"
)


# ---------------------------------------------------------------------------
# 请求/响应小工具
# ---------------------------------------------------------------------------

def content_to_text(content: Any) -> str:
    """把 OpenAI 的 content（字符串或多模态数组）压成纯文本。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") in (None, "text", "input_text"):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
        return "\n".join(parts)
    return ""


def turn_digest(role: str, text: str) -> str:
    norm = re.sub(r"\s+", " ", (text or "").strip())
    return hashlib.sha1(f"{role}|{norm}".encode("utf-8", "ignore")).hexdigest()


def resolve_session(headers: Any, body: dict[str, Any]) -> str:
    for key in ("X-Fademem-Session", "X-Session-Id", "X-Conversation-Id"):
        value = headers.get(key) if headers is not None else None
        if isinstance(value, str) and value.strip():
            return value.strip()[:120]
    user = body.get("user")
    if isinstance(user, str) and user.strip():
        return user.strip()[:120]
    meta = body.get("metadata")
    if isinstance(meta, dict):
        for key in ("session_id", "session", "conversation_id", "chat_id"):
            value = meta.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:120]
    return "default"


def inject_system(
    messages: Sequence[dict[str, Any]], block: str, merge: bool = True
) -> list[dict[str, Any]]:
    """把 system 段插进去。默认合并到已有 system，避免多 system 被某些服务商拒绝。"""
    out = [dict(m) for m in messages]
    if merge and out and out[0].get("role") in ("system", "developer"):
        old = content_to_text(out[0].get("content")).strip()
        # 我们的段落放后面：模型对靠近末尾的指令更敏感
        out[0]["content"] = (old + "\n\n" + block).strip() if old else block
        return out
    out.insert(0, {"role": "system", "content": block})
    return out


class StreamRewriter:
    """去掉流式输出开头那段 AI 腔，同时**不改变事件结构**。

    做法是把还没决定的前几个 chunk 先扣在手里，等攒够字符（或碰到句末标点）
    再一次性算出去掉多少前缀，然后按原样把这些事件补发出去。这样既不会凭空
    造一个新事件，也不会丢字。
    """

    def __init__(self, hold: int = 24) -> None:
        self.hold = hold
        self.acc = ""
        self.held: list[dict[str, Any]] = []
        self.decided = False

    @staticmethod
    def _delta_text(obj: dict[str, Any]) -> str:
        out = []
        for choice in obj.get("choices") or []:
            delta = choice.get("delta") or {}
            text = delta.get("content")
            if isinstance(text, str):
                out.append(text)
        return "".join(out)

    @staticmethod
    def _finished(obj: dict[str, Any]) -> bool:
        for choice in obj.get("choices") or []:
            if choice.get("finish_reason"):
                return True
        return False

    def feed(self, obj: dict[str, Any]) -> list[dict[str, Any]]:
        if self.decided:
            return [obj]
        self.held.append(obj)
        self.acc += self._delta_text(obj)
        if (
            len(self.acc) >= self.hold
            or re.search(r"[。！？\n…]", self.acc)
            or self._finished(obj)
        ):
            return self._decide()
        return []

    def finish(self) -> list[dict[str, Any]]:
        if self.decided or not self.held:
            self.decided = True
            return []
        return self._decide()

    def _decide(self) -> list[dict[str, Any]]:
        self.decided = True
        stripped = _LEADING_NOISE.sub("", self.acc, count=1)
        cut = len(self.acc) - len(stripped)
        out, self.held = self.held, []
        if cut <= 0:
            return out
        for obj in out:
            for choice in obj.get("choices") or []:
                delta = choice.get("delta")
                if not isinstance(delta, dict):
                    continue
                text = delta.get("content")
                if isinstance(text, str) and text and cut > 0:
                    take = min(cut, len(text))
                    delta["content"] = text[take:]
                    cut -= take
        return out


# ---------------------------------------------------------------------------
# 应用
# ---------------------------------------------------------------------------

class FadeMemApp:
    def __init__(self, cfg: dict[str, Any] | None = None) -> None:
        self.cfg = cfg or load_config()
        mem = self.cfg.get("memory", {})
        self.store = Store(
            path=mem.get("db") or ":memory:",
            base_stability=float(mem.get("base_stability_days", 6.0)),
        )
        self.scoring = retrieve.ScoringConfig(
            alpha=float(mem.get("alpha", 1.4)),
            beta=float(mem.get("beta", 0.9)),
            max_items=int(mem.get("max_items", 8)),
            min_score=float(mem.get("min_score", 0.02)),
            forget_below=float(mem.get("forget_below", 0.06)),
            ambient_items=int(mem.get("ambient_items", 2)),
            ambient_min_importance=float(mem.get("ambient_min_importance", 0.55)),
            flashback_rate=float(mem.get("flashback_rate", 0.05)),
            fuzz_below=float(mem.get("fuzz_below", 0.55)),
            reinforce_on_recall=bool(mem.get("reinforce_on_recall", True)),
        )
        style = self.cfg.get("style", {})
        self.style = StyleConfig(
            enabled=bool(style.get("enabled", True)),
            max_sentences=int(style.get("max_sentences", 3)),
            include_self_check=bool(style.get("include_self_check", True)),
            include_banned=bool(style.get("include_banned", True)),
            extra_rules=list(style.get("extra_rules") or []),
        )
        self._persona: Persona | None = None
        self._persona_mtime: float = -1.0
        self.lock = threading.RLock()
        self._counter = 0

    # ---- 人设 -----------------------------------------------------------
    @property
    def persona(self) -> Persona:
        path = self.cfg.get("persona")
        try:
            mtime = __import__("os").path.getmtime(path) if path else -1.0
        except OSError:
            mtime = -1.0
        if self._persona is None or mtime != self._persona_mtime:
            self._persona = load_persona(path)
            self._persona_mtime = mtime
        return self._persona

    # ---- 核心流程 -------------------------------------------------------
    def prepare(
        self, session: str, messages: Sequence[dict[str, Any]]
    ) -> tuple[list[retrieve.Hit], list[dict[str, Any]]]:
        """抽取新记忆 → 召回 → 注入。返回 (hits, 新的 messages)。"""
        query = ""
        with self.lock:
            messages = list(messages)
            for idx, msg in enumerate(messages):
                role = str(msg.get("role") or "")
                if role != "user":
                    continue
                text = content_to_text(msg.get("content"))
                if not text.strip():
                    continue
                query = text
                digest = turn_digest("user", text)
                if self.store.seen(session, digest):
                    continue
                self.store.mark_seen(session, digest)
                self._absorb(session, text, "user", messages, idx)

            self._counter += 1
            every = int(self.cfg.get("memory", {}).get("sweep_every", 50) or 0)
            if every and self._counter % every == 0:
                try:
                    self.store.sweep(session, self.scoring.forget_below)
                    self.store.prune_seen()
                except Exception:  # noqa: BLE001
                    pass

        hits = retrieve.recall(self.store, session, query, self.scoring) if query else []
        block = build_system_block(
            self.persona,
            hits,
            self.style,
            language=str(self.cfg.get("prompt", {}).get("language", "auto")),
        )
        merged = inject_system(
            messages, block, merge=bool(self.cfg.get("prompt", {}).get("merge_system", True))
        )
        return hits, merged

    def _absorb(
        self,
        session: str,
        text: str,
        role: str,
        messages: Sequence[dict[str, Any]],
        idx: int,
    ) -> None:
        mode = str(self.cfg.get("extraction", {}).get("mode", "heuristic"))
        if mode == "off":
            return
        max_items = int(self.cfg.get("extraction", {}).get("max_items", 6))
        allow_generic = bool(self.cfg.get("extraction", {}).get("allow_generic", True))

        for cand in extract.extract(
            text, role=role, max_items=max_items, allow_generic=allow_generic
        ):
            self.store.add(session, cand)

        # 角色自己刚说过的承诺，也值得记住
        if idx > 0 and messages[idx - 1].get("role") == "assistant":
            prev = content_to_text(messages[idx - 1].get("content"))
            if prev and re.search(r"我(?:答应|保证|会|一定|尽量)", prev):
                for cand in extract.extract(prev, role="assistant", max_items=2,
                                            allow_generic=False):
                    self.store.add(session, cand)


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = f"fademem/{__version__}"
    app: FadeMemApp

    # ---- 基础 -----------------------------------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        if getattr(self.app, "cfg", {}).get("server", {}).get("verbose"):
            super().log_message(fmt, *args)

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

    def _write(self, data: bytes) -> bool:
        """写回客户端。客户端中途断开是常态（用户关窗口、切会话），不该刷堆栈。"""
        if not data:
            return True
        try:
            self.wfile.write(data)
            return True
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            self.close_connection = True
            return False

    def _json(self, status: int, payload: Any) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self._cors()
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            self.close_connection = True
            return
        self._write(data)

    def _chunk(self, data: bytes) -> None:
        if not data:
            return
        try:
            if not self._write(f"{len(data):X}\r\n".encode("ascii")):
                return
            if not self._write(data):
                return
            if not self._write(b"\r\n"):
                return
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
            self.close_connection = True

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length > 0 else b""
        if not raw:
            return {}
        data = json.loads(raw.decode("utf-8", "replace"))
        return data if isinstance(data, dict) else {}

    # ---- 路由 -----------------------------------------------------------
    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors()
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/") or "/"
        if path in ("/", "/debug"):
            return self._page()
        if path == "/health":
            return self._json(200, {"ok": True, "version": __version__,
                                    **self.app.store.stats()})
        if path == "/v1/fademem/memories":
            session = "default"
            if "?" in self.path:
                from urllib.parse import parse_qs
                qs = parse_qs(self.path.split("?", 1)[1])
                session = (qs.get("session") or ["default"])[0]
            mems = self.app.store.list(session, limit=200)
            return self._json(200, {"session": session,
                                    "memories": [m.as_dict() for m in mems]})
        if path == "/v1/fademem/sessions":
            return self._json(200, {"sessions": self.app.store.sessions()})
        if path == "/v1/models":
            return self._models()
        return self._json(404, {"error": {"message": f"没有这个路径：{path}"}})

    def do_POST(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        try:
            if path in ("/v1/chat/completions", "/chat/completions"):
                return self._chat()
            if path == "/v1/fademem/remember":
                return self._remember()
            if path == "/v1/fademem/forget":
                return self._forget()
            if path == "/v1/fademem/consolidate":
                return self._consolidate()
            return self._json(404, {"error": {"message": f"没有这个路径：{path}"}})
        except BrokenPipeError:
            return
        except Exception as exc:  # noqa: BLE001
            try:
                self._json(500, {"error": {"message": f"{type(exc).__name__}: {exc}"}})
            except Exception:  # noqa: BLE001
                pass

    # ---- 端点 -----------------------------------------------------------
    def _remember(self) -> None:
        body = self._read_json()
        text = str(body.get("text") or "").strip()
        if not text:
            return self._json(400, {"error": {"message": "text 不能为空"}})
        session = str(body.get("session") or "default")
        cand = extract.Candidate(
            kind=str(body.get("kind") or "fact"),
            text=extract.redact(text),
            subject=str(body.get("subject") or ""),
            importance=float(body.get("importance") or 0.7),
            salience=float(body.get("salience") or 0.3),
            source="api",
        )
        mem, created = self.app.store.add(session, cand)
        return self._json(200, {"created": created, "memory": mem.as_dict()})

    def _forget(self) -> None:
        body = self._read_json()
        if body.get("id"):
            ok = self.app.store.forget(int(body["id"]))
            return self._json(200, {"forgotten": ok})
        if body.get("all") and body.get("session"):
            mems = self.app.store.list(str(body["session"]), limit=100000)
            for mem in mems:
                self.app.store.forget(mem.id)
            return self._json(200, {"forgotten": len(mems)})
        return self._json(400, {"error": {"message": "需要 id，或 session + all=true"}})

    def _consolidate(self) -> None:
        body = self._read_json()
        session = body.get("session")
        result = self.app.store.consolidate(str(session) if session else None)
        return self._json(200, result)

    _FALLBACK_MODELS = {
        "object": "list",
        "data": [{"id": "fademem-proxy", "object": "model", "owned_by": "fademem"}],
    }

    def _models(self) -> None:
        up = self.app.cfg.get("upstream", {})
        base = str(up.get("base_url", "")).rstrip("/")
        key = (up.get("api_key") or "").strip()
        auth = self.headers.get("Authorization")
        # 客户端（Cherry Studio / Chatbox / 酒馆）常在启动时拉一次模型列表。
        # 没配上游或没凭据时直接给兜底列表，别让 UI 卡在转圈上。
        if not base or not (key or (up.get("passthrough_auth", True) and auth)):
            return self._json(200, self._FALLBACK_MODELS)
        try:
            req = urllib.request.Request(base + "/models", method="GET")
            if up.get("passthrough_auth", True) and auth:
                req.add_header("Authorization", auth)
            else:
                req.add_header("Authorization", f"Bearer {key}")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return self._json(200, json.loads(resp.read().decode("utf-8")))
        except Exception:  # noqa: BLE001
            return self._json(200, self._FALLBACK_MODELS)

    def _page(self) -> None:
        stats = self.app.store.stats()
        sessions = self.app.store.sessions()
        rows = []
        now = time.time()
        for sess in sessions[:12]:
            for mem in self.app.store.list(sess["session"], limit=40):
                r = mem.retrievability(now)
                pct = int(max(0.0, min(1.0, r)) * 100)
                rows.append(
                    f"<tr><td class='s'>{sess['session']}</td><td class='k'>{mem.kind}</td>"
                    f"<td>{mem.text}</td><td class='bar'><span style='width:{pct}%'></span></td>"
                    f"<td class='n'>{r:.2f}</td><td class='n'>{mem.stability:.1f}d</td></tr>"
                )
        body = (
            "<!doctype html><meta charset='utf-8'><title>fademem</title>"
            "<style>body{font:14px/1.6 -apple-system,'Segoe UI',sans-serif;margin:32px;"
            "color:#1a1a1a;background:#fafafa}h1{font-size:18px;font-weight:600}"
            ".meta{color:#666;margin-bottom:20px}table{border-collapse:collapse;width:100%;"
            "background:#fff;box-shadow:0 1px 3px rgba(0,0,0,.08);border-radius:6px}"
            "th,td{padding:7px 10px;text-align:left;border-bottom:1px solid #eee;font-size:13px}"
            "th{background:#f4f4f4;font-weight:600;color:#444}"
            ".s{color:#888;font-family:ui-monospace,monospace;font-size:12px}"
            ".k{color:#0a7;font-size:12px}.n{text-align:right;color:#666;"
            "font-family:ui-monospace,monospace}.bar{width:90px}"
            ".bar span{display:block;height:5px;background:#3b82f6;border-radius:3px}"
            "</style>"
            f"<h1>fademem {__version__}</h1>"
            f"<div class='meta'>{stats['memories']} 条记忆 / {stats['sessions']} 个会话 / "
            f"已遗忘 {stats['archived']}<br>把客户端的 Base URL 设成 "
            f"<code>http://127.0.0.1:{self.app.cfg['server']['port']}/v1</code></div>"
            "<table><tr><th>会话</th><th>类型</th><th>内容</th><th>可提取度</th>"
            "<th>R</th><th>半衰期</th></tr>" + "".join(rows) + "</table>"
        )
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self._cors()
        self.end_headers()
        self.wfile.write(data)

    # ---- 聊天转发 -------------------------------------------------------
    def _open_upstream(self, payload: dict[str, Any]) -> Any:
        up = self.app.cfg.get("upstream", {})
        url = str(up.get("base_url") or "").rstrip("/") + "/chat/completions"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "text/event-stream, application/json")

        key = (up.get("api_key") or "").strip()
        client_auth = self.headers.get("Authorization")
        if up.get("passthrough_auth", True) and client_auth:
            req.add_header("Authorization", client_auth)
        elif key:
            req.add_header("Authorization", f"Bearer {key}")
        for name, value in (up.get("extra_headers") or {}).items():
            req.add_header(str(name), str(value))
        return urllib.request.urlopen(req, timeout=float(up.get("timeout") or 180))

    def _upstream_hint(self) -> str:
        """第一次跑就会撞上的错误，提示要能直接照着改。"""
        up = self.app.cfg.get("upstream", {})
        key = (up.get("api_key") or "").strip()
        client_auth = self.headers.get("Authorization")
        if not key and not (up.get("passthrough_auth", True) and client_auth):
            return "还没配 API key：编辑 config.json 里的 upstream.api_key。"
        return (
            "检查 config.json 里的 upstream.base_url 是否写对；"
            "若上游在墙外，需要给本进程设置 HTTPS_PROXY 环境变量。"
        )

    def _chat(self) -> None:
        try:
            payload = self._read_json()
        except json.JSONDecodeError:
            return self._json(400, {"error": {"message": "请求体不是合法 JSON"}})

        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            return self._json(
                400, {"error": {"message": "messages 不能为空", "type": "invalid_request_error"}}
            )

        session = resolve_session(self.headers, payload)
        try:
            hits, injected = self.app.prepare(session, messages)
        except Exception as exc:  # noqa: BLE001 - 记忆层挂了不能拖垮对话
            print(f"[fademem] 记忆层异常，降级为纯代理：{type(exc).__name__}: {exc}")
            hits, injected = [], list(messages)

        payload = dict(payload)
        payload["messages"] = injected
        if getattr(self.app, "cfg", {}).get("server", {}).get("verbose"):
            print(f"[fademem] session={session} 注入记忆 {len(hits)} 条")

        stream = bool(payload.get("stream"))
        try:
            resp = self._open_upstream(payload)
        except urllib.error.HTTPError as exc:
            body = exc.read()
            self.send_response(exc.code)
            self.send_header("Content-Type", exc.headers.get("Content-Type", "application/json"))
            self.send_header("Content-Length", str(len(body)))
            self._cors()
            self.end_headers()
            self.wfile.write(body)
            return
        except Exception as exc:  # noqa: BLE001
            return self._json(
                502,
                {
                    "error": {
                        "message": f"上游请求失败：{type(exc).__name__}: {exc}\n{self._upstream_hint()}",
                        "type": "upstream_error",
                    }
                },
            )

        with resp:
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if stream and "event-stream" in ctype:
                return self._relay_stream(resp)
            raw = resp.read()
            if "application/json" not in ctype:
                self.send_response(resp.status)
                self.send_header("Content-Type", ctype or "application/octet-stream")
                self.send_header("Content-Length", str(len(raw)))
                self._cors()
                self.end_headers()
                self.wfile.write(raw)
                return
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                return self._json(502, {"error": {"message": "上游返回了无法解析的内容"}})
            if isinstance(data, dict) and self.app.cfg.get("style", {}).get("deslop", True):
                for choice in data.get("choices") or []:
                    message = choice.get("message") or {}
                    text = message.get("content")
                    if isinstance(text, str):
                        message["content"] = deslop(text)
            return self._json(resp.status, data)

    def _relay_stream(self, resp: Any) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("Transfer-Encoding", "chunked")
        self._cors()
        self.end_headers()

        enabled = bool(self.app.cfg.get("style", {}).get("deslop_stream", True))
        rewriter = StreamRewriter() if enabled else None

        pending: list[str] = []
        saw_done = False

        def emit(objs: Iterable[dict[str, Any]], head: Sequence[str] = ()) -> None:
            for obj in objs:
                lines = list(head)
                lines.append("data: " + json.dumps(obj, ensure_ascii=False))
                self._chunk(("\n".join(lines) + "\n\n").encode("utf-8"))

        try:
            for raw in resp:
                line = raw.decode("utf-8", "replace").rstrip("\r\n")
                if line == "":
                    if not pending:
                        continue
                    saw_done = self._relay_event(pending, rewriter, emit, saw_done)
                    pending = []
                    continue
                pending.append(line)
            if pending:
                saw_done = self._relay_event(pending, rewriter, emit, saw_done)

            if rewriter is not None:
                emit(rewriter.finish())
            if saw_done:
                self._chunk(b"data: [DONE]\n\n")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return

    def _relay_event(self, lines: Sequence[str], rewriter: StreamRewriter | None,
                     emit: Any, saw_done: bool) -> bool:
        data_idx = [i for i, ln in enumerate(lines) if ln.startswith("data:")]
        if not data_idx:
            self._chunk(("\n".join(lines) + "\n\n").encode("utf-8"))
            return saw_done

        payload_text = "\n".join(lines[i][5:].lstrip() for i in data_idx).strip()
        head = [ln for i, ln in enumerate(lines) if i not in data_idx]

        if payload_text == "[DONE]":
            return True
        try:
            obj = json.loads(payload_text)
        except json.JSONDecodeError:
            self._chunk(("\n".join(lines) + "\n\n").encode("utf-8"))
            return saw_done

        to_send = rewriter.feed(obj) if rewriter is not None else [obj]
        emit(to_send, head)
        return saw_done


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------

def serve(cfg: dict[str, Any] | None = None) -> None:
    app = FadeMemApp(cfg)
    Handler.app = app
    server_cfg = app.cfg.get("server", {})
    host = str(server_cfg.get("host", "127.0.0.1"))
    port = int(server_cfg.get("port", 8770))
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True

    persona = app.persona
    print(f"fademem {__version__}")
    print(f"  监听     http://{host}:{port}/v1")
    print(f"  调试页   http://{host}:{port}/")
    print(f"  角色     {persona.name}")
    print(f"  记忆库   {app.store.stats()['db']}")
    print("  把客户端的 Base URL 指向上面的 /v1 地址即可。Ctrl+C 退出。")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n收工。")
    finally:
        httpd.server_close()
        app.store.close()
