"""端到端：起一个假的上游模型，验证记忆注入与流式透传。"""

import json
import os
import shutil
import tempfile
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from fademem.server import FadeMemApp, Handler

SEEN_SYSTEMS: list[str] = []
MODE = {"stream": False}


class MockUpstream(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        messages = body.get("messages") or []
        SEEN_SYSTEMS.append(
            "\n".join(
                m.get("content") or "" for m in messages if m.get("role") == "system"
            )
        )
        if body.get("stream"):
            return self._stream()
        return self._json()

    def _json(self):
        payload = {
            "id": "mock-1",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "当然可以！我在楼下。"},
                    "finish_reason": "stop",
                }
            ],
        }
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _stream(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def send(text: bytes) -> None:
            self.wfile.write(f"{len(text):X}\r\n".encode() + text + b"\r\n")
            self.wfile.flush()

        for piece in ("当然可以！", "我在楼", "下，", "你下来吧。"):
            obj = {"id": "mock-1", "choices": [{"index": 0, "delta": {"content": piece}}]}
            send(("data: " + json.dumps(obj, ensure_ascii=False) + "\n\n").encode())
        tail = {"id": "mock-1", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
        send(("data: " + json.dumps(tail) + "\n\n").encode())
        send(b"data: [DONE]\n\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


class IntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        cls.mockserver = ThreadingHTTPServer(("127.0.0.1", 0), MockUpstream)
        cls.mockserver.daemon_threads = True
        threading.Thread(target=cls.mockserver.serve_forever, daemon=True).start()
        mock_port = cls.mockserver.server_address[1]

        cfg = {
            "server": {"host": "127.0.0.1", "port": 0, "verbose": False},
            "upstream": {
                "base_url": f"http://127.0.0.1:{mock_port}/v1",
                "api_key": "test",
                "passthrough_auth": False,
                "timeout": 20,
            },
            "memory": {"db": os.path.join(cls.tmp, "m.db"), "sweep_every": 0},
            "persona": os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "examples",
                "persona.example.json",
            ),
            "style": {"enabled": True, "deslop": True, "deslop_stream": True},
            "prompt": {"merge_system": True},
        }
        cls.app = FadeMemApp(cfg)
        Handler.app = cls.app
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.httpd.daemon_threads = True
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()
        cls.mockserver.shutdown()
        cls.mockserver.server_close()
        cls.app.store.close()
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _post(self, payload):
        req = urllib.request.Request(
            self.base + "/v1/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            return resp.read()

    def test_1_extracts_memory_from_first_turn(self):
        SEEN_SYSTEMS.clear()
        self._post(
            {
                "model": "x",
                "user": "sess-a",
                "messages": [
                    {"role": "system", "content": "你是一个助手。"},
                    {"role": "user", "content": "我叫阿棠，我住在城西，我特别喜欢黑咖啡"},
                ],
            }
        )
        texts = [m.text for m in self.app.store.list("sess-a")]
        joined = " ".join(texts)
        self.assertIn("叫阿棠", joined)
        self.assertIn("住在城西", joined)
        self.assertTrue(any(k.startswith("喜欢") for k in texts))

    def test_2_injects_recalled_memory(self):
        SEEN_SYSTEMS.clear()
        self._post(
            {
                "model": "x",
                "user": "sess-a",
                "messages": [
                    {"role": "system", "content": "你是一个助手。"},
                    {"role": "user", "content": "我叫阿棠，我住在城西，我特别喜欢黑咖啡"},
                    {"role": "assistant", "content": "嗯。"},
                    {"role": "user", "content": "我住哪儿来着"},
                ],
            }
        )
        self.assertTrue(SEEN_SYSTEMS, "上游没有收到请求")
        system = SEEN_SYSTEMS[-1]
        self.assertIn("住在城西", system)
        self.assertIn("【你是谁】", system)
        self.assertIn("沈知夏", system)
        self.assertIn("说话方式", system)
        # 客户端的原始 system 必须被保留，只是被拼在后面
        self.assertIn("你是一个助手。", system)

    def test_3_deslop_applied_to_non_stream(self):
        body = json.loads(
            self._post(
                {
                    "model": "x",
                    "user": "sess-b",
                    "messages": [{"role": "user", "content": "在吗"}],
                }
            )
        )
        content = body["choices"][0]["message"]["content"]
        self.assertNotIn("当然可以", content)
        self.assertIn("我在楼下。", content)

    def test_4_streaming_relay(self):
        raw = self._post(
            {
                "model": "x",
                "user": "sess-c",
                "stream": True,
                "messages": [{"role": "user", "content": "在吗"}],
            }
        ).decode("utf-8")
        self.assertIn("data: [DONE]", raw)
        pieces = []
        for line in raw.splitlines():
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                continue
            obj = json.loads(payload)
            for choice in obj.get("choices") or []:
                delta = choice.get("delta") or {}
                if isinstance(delta.get("content"), str):
                    pieces.append(delta["content"])
        text = "".join(pieces)
        self.assertNotIn("当然可以", text)
        self.assertEqual(text, "我在楼下，你下来吧。")

    def test_5_session_isolation(self):
        self._post(
            {
                "model": "x",
                "user": "sess-d",
                "messages": [{"role": "user", "content": "我喜欢吃火锅"}],
            }
        )
        self.assertFalse(
            any("火锅" in m.text for m in self.app.store.list("sess-a")),
            "不同会话的记忆不能串",
        )
        self.assertTrue(any("火锅" in m.text for m in self.app.store.list("sess-d")))

    def test_6_remember_endpoint(self):
        req = urllib.request.Request(
            self.base + "/v1/fademem/remember",
            data=json.dumps({"session": "sess-e", "text": "不吃香菜", "kind": "dislike"}).encode(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            body = json.loads(resp.read())
        self.assertTrue(body["created"])
        self.assertEqual(body["memory"]["text"], "不吃香菜")

    def test_7_health(self):
        with urllib.request.urlopen(self.base + "/health", timeout=10) as resp:
            body = json.loads(resp.read())
        self.assertTrue(body["ok"])


if __name__ == "__main__":
    unittest.main()
