import json
import types
import unittest
from unittest import mock

from fademem.server import (
    Handler,
    StreamRewriter,
    content_to_text,
    inject_system,
    resolve_session,
)
from fademem.store import Candidate


def delta(text):
    return {"id": "1", "choices": [{"index": 0, "delta": {"content": text}}]}


def joined(objs):
    out = []
    for obj in objs:
        for choice in obj.get("choices") or []:
            text = (choice.get("delta") or {}).get("content")
            if isinstance(text, str):
                out.append(text)
    return "".join(out)


class StreamRewriterTest(unittest.TestCase):
    def test_removes_leading_noise_across_chunks(self):
        rw = StreamRewriter()
        emitted = []
        emitted += rw.feed(delta("当然"))
        emitted += rw.feed(delta("可以！"))
        emitted += rw.feed(delta("我在。"))
        emitted += rw.finish()
        self.assertEqual(joined(emitted), "我在。")

    def test_keeps_normal_text_intact(self):
        rw = StreamRewriter()
        emitted = []
        for piece in ("嗯，", "我在楼下", "。", "你下来吧"):
            emitted += rw.feed(delta(piece))
        emitted += rw.finish()
        self.assertEqual(joined(emitted), "嗯，我在楼下。你下来吧")

    def test_holds_until_punctuation(self):
        rw = StreamRewriter(hold=24)
        self.assertEqual(rw.feed(delta("嗯")), [])
        out = rw.feed(delta("。"))
        self.assertTrue(out)

    def test_finish_flushes_short_stream(self):
        rw = StreamRewriter(hold=24)
        self.assertEqual(rw.feed(delta("在")), [])
        self.assertEqual(joined(rw.finish()), "在")

    def test_preserves_event_structure(self):
        rw = StreamRewriter()
        first = {
            "id": "abc", "model": "m",
            "choices": [{"index": 0, "delta": {"role": "assistant"}}],
        }
        emitted = rw.feed(first)
        emitted += rw.feed(delta("好的，走吧。"))
        self.assertTrue(all("choices" in o for o in emitted))
        self.assertEqual(emitted[0]["model"], "m")
        self.assertEqual(joined(emitted), "走吧。")

    def test_no_content_deltas_pass_through(self):
        rw = StreamRewriter()
        usage = {"choices": [], "usage": {"total_tokens": 5}}
        out = rw.feed(usage) + rw.finish()
        self.assertEqual(out[-1]["usage"]["total_tokens"], 5)

    def test_finish_reason_triggers_decision(self):
        rw = StreamRewriter(hold=999)
        obj = {"choices": [{"delta": {"content": "好"}, "finish_reason": "stop"}]}
        self.assertTrue(rw.feed(obj))


class HelperTest(unittest.TestCase):
    def test_content_to_text_string(self):
        self.assertEqual(content_to_text("abc"), "abc")

    def test_content_to_text_multimodal(self):
        content = [
            {"type": "text", "text": "第一段"},
            {"type": "image_url", "image_url": {"url": "x"}},
            {"type": "text", "text": "第二段"},
        ]
        self.assertEqual(content_to_text(content), "第一段\n第二段")

    def test_content_to_text_none(self):
        self.assertEqual(content_to_text(None), "")

    def test_resolve_session_header_wins(self):
        headers = {"X-Fademem-Session": "  chat-7 "}
        self.assertEqual(resolve_session(headers, {"user": "ignored"}), "chat-7")

    def test_resolve_session_falls_back_to_user(self):
        self.assertEqual(resolve_session({}, {"user": "u1"}), "u1")

    def test_resolve_session_default(self):
        self.assertEqual(resolve_session({}, {}), "default")

    def test_inject_merges_into_existing_system(self):
        msgs = [{"role": "system", "content": "你是助手。"}, {"role": "user", "content": "hi"}]
        out = inject_system(msgs, "【你是沈知夏】", merge=True)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["role"], "system")
        self.assertIn("你是助手。", out[0]["content"])
        self.assertTrue(out[0]["content"].rstrip().endswith("【你是沈知夏】"))

    def test_inject_adds_when_no_system(self):
        msgs = [{"role": "user", "content": "hi"}]
        out = inject_system(msgs, "块", merge=True)
        self.assertEqual(len(out), 2)
        self.assertEqual(out[0]["role"], "system")

    def test_inject_handler_does_not_mutate_input(self):
        msgs = [{"role": "system", "content": "orig"}]
        inject_system(msgs, "new", merge=True)
        self.assertEqual(msgs[0]["content"], "orig")


class ModelsFallbackTest(unittest.TestCase):
    """客户端启动会拉一次模型列表，上游不可达时不能把 UI 卡住。"""

    def _handler(self, upstream, auth=None):
        h = object.__new__(Handler)
        h.app = types.SimpleNamespace(cfg={"upstream": upstream})
        h.headers = {"Authorization": auth} if auth else {}
        h.captured = None
        h.close_connection = False
        h._json = lambda status, payload: setattr(h, "captured", (status, payload))
        return h

    def test_no_key_and_no_client_auth_skips_network(self):
        h = self._handler({"base_url": "https://例.invalid/v1", "api_key": ""})
        with mock.patch("fademem.server.urllib.request.urlopen",
                        side_effect=AssertionError("不该发请求")):
            h._models()
        self.assertEqual(h.captured[0], 200)
        self.assertEqual(h.captured[1]["data"][0]["id"], "fademem-proxy")

    def test_unreachable_upstream_falls_back(self):
        h = self._handler({"base_url": "https://例.invalid/v1", "api_key": "sk-x"})
        with mock.patch("fademem.server.urllib.request.urlopen",
                        side_effect=OSError("boom")):
            h._models()
        self.assertEqual(h.captured[1]["data"][0]["id"], "fademem-proxy")

    def test_probes_upstream_when_client_sends_auth(self):
        h = self._handler({"base_url": "https://例.invalid/v1", "api_key": ""}, auth="Bearer abc")
        fake = mock.MagicMock()
        fake.__enter__.return_value.read.return_value = b'{"object":"list","data":[]}'
        with mock.patch("fademem.server.urllib.request.urlopen", return_value=fake) as m:
            h._models()
        self.assertTrue(m.called)
        self.assertEqual(h.captured[1], {"object": "list", "data": []})


class BrokenPipeTest(unittest.TestCase):
    """客户端中途断开是常态，不该把堆栈吐到用户终端。"""

    def _handler(self):
        h = object.__new__(Handler)
        h.close_connection = False
        h.wfile = mock.MagicMock()
        h.wfile.write.side_effect = ConnectionAbortedError(10053, "aborted")
        return h

    def test_write_reports_failure_without_raising(self):
        h = self._handler()
        self.assertFalse(h._write(b"data"))
        self.assertTrue(h.close_connection)

    def test_chunk_swallows_disconnect(self):
        h = self._handler()
        h._chunk(b"data")  # 不抛异常即通过
        self.assertTrue(h.close_connection)

    def test_write_passes_through_normally(self):
        h = self._handler()
        h.wfile = mock.MagicMock()
        self.assertTrue(h._write(b"data"))
        self.assertFalse(h.close_connection)


class UpstreamHintTest(unittest.TestCase):

    def _handler(self, upstream, auth=None):
        h = object.__new__(Handler)
        h.app = types.SimpleNamespace(cfg={"upstream": upstream})
        h.headers = {"Authorization": auth} if auth else {}
        return h

    def test_hints_missing_key(self):
        hint = self._handler({"base_url": "https://x/v1", "api_key": ""})._upstream_hint()
        self.assertIn("api_key", hint)

    def test_hints_bad_url_when_key_present(self):
        hint = self._handler({"base_url": "https://x/v1", "api_key": "sk-1"})._upstream_hint()
        self.assertIn("base_url", hint)

    def test_passthrough_auth_counts_as_configured(self):
        h = self._handler({"base_url": "https://x/v1", "api_key": ""}, auth="Bearer u")
        self.assertIn("base_url", h._upstream_hint())


if __name__ == "__main__":
    unittest.main()
