import os
import random
import shutil
import tempfile
import time
import unittest

from fademem.retrieve import ScoringConfig, recall
from fademem.store import Candidate, Store

DAY = 86400


class RecallTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.store = Store(os.path.join(self.dir, "m.db"))
        self.now = time.time()

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def _add(self, text, kind="fact", importance=0.7, subject="", days_ago=0.0, stability=None):
        mem, _ = self.store.add(
            "s", Candidate(kind, text, subject=subject or text[:6], importance=importance)
        )
        if stability is not None:
            self.store.conn.execute(
                "UPDATE memories SET stability=?, updated_at=? WHERE id=?",
                (stability, self.now - days_ago * DAY, mem.id),
            )
            self.store.conn.commit()
        elif days_ago:
            self.store.conn.execute(
                "UPDATE memories SET updated_at=? WHERE id=?", (self.now - days_ago * DAY, mem.id)
            )
            self.store.conn.commit()
        self.store._invalidate_index()
        return mem

    def test_relevant_memory_ranks_first(self):
        self._add("养了一只三花猫", subject="三花猫")
        self._add("在杂志社上班", subject="杂志社")
        self._add("讨厌下雨天", subject="下雨天")
        hits = recall(self.store, "s", "我家那只猫最近掉毛", now=self.now)
        self.assertTrue(hits)
        self.assertIn("猫", hits[0].memory.text)

    def test_forgotten_memory_excluded(self):
        mem = self._add("随口说过的一句闲话", importance=0.1)
        self.store.conn.execute(
            "UPDATE memories SET updated_at=? WHERE id=?", (self.now - 300 * DAY, mem.id)
        )
        self.store.conn.commit()
        self.store._invalidate_index()
        hits = recall(self.store, "s", "闲话", ScoringConfig(flashback_rate=0), now=self.now)
        self.assertEqual([h for h in hits if h.memory.id == mem.id], [])

    def test_no_memories_returns_empty(self):
        self.assertEqual(recall(self.store, "nobody", "随便问问", now=self.now), [])

    def test_important_memory_surfaces_without_keyword(self):
        """没有关键词命中，但重要的记忆也该在背景里。"""
        self._add("叫阿棠", kind="identity", importance=0.95, subject="阿棠")
        hits = recall(
            self.store, "s", "今天天气怎么样", ScoringConfig(flashback_rate=0), now=self.now
        )
        self.assertTrue(any("阿棠" in h.memory.text for h in hits))

    def test_irrelevant_low_importance_stays_out(self):
        self._add("上周三买了把雨伞", importance=0.2, subject="雨伞")
        hits = recall(
            self.store, "s", "今天天气怎么样", ScoringConfig(flashback_rate=0), now=self.now
        )
        self.assertFalse(any("雨伞" in h.memory.text for h in hits))

    def test_ambient_items_capped(self):
        for i in range(10):
            self._add(f"重要的事第{i}条", importance=0.9, subject=f"事{i}")
        hits = recall(
            self.store, "s", "无关的话",
            ScoringConfig(ambient_items=2, flashback_rate=0), now=self.now,
        )
        self.assertLessEqual(len(hits), 2)

    def test_flashback_marks_spontaneous(self):
        mem = self._add("说过讨厌雨天", importance=0.9, subject="雨天")
        self.store.conn.execute(
            "UPDATE memories SET stability=3.0, updated_at=? WHERE id=?",
            (self.now - 3 * DAY, mem.id),
        )
        self.store.conn.commit()
        self.store._invalidate_index()
        hits = recall(
            self.store, "s", "完全无关的一句",
            ScoringConfig(flashback_rate=1.0, ambient_items=0),
            now=self.now, rng=random.Random(0),
        )
        self.assertTrue(any(h.spontaneous for h in hits))
        flash = [h for h in hits if h.spontaneous][0]
        self.assertIn("雨天", flash.memory.text)

    def test_no_flashback_when_rate_zero(self):
        mem = self._add("说过讨厌雨天", importance=0.9, subject="雨天")
        self.store.conn.execute(
            "UPDATE memories SET stability=3.0, updated_at=? WHERE id=?",
            (self.now - 3 * DAY, mem.id),
        )
        self.store.conn.commit()
        self.store._invalidate_index()
        hits = recall(
            self.store, "s", "完全无关的一句",
            ScoringConfig(flashback_rate=0.0, ambient_items=0), now=self.now,
        )
        self.assertFalse(any(h.spontaneous for h in hits))

    def test_recall_reinforces(self):
        mem = self._add("养了一只三花猫", subject="三花猫")
        before = self.store.get(mem.id).access_count
        recall(self.store, "s", "猫", ScoringConfig(flashback_rate=0), now=self.now)
        self.assertGreater(self.store.get(mem.id).access_count, before)

    def test_hazy_memory_is_fuzzed(self):
        mem = self._add(
            "养过一只三花猫（叫咪咪），是2019年从朋友那儿抱来的",
            importance=0.7, subject="三花猫",
        )
        self.store.conn.execute(
            "UPDATE memories SET stability=4.0, updated_at=? WHERE id=?",
            (self.now - 3 * DAY, mem.id),
        )
        self.store.conn.commit()
        self.store._invalidate_index()
        hits = recall(self.store, "s", "三花猫", ScoringConfig(flashback_rate=0), now=self.now)
        hit = [h for h in hits if h.memory.id == mem.id][0]
        self.assertLess(hit.r, 0.55)
        self.assertNotIn("咪咪", hit.text)
        self.assertLess(len(hit.text), len(mem.text))

    def test_hits_capped_by_max_items(self):
        for i in range(30):
            self._add(f"喜欢第{i}种咖啡", kind="preference", importance=0.8, subject=f"咖啡{i}")
        hits = recall(
            self.store, "s", "咖啡", ScoringConfig(max_items=5, flashback_rate=0),
            now=self.now,
        )
        self.assertLessEqual(len(hits), 5)


class ScoringConfigTest(unittest.TestCase):
    def test_defaults_are_sane(self):
        cfg = ScoringConfig()
        self.assertGreater(cfg.max_items, 0)
        self.assertGreater(cfg.alpha, 1.0)
        self.assertTrue(0 <= cfg.flashback_rate < 1)


if __name__ == "__main__":
    unittest.main()
