import os
import shutil
import tempfile
import time
import unittest

from fademem.store import Candidate, Store


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.store = Store(os.path.join(self.dir, "m.db"))

    def tearDown(self):
        self.store.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def test_add_new(self):
        mem, created = self.store.add("s1", Candidate("identity", "叫阿棠", subject="阿棠", importance=0.9))
        self.assertTrue(created)
        self.assertEqual(mem.text, "叫阿棠")
        self.assertGreater(mem.stability, 0)
        self.assertEqual(mem.access_count, 0)

    def test_repeat_strengthens_instead_of_duplicating(self):
        cand = Candidate("identity", "叫阿棠", subject="阿棠", importance=0.9)
        first, created1 = self.store.add("s1", cand)
        second, created2 = self.store.add("s1", cand)
        self.assertTrue(created1)
        self.assertFalse(created2)
        self.assertEqual(len(self.store.list("s1")), 1)
        self.assertGreater(second.stability, first.stability)
        self.assertEqual(second.access_count, 1)

    def test_sessions_are_isolated(self):
        self.store.add("a", Candidate("fact", "喜欢黑咖啡", subject="咖啡"))
        self.store.add("b", Candidate("fact", "讨厌下雨", subject="雨"))
        self.assertEqual(len(self.store.list("a")), 1)
        self.assertEqual(len(self.store.list("b")), 1)

    def test_search_finds_relevant(self):
        self.store.add("s1", Candidate("preference", "喜欢黑咖啡", subject="黑咖啡"))
        self.store.add("s1", Candidate("pet", "养了一只橘猫", subject="猫"))
        hits = self.store.search("s1", "今天想喝咖啡")
        self.assertTrue(hits)
        self.assertIn("咖啡", hits[0][0].text)

    def test_forget_and_restore(self):
        mem, _ = self.store.add("s1", Candidate("fact", "住在城西", subject="城西"))
        self.assertTrue(self.store.forget(mem.id))
        self.assertEqual(len(self.store.list("s1")), 0)
        self.assertEqual(len(self.store.list("s1", include_archived=True)), 1)
        self.assertTrue(self.store.restore(mem.id))
        self.assertEqual(len(self.store.list("s1")), 1)

    def test_sweep_archives_forgotten(self):
        mem, _ = self.store.add("s1", Candidate("fact", "随口提过的一句话", importance=0.05))
        long_ago = time.time() - 400 * 86400
        self.store.conn.execute("UPDATE memories SET updated_at=? WHERE id=?", (long_ago, mem.id))
        self.store.conn.commit()
        self.store._invalidate_index()
        self.assertEqual(self.store.sweep("s1", forget_below=0.06), 1)
        self.assertEqual(len(self.store.list("s1")), 0)

    def test_sweep_keeps_fresh(self):
        self.store.add("s1", Candidate("preference", "喜欢黑咖啡", importance=0.9))
        self.assertEqual(self.store.sweep("s1", forget_below=0.06), 0)
        self.assertEqual(len(self.store.list("s1")), 1)

    def test_consolidate_merges_near_duplicates(self):
        self.store.add("s1", Candidate("preference", "喜欢喝黑咖啡不加糖"))
        self.store.add("s1", Candidate("preference", "喜欢喝黑咖啡，不加糖"))
        self.assertEqual(len(self.store.list("s1")), 2)
        result = self.store.consolidate("s1")
        self.assertEqual(result["merged"], 1)
        self.assertEqual(len(self.store.list("s1")), 1)

    def test_consolidate_ignores_unrelated(self):
        self.store.add("s1", Candidate("preference", "喜欢黑咖啡"))
        self.store.add("s1", Candidate("preference", "讨厌香菜"))
        self.assertEqual(self.store.consolidate("s1")["merged"], 0)

    def test_seen_turns(self):
        self.assertFalse(self.store.seen("s1", "abc"))
        self.store.mark_seen("s1", "abc")
        self.assertTrue(self.store.seen("s1", "abc"))
        self.assertFalse(self.store.seen("s2", "abc"))

    def test_stats(self):
        self.store.add("s1", Candidate("fact", "喜欢黑咖啡", subject="咖啡"))
        stats = self.store.stats()
        self.assertEqual(stats["memories"], 1)
        self.assertEqual(stats["sessions"], 1)


if __name__ == "__main__":
    unittest.main()
