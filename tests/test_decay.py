import math
import unittest

from fademem import decay


class RetrievabilityTest(unittest.TestCase):
    def test_starts_at_one(self):
        self.assertAlmostEqual(decay.retrievability(0, 5), 1.0)

    def test_decays_monotonically(self):
        values = [decay.retrievability(d, 5.0) for d in (0, 1, 2, 5, 10, 30)]
        self.assertEqual(values, sorted(values, reverse=True))
        self.assertGreater(values[0], values[-1])

    def test_half_life_means_half(self):
        s = 4.0
        self.assertAlmostEqual(decay.retrievability(decay.half_life(s), s), 0.5)

    def test_zero_stability_is_forgotten(self):
        self.assertEqual(decay.retrievability(1, 0), 0.0)

    def test_larger_stability_forgets_slower(self):
        self.assertGreater(
            decay.retrievability(7, 20), decay.retrievability(7, 5)
        )


class StabilityTest(unittest.TestCase):
    def test_importance_raises_initial_stability(self):
        low = decay.initial_stability(0.2)
        high = decay.initial_stability(0.95)
        self.assertGreater(high, low)

    def test_salience_raises_initial_stability(self):
        plain = decay.initial_stability(0.5, salience=0.0)
        emotional = decay.initial_stability(0.5, salience=1.0)
        self.assertGreater(emotional, plain * 1.5)

    def test_spacing_effect(self):
        """越模糊的时候想起来，记得越牢。"""
        fresh = decay.reinforce(10.0, r_now=0.99)
        hazy = decay.reinforce(10.0, r_now=0.20)
        self.assertGreater(hazy, fresh)
        self.assertAlmostEqual(fresh, 10.0, delta=0.2)

    def test_reinforce_is_capped(self):
        out = decay.reinforce(3000.0, 0.0, cap=3650.0)
        self.assertLessEqual(out, 3650.0)

    def test_repeated_recall_compounds(self):
        s = decay.initial_stability(0.5)
        for _ in range(3):
            s = decay.reinforce(s, r_now=0.3)
        self.assertGreater(s, decay.initial_stability(0.5) * 3)


class StateTest(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(decay.decay_state(0.9), "vivid")
        self.assertEqual(decay.decay_state(0.6), "clear")
        self.assertEqual(decay.decay_state(0.3), "hazy")
        self.assertEqual(decay.decay_state(0.1), "faint")
        self.assertEqual(decay.decay_state(0.01), "gone")

    def test_all_states_have_labels(self):
        for state in decay.STATE_ORDER:
            self.assertIn(state, decay.STATE_LABEL)


class FuzzTest(unittest.TestCase):
    def test_clear_memory_kept_verbatim(self):
        text = "住在城西的阳光小区"
        self.assertEqual(decay.fuzz_text(text, 0.9), text)

    def test_hazy_memory_loses_detail(self):
        text = "养过一只三花猫（叫咪咪），是2019年从朋友那儿抱来的"
        out = decay.fuzz_text(text, 0.2)
        self.assertLess(len(out), len(text))
        self.assertNotIn("2019", out)
        self.assertNotIn("咪咪", out)

    def test_fuzz_is_deterministic(self):
        text = "上周三下午在公司楼下的咖啡馆见了客户"
        a = decay.fuzz_text(text, 0.3, seed=7)
        b = decay.fuzz_text(text, 0.3, seed=7)
        self.assertEqual(a, b)

    def test_fuzz_never_returns_empty(self):
        for r in (0.0, 0.1, 0.3, 0.5):
            self.assertTrue(decay.fuzz_text("喜欢黑咖啡", r).strip())


if __name__ == "__main__":
    unittest.main()
