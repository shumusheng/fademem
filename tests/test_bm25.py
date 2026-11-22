import unittest

from fademem import bm25


class TokenizeTest(unittest.TestCase):
    def test_chinese_produces_bigrams(self):
        tokens = bm25.tokenize("黑咖啡")
        self.assertIn("咖啡", tokens)
        self.assertIn("黑咖", tokens)

    def test_stopwords_dropped(self):
        self.assertNotIn("的", bm25.tokenize("我的猫"))

    def test_english_words(self):
        tokens = bm25.tokenize("I love Espresso")
        self.assertIn("love", tokens)
        self.assertIn("espresso", tokens)

    def test_mixed_script(self):
        tokens = bm25.tokenize("用Python写的脚本")
        self.assertIn("python", tokens)
        self.assertIn("脚本", tokens)

    def test_empty(self):
        self.assertEqual(bm25.tokenize(""), [])


class BM25Test(unittest.TestCase):
    def setUp(self):
        self.docs = [
            "喜欢黑咖啡，不加糖",
            "养了一只橘猫叫年糕",
            "在城西的杂志社上班",
        ]
        self.index = bm25.BM25([bm25.tokenize(d) for d in self.docs])

    def test_ranks_relevant_document_first(self):
        scores = self.index.scores(bm25.tokenize("咖啡"))
        self.assertEqual(scores.index(max(scores)), 0)
        self.assertGreater(max(scores), 0)

    def test_cat_query(self):
        scores = self.index.scores(bm25.tokenize("那只猫"))
        self.assertEqual(scores.index(max(scores)), 1)

    def test_unrelated_query_scores_zero(self):
        scores = self.index.scores(bm25.tokenize("量子力学"))
        self.assertEqual(max(scores), 0.0)

    def test_empty_corpus_is_safe(self):
        empty = bm25.BM25([])
        self.assertEqual(empty.scores(["x"]), [])


if __name__ == "__main__":
    unittest.main()
