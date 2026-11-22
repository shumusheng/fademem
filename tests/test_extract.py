import unittest

from fademem.extract import extract, redact


def kinds(cands):
    return {c.kind for c in cands}


def texts(cands):
    return [c.text for c in cands]


class ExtractTest(unittest.TestCase):
    def test_name(self):
        cands = extract("你好，我叫阿棠")
        self.assertIn("identity", kinds(cands))
        self.assertIn("叫阿棠", texts(cands))

    def test_name_with_you_can_call_me(self):
        cands = extract("你可以叫我小林，也可以叫林工")
        self.assertTrue(any("小林" in t for t in texts(cands)))

    def test_preference(self):
        cands = extract("我特别喜欢喝黑咖啡，不加糖")
        self.assertIn("preference", kinds(cands))

    def test_dislike_and_fear(self):
        a = extract("我讨厌下雨天")
        b = extract("我有点怕黑")
        self.assertIn("dislike", kinds(a))
        self.assertIn("fear", kinds(b))

    def test_location_and_occupation(self):
        a = extract("我住在城西的老小区")
        b = extract("我在一家杂志社上班")
        self.assertIn("location", kinds(a))
        self.assertIn("occupation", kinds(b))

    def test_pet(self):
        cands = extract("我养了一只橘猫叫年糕")
        self.assertIn("pet", kinds(cands))

    def test_relation(self):
        cands = extract("我女朋友最近出差了")
        self.assertIn("relation", kinds(cands))

    def test_event(self):
        cands = extract("昨天去医院看牙了，医生说有一颗要拔")
        self.assertIn("event", kinds(cands))

    def test_generic_fallback(self):
        cands = extract("我这几天一直在想要不要换工作")
        self.assertTrue(cands)
        self.assertEqual(cands[0].kind, "fact")
        self.assertLess(cands[0].importance, 0.45)

    def test_question_only_gets_nothing(self):
        self.assertEqual(extract("你在干嘛？"), [])
        self.assertEqual(extract("今天几号"), [])

    def test_too_short_ignored(self):
        self.assertEqual(extract("嗯"), [])

    def test_emotion_increases_salience(self):
        plain = extract("我在城西上班")
        emotional = extract("我在城西上班，真的好累，每天都很烦")
        self.assertGreater(emotional[0].salience, plain[0].salience)

    def test_text_has_no_pronoun(self):
        """记忆存的是谓语短语，不带人称代词，避免渲染时人混乱。"""
        for cand in extract("我叫阿棠，我住在城西，我喜欢黑咖啡"):
            self.assertFalse(cand.text.startswith("我"))

    def test_assistant_role_lowered(self):
        user = extract("我答应你明天早点回来", role="user")
        assistant = extract("我答应你明天早点回来", role="assistant")
        self.assertTrue(assistant)
        self.assertGreater(user[0].importance, assistant[0].importance)

    def test_candidates_capped(self):
        text = "我叫A我住在B我喜欢C我讨厌D我怕E我养了一只猫我女朋友在F"
        self.assertLessEqual(len(extract(text, max_items=3)), 3)


class RedactTest(unittest.TestCase):
    def test_phone(self):
        self.assertIn("手机号", redact("我的电话是13812345678"))

    def test_id_card(self):
        self.assertIn("证件号", redact("身份证 110101199003071234"))

    def test_email(self):
        self.assertIn("邮箱", redact("发我邮箱 a.b@example.com"))

    def test_password(self):
        out = redact("密码是 hunter2")
        self.assertNotIn("hunter2", out)

    def test_plain_text_untouched(self):
        self.assertEqual(redact("喜欢黑咖啡"), "喜欢黑咖啡")

    def test_extract_redacts(self):
        cands = extract("我叫阿棠，手机13812345678")
        joined = " ".join(c.text for c in cands)
        self.assertNotIn("13812345678", joined)


if __name__ == "__main__":
    unittest.main()
