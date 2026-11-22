import unittest

from fademem.persona import Persona, Speech, load_persona
from fademem.prompt import build_system_block, render_memory, render_persona
from fademem.prompt_style import StyleConfig, deslop, render_style
from fademem.retrieve import Hit
from fademem.store import Memory


def make_memory(mid=1, text="叫阿棠", kind="identity"):
    return Memory(
        id=mid, session="s", kind=kind, subject="x", text=text, canonical="c",
        importance=0.9, salience=0.2, stability=30.0,
        created_at=0.0, updated_at=0.0, access_count=0,
    )


def make_hit(text="叫阿棠", r=0.9, state="vivid", spontaneous=False, mid=1):
    return Hit(memory=make_memory(mid, text), score=1.0, r=r, state=state,
               text=text, spontaneous=spontaneous)


class PersonaRenderTest(unittest.TestCase):
    def setUp(self):
        self.persona = Persona(
            name="沈知夏",
            description="二十七岁，画图的。",
            personality="嘴硬心软。",
            relationship="认识三个月的邻居。",
            scenario="晚上十一点，在楼道里找钥匙。",
            speech=Speech(address="你", sentence_length=(4, 20), verbal_tics=["……行吧"]),
            boundaries=["不主动说自己的过去"],
        )

    def test_contains_name_and_role_lock(self):
        out = render_persona(self.persona)
        self.assertIn("沈知夏", out)
        self.assertIn("不跳出这个身份", out)

    def test_relationship_surfaces(self):
        self.assertIn("认识三个月的邻居", render_persona(self.persona))

    def test_style_mentions_verbal_tics(self):
        out = render_style(self.persona, StyleConfig())
        self.assertIn("……行吧", out)

    def test_style_can_be_disabled(self):
        self.assertEqual(render_style(self.persona, StyleConfig(enabled=False)), "")

    def test_banned_phrases_listed(self):
        out = render_style(self.persona, StyleConfig())
        self.assertIn("作为一个AI", out)
        self.assertIn("希望能帮到你", out)

    def test_banned_phrases_can_be_omitted(self):
        out = render_style(self.persona, StyleConfig(include_banned=False))
        self.assertNotIn("希望能帮到你", out)

    def test_full_block_order(self):
        block = build_system_block(self.persona, [make_hit()], StyleConfig())
        self.assertLess(block.index("【你是谁】"), block.index("【你记得的事"))
        self.assertLess(block.index("【你记得的事"), block.index("【说话方式】"))


class MemoryRenderTest(unittest.TestCase):
    def test_state_label_appears(self):
        out = render_memory([make_hit("叫阿棠", r=0.9, state="vivid")])
        self.assertIn("很清楚", out)
        self.assertIn("叫阿棠", out)

    def test_faint_memory_marked(self):
        out = render_memory([make_hit("讨厌下雨", r=0.1, state="faint")])
        self.assertIn("只剩一点印象", out)

    def test_spontaneous_marked(self):
        out = render_memory([make_hit("说过讨厌下雨", r=0.3, state="hazy", spontaneous=True)])
        self.assertIn("突然想起来", out)

    def test_empty(self):
        self.assertEqual(render_memory([]), "")

    def test_budget_truncates(self):
        hits = [make_hit(f"第{i}条记忆内容", mid=i) for i in range(60)]
        out = render_memory(hits, budget=200)
        self.assertLess(len(out), 700)
        self.assertNotIn("第59条", out)

    def test_no_meta_language_instruction(self):
        """记忆块必须明确禁止"我记得你说过"这种出戏表达。"""
        block = build_system_block(
            Persona(name="甲"), [make_hit()], StyleConfig()
        )
        self.assertIn("根据我的记忆", block)


class DeslopTest(unittest.TestCase):
    def test_strips_opening_politeness(self):
        self.assertEqual(deslop("当然可以！我在这儿。"), "我在这儿。")

    def test_strips_ai_disclaimer(self):
        out = deslop("作为一个AI，我不能说。")
        self.assertNotIn("作为一个AI", out)

    def test_strips_closing_offer(self):
        out = deslop("行吧。希望能帮到你。")
        self.assertNotIn("希望能帮到你", out)

    def test_removes_markdown_bold(self):
        self.assertEqual(deslop("**嗯**。"), "嗯。")

    def test_flattens_bullet_list(self):
        out = deslop("- 第一件事\n- 第二件事\n- 第三件事")
        self.assertNotIn("- ", out)

    def test_plain_text_untouched(self):
        self.assertEqual(deslop("嗯。"), "嗯。")

    def test_repeated_dashes_collapsed(self):
        out = deslop("这——那样——还有——最后")
        self.assertLessEqual(out.count("——"), 1)


class PersonaLoadTest(unittest.TestCase):
    def test_missing_file_falls_back(self):
        persona = load_persona("/nonexistent/path/persona.json")
        self.assertEqual(persona.name, "未命名")

    def test_none_path_falls_back(self):
        self.assertIsNotNone(load_persona(None))


if __name__ == "__main__":
    unittest.main()
