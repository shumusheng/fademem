"""CLI：重点验证 -c 指定自定义配置时，角色卡和记忆库跟着配置走。"""

import argparse
import json
import os
import tempfile
import unittest
from unittest import mock

from fademem import cli
from fademem.config import DEFAULT_CONFIG, default_paths, write_default_config


def load(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class WriteDefaultConfigTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fademem_cfg_")
        self.path = os.path.join(self.tmp, "app", "config.json")

    def test_default_paths_inside_home(self):
        cfg_path = write_default_config(self.path)
        data = load(cfg_path)
        self.assertEqual(data["memory"]["db"], default_paths()["db"])
        self.assertEqual(data["persona"], default_paths()["persona"])

    def test_localize_puts_files_beside_config(self):
        cfg_path = write_default_config(self.path, localize=True)
        base = os.path.dirname(os.path.abspath(cfg_path))
        data = load(cfg_path)
        self.assertEqual(data["memory"]["db"], os.path.join(base, "memory.db"))
        self.assertEqual(data["persona"], os.path.join(base, "persona.json"))

    def test_localize_does_not_mutate_defaults(self):
        before = json.dumps(DEFAULT_CONFIG, sort_keys=True)
        write_default_config(self.path, localize=True)
        self.assertEqual(json.dumps(DEFAULT_CONFIG, sort_keys=True), before)

    def test_existing_file_not_overwritten(self):
        write_default_config(self.path)
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write('{"server": {"port": 1234}}')
        write_default_config(self.path)
        data = load(self.path)
        self.assertEqual(data["server"]["port"], 1234)


class InitCommandTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="fademem_init_")
        self.cfg = os.path.join(self.tmp, "b.json")

    def _run(self, **kw):
        args = argparse.Namespace(config=kw.get("config"), persona=kw.get("persona"),
                                  force=kw.get("force", False))
        with mock.patch("builtins.print"):
            return cli.cmd_init(args)

    def test_custom_config_creates_persona_next_to_it(self):
        self.assertEqual(self._run(config=self.cfg), 0)
        self.assertTrue(os.path.exists(os.path.join(self.tmp, "persona.json")))
        data = load(self.cfg)
        self.assertEqual(
            os.path.abspath(data["persona"]),
            os.path.abspath(os.path.join(self.tmp, "persona.json")),
        )
        self.assertEqual(
            os.path.abspath(data["memory"]["db"]),
            os.path.abspath(os.path.join(self.tmp, "memory.db")),
        )

    def test_explicit_persona_wins(self):
        target = os.path.join(self.tmp, "who.json")
        self._run(config=self.cfg, persona=target)
        self.assertTrue(os.path.exists(target))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "persona.json")))

    def test_persona_template_has_core_fields(self):
        self._run(config=self.cfg)
        card = load(os.path.join(self.tmp, "persona.json"))
        for key in ("name", "description", "personality", "relationship", "first_mes"):
            self.assertIn(key, card)
            self.assertTrue(str(card[key]).strip(), f"{key} 不能为空")


if __name__ == "__main__":
    unittest.main()
