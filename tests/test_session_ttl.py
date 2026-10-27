"""闲置自动开新会话（TTL）的行为与边界。

背景：QQ 里"新开聊天"不等于新会话，AstrBot 按 UMO 复用同一个 conversation。
这里只测判定与清理动作本身，不启动 AstrBot。
"""
import asyncio
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


natural = load("natural_plugin_session", "data/plugins/astrbot_plugin_chat_natural/main.py")
HOUR = 3600.0


class PureDecisionTests(unittest.TestCase):
    def test_first_contact_never_resets(self):
        self.assertFalse(natural.should_reset_session(1000.0, None, 8 * HOUR))

    def test_inside_window_keeps_context(self):
        self.assertFalse(natural.should_reset_session(1000.0 + 7 * HOUR, 1000.0, 8 * HOUR))

    def test_past_window_resets(self):
        self.assertTrue(natural.should_reset_session(1000.0 + 9 * HOUR, 1000.0, 8 * HOUR))

    def test_exactly_at_threshold_keeps_context(self):
        self.assertFalse(natural.should_reset_session(1000.0 + 8 * HOUR, 1000.0, 8 * HOUR))

    def test_zero_disables_feature(self):
        self.assertFalse(natural.should_reset_session(10 ** 9, 0.0, 0.0))

    def test_idle_seconds_reads_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session_policy.json"
            path.write_text(json.dumps({"idle_reset_hours": 2}), encoding="utf-8")
            with patch.object(natural, "SESSION_CONFIG", path):
                self.assertEqual(natural.idle_seconds(), 2 * HOUR)

    def test_idle_seconds_falls_back_when_missing(self):
        with patch.object(natural, "SESSION_CONFIG", Path(tempfile.gettempdir()) / "no-such-policy.json"):
            self.assertEqual(natural.idle_seconds(), natural.DEFAULT_IDLE_HOURS * HOUR)

    def test_idle_seconds_tolerates_garbage(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session_policy.json"
            path.write_text("{not json", encoding="utf-8")
            with patch.object(natural, "SESSION_CONFIG", path):
                self.assertEqual(natural.idle_seconds(), natural.DEFAULT_IDLE_HOURS * HOUR)
            path.write_text(json.dumps({"idle_reset_hours": "abc"}), encoding="utf-8")
            with patch.object(natural, "SESSION_CONFIG", path):
                self.assertEqual(natural.idle_seconds(), natural.DEFAULT_IDLE_HOURS * HOUR)


class StateFileTests(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "state.json"
            natural.save_session_state(path, {"umo:a": 123.5, "umo:b": 7})
            self.assertEqual(natural.load_session_state(path), {"umo:a": 123.5, "umo:b": 7.0})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["umo:b"], 7)

    def test_corrupt_file_reads_as_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            path.write_text("[1,2,3]", encoding="utf-8")
            self.assertEqual(natural.load_session_state(path), {})
            path.write_text("{}", encoding="utf-8")
            self.assertEqual(natural.load_session_state(path), {})

    def test_non_numeric_values_are_dropped(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            path.write_text(json.dumps({"a": "x", "b": 5}), encoding="utf-8")
            self.assertEqual(natural.load_session_state(path), {"b": 5.0})

    def test_missing_file_reads_as_empty(self):
        self.assertEqual(natural.load_session_state(Path(tempfile.gettempdir()) / "nope-state.json"), {})


class FakeManager:
    def __init__(self, conversation_id="cid-1", blow_up=False):
        self.conversation_id = conversation_id
        self.blow_up = blow_up
        self.cleared = []

    async def get_curr_conversation_id(self, umo):
        if self.blow_up:
            raise RuntimeError("db locked")
        return self.conversation_id

    async def update_conversation(self, umo, conversation_id, history=None):
        self.cleared.append((umo, conversation_id, history))


def make_plugin(manager, state):
    plugin = object.__new__(natural.NaturalChatPlugin)
    plugin.context = SimpleNamespace(conversation_manager=manager)
    plugin._last_active = state
    plugin._meme_history = {}
    return plugin


def make_event(umo="qq:FriendMessage:u1"):
    extras = {}
    return SimpleNamespace(unified_msg_origin=umo, set_extra=lambda k, v: extras.__setitem__(k, v),
                           get_extra=extras.get), extras


class ResetActionTests(unittest.TestCase):
    def run_reset(self, plugin, event, req):
        asyncio.run(plugin._apply_idle_reset(event, req))

    def test_idle_session_is_cleared(self):
        manager = FakeManager()
        now = time.time()  # _apply_idle_reset 内部取真实时钟，测试数据也要用真实时间
        plugin = make_plugin(manager, {"qq:FriendMessage:u1": now - 9 * HOUR})
        (event, extras), req = make_event(), SimpleNamespace(contexts=[{"role": "user", "content": "旧"}] * 4)
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(natural, "SESSION_STATE", Path(tmp) / "state.json"):
            self.run_reset(plugin, event, req)
        self.assertEqual(req.contexts, [])
        self.assertEqual(manager.cleared, [("qq:FriendMessage:u1", "cid-1", [])])
        self.assertTrue(extras.get("chat_natural_session_reset"))
        self.assertGreaterEqual(plugin._last_active["qq:FriendMessage:u1"], now)

    def test_recent_session_is_untouched(self):
        manager = FakeManager()
        plugin = make_plugin(manager, {"qq:FriendMessage:u1": time.time() - 60})
        (event, extras), req = make_event(), SimpleNamespace(contexts=[{"role": "user", "content": "旧"}])
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(natural, "SESSION_STATE", Path(tmp) / "state.json"):
            self.run_reset(plugin, event, req)
        self.assertEqual(req.contexts, [{"role": "user", "content": "旧"}])
        self.assertEqual(manager.cleared, [])
        self.assertFalse(extras.get("chat_natural_session_reset", False))

    def test_db_failure_still_clears_request_context(self):
        manager = FakeManager(blow_up=True)
        plugin = make_plugin(manager, {"qq:FriendMessage:u1": time.time() - 30 * HOUR})
        (event, extras), req = make_event(), SimpleNamespace(contexts=[{"role": "user", "content": "旧"}])
        with tempfile.TemporaryDirectory() as tmp, \
                patch.object(natural, "SESSION_STATE", Path(tmp) / "state.json"):
            self.run_reset(plugin, event, req)
        self.assertEqual(req.contexts, [])
        self.assertTrue(extras.get("chat_natural_session_reset"))

    def test_first_contact_records_state_without_reset(self):
        manager = FakeManager()
        plugin = make_plugin(manager, {})
        (event, extras), req = make_event("qq:FriendMessage:new"), SimpleNamespace(contexts=[{"role": "user", "content": "第一句"}])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            with patch.object(natural, "SESSION_STATE", path):
                self.run_reset(plugin, event, req)
            saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(req.contexts, [{"role": "user", "content": "第一句"}])
        self.assertEqual(manager.cleared, [])
        self.assertFalse(extras.get("chat_natural_session_reset", False))
        self.assertIn("qq:FriendMessage:new", saved)

    def test_state_file_is_trimmed(self):
        manager = FakeManager()
        plugin = make_plugin(manager, {f"umo:{i}": float(i) for i in range(natural.SESSION_STATE_LIMIT + 5)})
        event, _extras = make_event("umo:new")
        req = SimpleNamespace(contexts=[])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "state.json"
            with patch.object(natural, "SESSION_STATE", path):
                self.run_reset(plugin, event, req)
            saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertLessEqual(len(saved), natural.SESSION_STATE_LIMIT)
        self.assertIn("umo:new", saved)


if __name__ == "__main__":
    unittest.main()
