"""Render the exact reply the QQ command would produce, without touching QQ.

Builds the real plugin class, the real adapter and the real routing store, then
runs `/助手 会话` and `/助手 额度` through the same code path the bot uses.
"""
import asyncio
import importlib.util
import io
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PLUGIN = ROOT / "data" / "plugins" / "astrbot_plugin_harness_manager" / "main.py"
OUT = ROOT / "data" / "logs" / "probe-harness-command.txt"
HARNESS_CONFIG = ROOT / "data" / "config" / "astrbot_plugin_harness_manager_config.json"


def owner_umo():
    """Whitelisted private chat UMO; a real openid never belongs in the source."""
    override = os.environ.get("HARNESS_OWNER_UMO")
    if override:
        return override
    try:
        allowed = json.loads(HARNESS_CONFIG.read_text(encoding="utf-8")).get("allowed_private_sessions") or []
    except (OSError, ValueError):
        allowed = []
    if not allowed:
        raise SystemExit("Set HARNESS_OWNER_UMO or configure allowed_private_sessions first.")
    return str(allowed[0])


OWNER = owner_umo()


def load_plugin_class():
    spec = importlib.util.spec_from_file_location("harness_plugin_probe", PLUGIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.HarnessManager


def event(text):
    return SimpleNamespace(unified_msg_origin=OWNER, get_message_str=lambda: text,
                           plain_result=lambda value: value)



async def run(plugin_class, command):
    plugin = plugin_class.__new__(plugin_class)
    plugin.config = {"allowed_private_sessions": [OWNER], "exclusive_codex_sessions": []}
    from assistant_harness.adapters import CodexAdapter
    from assistant_harness.store import SessionStore
    plugin.adapters = {"codex": CodexAdapter(exclusive_sessions=[])}
    plugin.store = SessionStore(ROOT / "data/harness/state.sqlite3")
    try:
        replies = [value async for value in plugin.manage(event(command))]
        return replies
    finally:
        await plugin.adapters["codex"].close()
        plugin.store.close()


async def main():
    plugin_class = load_plugin_class()
    handle = io.open(OUT, "w", encoding="utf-8")
    try:
        for command in ("/助手 会话", "/助手 读取 1", "/助手 额度"):
            replies = await run(plugin_class, command)
            handle.write(f"### {command}\n" + "\n---\n".join(replies) + "\n\n")
            handle.flush()
            print(f"{command} -> {len(replies)} reply, {sum(len(r) for r in replies)} chars")
    finally:
        handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
