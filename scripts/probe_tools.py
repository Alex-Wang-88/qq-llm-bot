"""Read-only probe: inspect live tool registry entries as packages.

Loads the local plugins the way AstrBot does (as packages, so relative imports
work) and reports the `active` flag that get_func() would observe.
"""
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data/plugins"))

from astrbot.core.provider.func_tool_manager import FunctionToolManager  # noqa: E402
from astrbot.core.star.register import star_handler  # noqa: E402

out = {"imports": {}}
for package in (
    "astrbot_plugin_searxng_search",
    "astrbot_plugin_harness_manager",
    "astrbot_plugin_chat_natural",
):
    try:
        __import__(f"{package}.main")
        out["imports"][package] = "ok"
    except Exception as exc:  # noqa: BLE001
        out["imports"][package] = f"{type(exc).__name__}: {exc}"

registry = getattr(star_handler, "llm_tools", None)
func_list = list(getattr(registry, "func_list", []))
out["registry"] = [
    {
        "name": getattr(f, "name", "?"),
        "active": getattr(f, "active", "<missing>"),
        "module": getattr(f, "handler_module_path", None),
    }
    for f in func_list
]

manager = FunctionToolManager()
manager.func_list.extend(func_list)

persona_tools = []
with sqlite3.connect(str(ROOT / "data/data_v4.db")) as db:
    row = db.execute(
        "select tools from personas where persona_id=?", ("千问娘",)
    ).fetchone()
if row and row[0]:
    persona_tools = json.loads(row[0])
out["persona_tools"] = persona_tools
out["resolved"] = []
for name in persona_tools:
    tool = manager.get_func(name)
    out["resolved"].append(
        {
            "name": name,
            "found": tool is not None,
            "active": getattr(tool, "active", None) if tool else None,
            "module": getattr(tool, "handler_module_path", None) if tool else None,
        }
    )

target = ROOT / "data/logs/probe-tools.json"
target.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
