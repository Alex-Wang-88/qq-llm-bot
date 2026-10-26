"""Remove fabricated sticker markers from stored conversation history.

The model once invented "&&search&&" (treating a search verb as a sticker
category). That marker stayed in history and measurably starved tool calls on
later turns: 1/3 tool calls with it present versus 3/3 after stripping.

Stored assistant content is a list of content parts, not a plain string, so both
shapes are handled. Stop AstrBot before --apply, otherwise in-memory conversation
state can overwrite the row afterwards. Default is a dry run.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import sqlite3

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data/data_v4.db"
MARKER = re.compile(r"&&[\w-]+&&")


def clean_content(content):
    """Return (text, changed) after stripping sticker markers."""
    if isinstance(content, list):
        raw = "".join(
            str(part["text"]) for part in content
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    elif isinstance(content, str):
        raw = content
    else:
        return content, False
    if not MARKER.search(raw):
        return content, False
    return MARKER.sub("", raw).strip(), True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="write changes (default: dry run)")
    args = parser.parse_args()

    with sqlite3.connect(str(DB)) as db:
        plan = []
        for conv_id, content in db.execute("select inner_conversation_id, content from conversations"):
            if not content:
                continue
            try:
                history = json.loads(content)
            except ValueError:
                continue
            if not isinstance(history, list):
                continue
            kept, touched = [], 0
            for message in history:
                if isinstance(message, dict) and message.get("role") == "assistant":
                    text, changed = clean_content(message.get("content"))
                    if changed:
                        touched += 1
                        if not text:
                            continue  # drop a turn that was purely a marker
                        message["content"] = text
                kept.append(message)
            if touched:
                plan.append((conv_id, touched, json.dumps(kept, ensure_ascii=False)))

        print(f"conversations with markers: {len(plan)}")
        for conv_id, touched, payload in plan:
            print(f"  conversation {conv_id}: {touched} assistant turns -> {len(payload)} bytes")
        if not args.apply:
            print("dry run; re-run with --apply after stopping the bot")
            return

        backup = DB.with_name(DB.name + ".bak-markers")
        if not backup.exists():
            shutil.copy2(DB, backup)
            print(f"backup written: {backup.name}")
        for conv_id, _touched, payload in plan:
            db.execute(
                "update conversations set content=? where inner_conversation_id=?",
                (payload, conv_id),
            )
        db.commit()
        print(f"updated {len(plan)} conversations")


if __name__ == "__main__":
    main()
