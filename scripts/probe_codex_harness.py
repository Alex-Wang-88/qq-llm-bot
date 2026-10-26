"""Offline check for the Codex App Server bridge.

Resolves the CLI exactly like the plugin does, then exercises the same three
read-only calls the QQ commands use: thread/list, thread/read, quota.
Never sends a turn, never touches credentials.
"""
import asyncio
import io
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assistant_harness.codex_rpc import CodexRpc, extension_candidates, resolve_codex_executable
from assistant_harness.adapters import CodexAdapter, format_quota

OUT = ROOT / "data" / "logs" / "probe-codex-harness.jsonl"


def note(handle, kind, **fields):
    handle.write(json.dumps({"check": kind, **fields}, ensure_ascii=False) + "\n")
    handle.flush()


async def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    handle = io.open(OUT, "w", encoding="utf-8")
    try:
        executable = resolve_codex_executable()
        note(handle, "resolve", path=str(executable) if executable else None,
             extensions=[str(p) for p in extension_candidates()])
        if not executable:
            print("RESOLVE=FAILED")
            return 1
        print(f"RESOLVE=OK size={executable.stat().st_size}")

        rpc = CodexRpc(timeout=40)
        print("COMMAND=" + " ".join(rpc.command))
        try:
            await rpc.start()
            note(handle, "initialize", ok=True, result=rpc.initialized)
            print("INITIALIZE=OK")

            sessions = await CodexAdapter(rpc).list_sessions(limit=5)
            note(handle, "thread/list", state=str(sessions.state), data=sessions.data, reason=sessions.reason)
            print(f"THREAD_LIST=OK count={len(sessions.data or [])}")

            if sessions.data:
                target = sessions.data[0]
                detail = await CodexAdapter(rpc).read_session(target["native_id"])
                note(handle, "thread/read", state=str(detail.state), data=detail.data, reason=detail.reason)
                print(f"THREAD_READ=OK turns={len((detail.data or {}).get('recent_turns', []))}")

            quota = await CodexAdapter(rpc).quota()
            note(handle, "quota", state=str(quota.state), data=quota.data, reason=quota.reason)
            if quota.state != "Supported":
                print(f"QUOTA=DEGRADED state={quota.state}")
                print("QUOTA_TEXT=" + quota.reason.replace("\n", " | "))
            else:
                print("QUOTA=OK windows=%d" % len((quota.data or {}).get("windows", [])))
                print("QUOTA_TEXT=" + format_quota(quota.data).replace("\n", " | "))
        except Exception as exc:
            note(handle, "failure", error_type=type(exc).__name__, error=str(exc),
                 stderr=list(rpc.stderr))
            print(f"FAILED={type(exc).__name__}: {exc}")
            return 1
        finally:
            await rpc.close()
        return 0
    finally:
        handle.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
