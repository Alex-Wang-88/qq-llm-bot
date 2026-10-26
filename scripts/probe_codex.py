"""Read-only official protocol probe; never creates/resumes/sends a turn."""
import asyncio
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from assistant_harness.codex_rpc import CodexRpc


async def main():
    client=CodexRpc(timeout=35)
    report={"collected_at":datetime.now(timezone.utc).isoformat(),"transport":"independent_stdio", "desktop_control":False}
    try:
        await client.start()
        report["initialize"]=client.initialized
        threads=await client.request("thread/list",{"limit":8,"sortKey":"updated_at","sourceKinds":["appServer","cli","vscode"]})
        report["threads"]=[{k:t.get(k) for k in ("id","name","cwd","updatedAt","status","source")} for t in threads.get("data",[])]
        quota=await client.request("account/rateLimits/read")
        report["quota"]=quota
        if report["threads"]:
            sample=report["threads"][-1]
            result=await client.request("thread/read",{"threadId":sample["id"],"includeTurns":False})
            report["metadata_read"]={"thread_id":sample["id"],"success":bool(result.get("thread"))}
    except Exception as exc:
        report["error"]={"type":type(exc).__name__,"message":str(exc)[:250]}
    finally:
        await client.close()
    target=ROOT/"data/harness/codex-probe.json"
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=="__main__":asyncio.run(main())
