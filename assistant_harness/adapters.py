"""Capability-aware adapters. Separate desktop observation from write ownership."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum
from zoneinfo import ZoneInfo
import math

from .codex_rpc import CodexRpc, RpcError


class Capability(StrEnum):
    SUPPORTED="Supported"
    UNSUPPORTED="Unsupported"
    AUTH_REQUIRED="AuthenticationRequired"
    UNAVAILABLE="TemporarilyUnavailable"
    ERROR="Error"


@dataclass
class Result:
    state: Capability
    data: object = None
    reason: str = ""


def normalize_quota(raw, collected_at=None):
    buckets = raw.get("rateLimitsByLimitId")
    if not buckets:
        value=raw.get("rateLimits")
        buckets={value.get("limitId") or "codex":value} if value else {}
    windows=[]
    for name,bucket in buckets.items():
        for key in ("primary","secondary"):
            window=bucket.get(key)
            if not window:continue
            used=window.get("usedPercent")
            if used is None:continue
            windows.append({"category":name,"window_minutes":window.get("windowDurationMins"),
                            "used_percent":used,"remaining_percent":max(0,min(100,100-used)),
                            "resets_at":window.get("resetsAt")})
    return {"source":"codex account/rateLimits/read","collected_at":collected_at or datetime.now(timezone.utc).isoformat(),"windows":windows}


def format_quota(data, now=None, window_minutes=None):
    now = now or datetime.now(timezone.utc)
    lines=[]
    for window in data['windows']:
        minutes=window.get('window_minutes')
        if window_minutes is not None and minutes != window_minutes:continue
        duration=(f"{minutes//1440} 天" if minutes and minutes%1440==0 else f"{minutes//60} 小时" if minutes and minutes%60==0 else f"{minutes} 分钟" if minutes else "未知")
        stamp=window.get('resets_at')
        reset=datetime.fromtimestamp(stamp,ZoneInfo('Asia/Shanghai')).strftime('%m-%d %H:%M') if stamp else '接口未返回'
        line=f"{window['category']}：{duration}窗口剩余 {window['remaining_percent']}%（已用 {window['used_percent']}%），重置于 {reset}（北京时间）"
        if isinstance(stamp, (int, float)) and not isinstance(stamp, bool):
            remaining=stamp-now.timestamp()
            if remaining > 0:
                total=math.ceil(remaining/60)
                line+=f"，距重置约 {total//60} 小时 {total%60} 分钟"
            else:
                line+='，重置时间已到或已过，请稍后重新查询'
        lines.append(line+'。')
    if not lines:lines=['接口未返回可用额度窗口。']
    try:
        collected=datetime.fromisoformat(data['collected_at'].replace('Z','+00:00'))
        collected=collected.astimezone(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')+'（北京时间）'
    except (ValueError, TypeError, KeyError, AttributeError):
        collected='未知'
    return '\n'.join(lines)+f"\n查询时间：{collected}"


class CodexAdapter:
    name="codex"
    def __init__(self, client=None, exclusive_sessions=()):
        self.client=client or CodexRpc()
        self.exclusive_sessions=set(exclusive_sessions)

    async def start(self): await self.client.start()
    async def close(self): await self.client.close()

    async def list_sessions(self, limit=20):
        result=await self.client.request("thread/list",{"limit":min(max(limit,1),50),"sortKey":"updated_at","sourceKinds":["appServer","cli","vscode"]})
        return Result(Capability.SUPPORTED,[{
            "harness":"codex","native_id":t["id"],"title":t.get("name") or t.get("preview", "")[:100],
            "project_path":t.get("cwd"),"updated_at":t.get("updatedAt"),
            "status":t.get("status",{}).get("type","unknown"),
            "status_scope":"independent_app_server","can_send":t["id"] in self.exclusive_sessions,
        } for t in result.get("data",[])])

    async def read_session(self, identity):
        metadata=await self.client.request("thread/read",{"threadId":identity,"includeTurns":False})
        turns=await self.client.request("thread/turns/list",{"threadId":identity,"limit":3,"sortDirection":"desc","itemsView":"full"})
        output=[]
        for turn in turns.get("data",[]):
            texts=[];finals=[]
            for item in turn.get("items",[]):
                if item.get("type")=="agentMessage" and item.get("text"):
                    texts.append(item["text"])
                    if item.get("phase")=="final_answer":finals.append(item["text"])
            # Prefer the actual terminal answer; commentary is progress, not a result.
            output.append({"turn_id":turn.get("id"),"status":turn.get("status"),"messages":finals or texts[-1:],"output_phase":"final_answer" if finals else "unknown","error":turn.get("error")})
        return Result(Capability.SUPPORTED,{"thread":{k:metadata.get("thread",{}).get(k) for k in ("id","name","cwd","status")},"recent_turns":output})

    async def quota(self):
        # The quota endpoint is remote, so it fails independently of the local
        # thread store: reaching it requires network access to chatgpt.com.
        try:
            raw=await self.client.request("account/rateLimits/read")
        except RpcError as exc:
            return Result(Capability.UNAVAILABLE,reason=f"额度接口未返回数据（{exc}）。会话列表仍可读取；额度需要本机可访问 chatgpt.com。")
        return Result(Capability.SUPPORTED,normalize_quota(raw))

    async def send(self, identity, text):
        if identity not in self.exclusive_sessions:
            return Result(Capability.UNSUPPORTED,reason="该会话尚未独占交接；本机桌面共享控制端点不可达，不能用独立 App Server 冒充桌面客户端写入。")
        current=await self.client.request('thread/read',{'threadId':identity,'includeTurns':False})
        state=(current.get('thread') or {}).get('status') or {}
        if state.get('type')=='active':
            return Result(Capability.UNSUPPORTED,reason='该会话当前正在执行；不能并发启动新回合。')
        # Explicitly handed-over sessions retain their native ID. No new thread
        # or fork is created. Approval remains on-request, not auto-approved.
        await self.client.request("thread/resume",{"threadId":identity,"excludeTurns":True,"approvalPolicy":"on-request","sandbox":"workspace-write"})
        result=await self.client.request("turn/start",{"threadId":identity,"input":[{"type":"text","text":text}]})
        return Result(Capability.SUPPORTED,{"state":"accepted","thread_id":identity,"turn":result.get("turn")})


class UnavailableAdapter:
    def __init__(self,name,reason,authentication_required=False,quota_reason=None):
        self.name=name;self.reason=reason
        self.state=Capability.AUTH_REQUIRED if authentication_required else Capability.UNSUPPORTED
        self.quota_reason=quota_reason

    # Same lifecycle surface as CodexAdapter so a missing backend degrades to a
    # readable reason instead of taking the whole plugin down at import time.
    async def start(self): return None
    async def close(self): return None
    async def list_sessions(self,limit=20): return Result(self.state,reason=self.reason)
    async def read_session(self,identity): return Result(self.state,reason=self.reason)
    async def send(self,identity,text): return Result(self.state,reason=self.reason)
    async def quota(self): return Result(Capability.UNSUPPORTED,reason=self.quota_reason) if self.quota_reason else Result(self.state,reason=self.reason)
