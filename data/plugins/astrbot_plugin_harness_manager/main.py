"""Explicit private-chat controls; no shell execution or guessed task targets."""
import json
import asyncio
from pathlib import Path
import time
import re
from types import SimpleNamespace
import sys

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star
from astrbot.api.message_components import Plain
from astrbot.api.event import MessageChain

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from assistant_harness.adapters import Capability, CodexAdapter, UnavailableAdapter, format_quota
from assistant_harness.store import SessionStore
from assistant_harness.zcode import ZCodeAdapter
from assistant_security import redact_text,redact
from assistant_dialogue import is_harness_dispatch

_QUOTA = re.compile(r'额度|限额|用量|配额|quota|usage|rate.?limit', re.I)
_CODEX = re.compile(r'codex', re.I)
_RESET_QUESTION = re.compile(r'(?:重置|刷新).{0,16}(?:多久|多少|时间|几点|何时)|(?:多久|多少|几点|何时).{0,16}(?:重置|刷新)')
_FIVE_HOURS = re.compile(r'5\s*(?:小时|h(?:ours?)?)|五小时', re.I)
_SEVEN_DAYS = re.compile(r'7\s*(?:天|日|d(?:ays?)?)|七天|一周', re.I)
_QUOTA_FOLLOWUP = re.compile(r'^\s*(?:那|那么)?\s*(?:现在|目前|再查一下|再查一遍|再查查)?\s*(?:还剩多少|剩多少|用量多少|额度多少|呢)?[？?。！!\s]*$')
_TASK_STATUS = re.compile(r'(?:任务|会话|codex|harness).{0,20}(?:跑.{0,8}多久|运行.{0,8}多久|状态|进度)|(?:跑了多久|运行了多久)',re.I)


def _context_text(message):
    content=message.get('content', '')
    if isinstance(content,str):return content
    if isinstance(content,list):return ' '.join(part.get('text','') for part in content if isinstance(part,dict) and part.get('type')=='text')
    return ''


def is_live_codex_quota_question(text, contexts, recent_quota=False):
    """Route explicit Codex quota requests and closely related reset follow-ups."""
    if not isinstance(text,str) or text.lstrip().startswith('/'):
        return False
    if _CODEX.search(text) and (_QUOTA.search(text) or _RESET_QUESTION.search(text)):
        return True
    if not _RESET_QUESTION.search(text) and not (_QUOTA_FOLLOWUP.fullmatch(text) and text.strip('？?。！! \t')):return False
    if recent_quota:return True
    messages=[m for m in (contexts or []) if isinstance(m,dict)]
    previous=[_context_text(m) for m in messages if m.get('role')=='user' and _context_text(m).strip()!=text.strip()]
    if not previous:return False
    last=previous[-1]
    return bool(_CODEX.search(last) and _QUOTA.search(last))


class HarnessManager(Star):
    def __init__(self,context:Context,config=None):
        super().__init__(context,config)
        self.config=config or {}
        try:
            codex=CodexAdapter(exclusive_sessions=self.config.get("exclusive_codex_sessions",[]))
        except Exception as exc:
            # A missing CLI must not remove the plugin: the read-only query path
            # for the other harnesses stays available and the reason is explicit.
            logger.warning("[Harness] Codex adapter unavailable: %s",exc)
            codex=UnavailableAdapter("codex",f"本机 Codex CLI 不可用：{exc}")
        self.adapters={"codex":codex,
                       "workbuddy":UnavailableAdapter("workbuddy","官方 Open API 需要第三方应用 OAuth 授权；当前未配置，不能把本地助理接口当成普通桌面会话接口。",True,quota_reason="WorkBuddy 额度暂时无法自动读取；尚未确认公开积分查询接口，不能保证补充授权即可读取。"),
                       "zcode":ZCodeAdapter()}
        self.store=SessionStore(ROOT/"data/harness/state.sqlite3")
        self.store.mark_tracking_unknown()
        self.store.mark_pending_notifications_unknown()
        self._tasks=set()
        self._quota_recent={}
        self._tracking_lock=asyncio.Lock()
        self._codex_watcher=None

    async def _send_and_track(self,event,target,adapter,text):
        async with self._tracking_lock:
            try:result=await adapter.send(target['native_id'],text)
            except Exception as exc:
                logger.warning('[Harness] send outcome unknown: %s',type(exc).__name__)
                return '发送结果未确认，请先检查原 Harness 会话，勿盲目重发。'
            if result.state!=Capability.SUPPORTED:
                return f'{result.state}：{result.reason}'
            turn=(result.data or {}).get('turn') or {}
            try:self.store.register_turn(event.unified_msg_origin,target,turn.get('id'))
            except Exception:
                return '指令已被接收，但无法确认回合跟踪状态；请到原 Harness 检查，勿盲目重发。'
            if target['harness']=='codex' and adapter.client.events_lost:
                self.store.mark_tracking_unknown()
                return '指令已被接收，但事件流曾丢失，运行及通知状态未知；请到原 Harness 检查。'
            if target['harness']=='codex' and (self._codex_watcher is None or self._codex_watcher.done()):
                self._codex_watcher=asyncio.create_task(self._watch_codex(adapter))
                self._tasks.add(self._codex_watcher)
                self._codex_watcher.add_done_callback(self._tasks.discard)
            return '指令已被 Harness 接收，等待实际执行结果。'

    async def _watch_codex(self,adapter):
        client=adapter.client
        try:
            while client.ready:
                if client.events_lost:
                    self.store.mark_tracking_unknown()
                    return
                try:message=await asyncio.wait_for(client.events.get(),2)
                except asyncio.TimeoutError:continue
                method=message.get('method')
                if method not in ('turn/started','turn/completed'):continue
                params=message.get('params') or {};turn=params.get('turn') or {}
                status='inProgress' if method=='turn/started' else turn.get('status')
                async with self._tracking_lock:
                    previous=self.store.observe_turn('codex',params.get('threadId'),turn.get('id'),status)
                if previous and status in ('completed','failed','interrupted'):
                    await self._notify_codex_turn(adapter,previous,params.get('threadId'),turn.get('id'),status)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning('[Harness] task watcher stopped: %s',type(exc).__name__)
        finally:
            self.store.mark_tracking_unknown()

    async def _notify_codex_turn(self,adapter,tracked,thread_id,turn_id,status):
        actor=tracked['actor']
        if not self._allowed(SimpleNamespace(unified_msg_origin=actor)):return
        key='codex-turn:'+turn_id
        if not self.store.reserve_notification(key,actor,tracked['session_id']):return
        summary='暂未取得该回合的最终摘要。'
        if status=='completed':
            try:
                result=await adapter.read_session(thread_id)
                for item in (result.data or {}).get('recent_turns',[]):
                    if item.get('turn_id')==turn_id and item.get('output_phase')=='final_answer' and item.get('messages'):
                        summary=redact_text(item['messages'][0])[:1800]
                        break
            except Exception:pass
        label={'completed':'已完成','failed':'执行失败','interrupted':'已中断'}[status]
        content=f'Codex：{label}\n{summary}' if status=='completed' else f'Codex：{label}，请查看原会话。'
        try:
            platform_id,_,openid=actor.split(':',2)
            platform=self.context.get_platform_inst(platform_id)
            if not platform or platform.meta().name!='qq_official':raise ValueError('unsupported transport')
            from astrbot.core.platform.sources.qqofficial.qqofficial_message_event import QQOfficialMessageEvent
            ret=await asyncio.wait_for(QQOfficialMessageEvent.post_c2c_message(SimpleNamespace(bot=platform.client),openid=openid,msg_type=0,content=content),15)
            message_id=ret.get('id') if isinstance(ret,dict) else getattr(ret,'id',None)
            if not message_id:raise RuntimeError('no QQ delivery receipt')
            self.store.finish_notification(key,'delivered',str(message_id))
        except Exception as exc:
            self.store.finish_notification(key,'delivery_unknown')
            logger.warning('[Harness] completion delivery unconfirmed: %s',type(exc).__name__)

    async def _tracked_status(self,event):
        rows=self.store.tracked(event.unified_msg_origin)
        adapter=self.adapters['codex'];client=getattr(adapter,'client',None)
        online=bool(client and client.ready and not client.events_lost and not client.server_requests.qsize())
        lines=[] if rows else ['本服务没有正在跟踪的回合；其他桌面任务的实时状态未知。']
        for row in rows[:5]:
            status=row['status']
            if status=='inProgress' and online and row['started_at']:
                minutes=max(0,int((time.time()-row['started_at'])//60))
                lines.append(f"{row['harness']} 会话 {row['session_id']}：收到运行事件后约 {minutes} 分钟。")
            elif status in ('accepted','inProgress','unknown'):
                lines.append(f"{row['harness']} 会话 {row['session_id']}：实时状态未知或等待输入。")
            else:
                label={'completed':'已完成','failed':'执行失败','interrupted':'已中断'}.get(status,'状态未知')
                lines.append(f"{row['harness']} 会话 {row['session_id']}：{label}。")
        if any(row['harness']=='codex' for row in rows) or (_CODEX.search(event.get_message_str()) and _QUOTA.search(event.get_message_str())):
            try:
                result=await (await self._ready('codex')).quota()
                lines.append(format_quota(result.data) if result.state==Capability.SUPPORTED else 'Codex 额度暂时不可用。')
            except Exception:lines.append('Codex 额度暂时不可用。')
        return '\n'.join(lines)

    @filter.on_astrbot_loaded()
    async def on_loaded(self):
        self.store.cleanup_pending()
        cleanup=asyncio.create_task(self._cleanup_pending_messages())
        self._tasks.add(cleanup)
        cleanup.add_done_callback(self._tasks.discard)
        if self.config.get('send_setup_message',False):
            task=asyncio.create_task(self._setup_notice())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def _cleanup_pending_messages(self):
        while True:
            await asyncio.sleep(60)
            self.store.cleanup_pending()

    async def _setup_notice(self):
        await asyncio.sleep(8)
        targets=self.config.get('allowed_private_sessions',[])
        if len(targets)!=1:return
        umo=targets[0]
        if not self._allowed(SimpleNamespace(unified_msg_origin=umo)):return
        key='harness-setup-v1'
        with self.store.db:
            cursor=self.store.db.execute('INSERT OR IGNORE INTO notifications VALUES(?,?,?,?,?,?,?)',(key,umo,None,None,'pending','{}',time.time()))
        if cursor.rowcount!=1:return
        try:
            adapter=await self._ready()
            sessions=await adapter.list_sessions(limit=5)
            self.store.save(sessions.data)
            platform_id,_,openid=umo.split(':',2)
            platform=self.context.get_platform_inst(platform_id)
            if not platform or platform.meta().name!='qq_official':raise ValueError('Unsupported receipt transport')
            from astrbot.core.platform.sources.qqofficial.qqofficial_message_event import QQOfficialMessageEvent
            ret=await QQOfficialMessageEvent.post_c2c_message(
                SimpleNamespace(bot=platform.client),openid=openid,msg_type=0,
                content='Harness 查询入口已接好。私聊发「/助手 会话」看 Codex 现有会话，「/助手 读取 编号」看结果，「/助手 额度」查实际额度。桌面会话当前只读，尚未把“能读取”当成“能接管续聊”。',
            )
            message_id=ret.get('id') if isinstance(ret,dict) else getattr(ret,'id',None)
            if not message_id:raise RuntimeError('No QQ delivery receipt')
            with self.store.db:self.store.db.execute("UPDATE notifications SET state='delivered',qq_message_id=? WHERE event_key=?",(str(message_id),key))
            logger.info('[Harness] setup notice accepted by QQ; receipt stored')
        except Exception as exc:
            # A timeout may have delivered. Do not resend blindly and create duplicates.
            with self.store.db:self.store.db.execute("UPDATE notifications SET state='delivery_unknown',payload=? WHERE event_key=?",(json.dumps({'error_type':type(exc).__name__}),key))
            logger.warning('[Harness] setup notice not confirmed: %s',type(exc).__name__)

    def _allowed(self,event):
        parts=event.unified_msg_origin.split(":",2)
        return len(parts)==3 and parts[1]=="FriendMessage" and event.unified_msg_origin in self.config.get("allowed_private_sessions",[])

    async def _ready(self,name="codex"):
        adapter=self.adapters.get(name)
        if adapter is None:raise ValueError("可选 Harness：codex、workbuddy、zcode")
        if name=="codex":await adapter.start()
        return adapter

    @filter.on_llm_request(priority=100)
    async def live_quota_request(self,event:AstrMessageEvent,req):
        # Never route an unauthorized actor, and never answer from old tool output.
        text=req.prompt or ''
        if not self._allowed(event):
            return
        key=event.unified_msg_origin
        last_quota=self._quota_recent.get(key)
        recent=last_quota is not None and time.monotonic()-last_quota<600
        if not is_live_codex_quota_question(text,getattr(req,'contexts',None),recent):
            self._quota_recent.pop(key,None)
            return
        self._quota_recent[key]=time.monotonic()
        try:
            adapter=await self._ready('codex')
            result=await adapter.quota()
            if result.state!=Capability.SUPPORTED:
                reply=f"实时额度查询未完成：{result.reason or result.state}"
            else:
                window=300 if _FIVE_HOURS.search(text) else 10080 if _SEVEN_DAYS.search(text) else None
                reply=format_quota(result.data,window_minutes=window)
        except Exception as exc:
            logger.warning('[Harness] live quota request failed: %s',type(exc).__name__)
            reply=f"实时额度查询未完成（{type(exc).__name__}），暂时无法确认当前额度。"
        await event.send(MessageChain([Plain(reply)]))
        event.stop_event()

    @filter.on_llm_request(priority=200)
    async def live_task_status(self,event:AstrMessageEvent,req):
        if not self._allowed(event) or not _TASK_STATUS.search(req.prompt or ''):
            return
        reply=await self._tracked_status(event)
        await event.send(MessageChain([Plain(reply)]))
        event.stop_event()

    @filter.event_message_type(filter.EventMessageType.PRIVATE_MESSAGE, priority=100)
    async def dispatch_guard(self,event:AstrMessageEvent):
        # Language generation cannot invent a write acknowledgement. Explicit
        # dispatch enters the structured selection/send path instead.
        if not self._allowed(event) or not is_harness_dispatch(event.get_message_str()):return
        try:self.store.stage_message(event.unified_msg_origin,event.get_message_str())
        except ValueError as exc:
            event.stop_event();yield event.plain_result(str(exc));return
        event.stop_event()
        yield event.plain_result('还没有发送。请先用「/助手 会话」查看目标，再回复「/助手 确认 harness 编号」；待发内容 5 分钟内有效。桌面会话未独占交接时只读。')

    @filter.llm_tool(name='harness_query')
    async def query_tool(self,event:AstrMessageEvent,operation:str,harness:str='all',session_ref:str='')->str:
        """只读查询已接入的编程 Harness，不执行开发、不发送指令。目标不明确时先列会话供用户选择。

        Args:
            operation(string): sessions 列出现有会话；read 读取选定会话结果；quota 查询真实额度
            harness(string): codex、workbuddy、zcode 或 all
            session_ref(string): read 必须指定列表编号或完整会话 ID，不能猜测
        """
        if not self._allowed(event):return json.dumps({'state':'Unsupported','reason':'仅配置白名单中的私聊可查询 Harness；群聊不可使用。'},ensure_ascii=False)
        try:
            if operation=='read':
                target=self.store.resolve(session_ref,None if harness=='all' else harness)
                adapter=await self._ready(target['harness']);result=await adapter.read_session(target['native_id'])
                if result.data:
                    for turn in result.data.get('recent_turns',[]):turn['messages']=[text[:1000] for text in turn['messages'][:1]]
                data={'state':result.state,'data':result.data,'reason':result.reason}
            elif operation in ('quota','sessions'):
                names=list(self.adapters) if harness=='all' else [harness]
                values=[]
                for name in names:
                    adapter=await self._ready(name)
                    result=await adapter.quota() if operation=='quota' else await adapter.list_sessions(limit=10)
                    if operation=='quota' and name=='codex':self._quota_recent[event.unified_msg_origin]=time.monotonic()
                    if operation=='quota' and result.state==Capability.SUPPORTED:result.data['display']=format_quota(result.data)
                    if operation=='sessions' and result.state==Capability.SUPPORTED:
                        self.store.save(result.data)
                        result.data=self.store.list(name)
                    values.append({'harness':name,'state':result.state,'data':result.data,'reason':result.reason})
                data={'results':values}
            else:raise ValueError('operation 只能是 sessions、read、quota。')
        except ValueError as exc:data={'state':'Unsupported','reason':str(exc)}
        except Exception as exc:data={'state':'TemporarilyUnavailable','reason':type(exc).__name__}
        return json.dumps(redact(data),ensure_ascii=False)

    @filter.command("助手")
    async def manage(self,event:AstrMessageEvent):
        if not self._allowed(event):
            yield event.plain_result("此私聊未获 Harness 管理授权；群聊不开放此功能。")
            return
        text=event.get_message_str()
        command=text.split("助手",1)[-1].lstrip()
        try:
            args=command.split(maxsplit=1)
            action=args[0] if args else "帮助"
            if action in ("帮助","help"):
                reply="可用：/助手 会话 [codex|workbuddy|zcode]、/助手 状态、/助手 读取 编号、/助手 额度 [codex]、/助手 继续 原文、/助手 确认 harness 编号、/助手 选择 编号、/助手 发送 指令。确认仅用于一次发送，5 分钟失效；桌面会话未独占交接时只读。"
            elif action=="会话":
                adapter=await self._ready(args[1].lower() if len(args)>1 else "codex")
                result=await adapter.list_sessions()
                if result.state!=Capability.SUPPORTED:reply=f"{result.state}：{result.reason}"
                else:
                    self.store.save(result.data)
                    rows=self.store.list(adapter.name);shown=rows[:10]
                    reply="\n".join(f"{r['id']}. {r['title']}\n{r['project_path']}" for r in shown) or "没有找到会话。"
                    if len(rows)>len(shown):reply+=f"\n（已保存 {len(rows)} 个，只列最近 {len(shown)} 个。）"
                    reply += "\n这些是已保存会话；独立进程返回的状态不能代表桌面任务的实时状态。"
            elif action=="读取" and len(args)==2:
                target=self.store.resolve(args[1]);adapter=await self._ready(target['harness'])
                result=await adapter.read_session(target['native_id'])
                if result.state!=Capability.SUPPORTED:reply=f"{result.state}：{result.reason}"
                else:
                    turns=result.data.get('recent_turns',[])
                    texts=[text for turn in turns for text in turn['messages']]
                    status=turns[0].get('status','unknown') if turns else 'unknown'
                    reply=f"{target['title']}\n最近保存回合：{status}；桌面实时状态未知。\n"+(redact_text(texts[0])[:2800] if texts else "最近记录没有最终助手输出，不能据此宣称任务已完成。")
                    if texts and len(redact_text(texts[0]))>2800:reply+='\n（内容过长，当前只展示前 2800 字；完整输出需在原 Harness 查看。）'
            elif action=="状态":
                reply=await self._tracked_status(event)
            elif action=="额度":
                adapter=await self._ready(args[1].lower() if len(args)>1 else "codex");result=await adapter.quota()
                if adapter.name=='codex':self._quota_recent[event.unified_msg_origin]=time.monotonic()
                if result.state!=Capability.SUPPORTED:reply=f"{result.state}：{result.reason}"
                else:
                    reply=format_quota(result.data)
            elif action=="继续" and len(args)==2:
                self.store.stage_message(event.unified_msg_origin,args[1])
                reply='准备发送的内容已暂存，尚未发送。请先用 /助手 会话 查看编号，再回复 /助手 确认 codex|workbuddy|zcode 编号；5 分钟内有效。'
            elif action=="确认" and len(args)==2:
                parts=args[1].split()
                if len(parts)!=2 or parts[0].lower() not in self.adapters:
                    raise ValueError('请指定 Harness 和会话编号，例如 /助手 确认 codex 1。')
                name,selector=parts[0].lower(),parts[1]
                target=self.store.resolve(selector,name)
                adapter=await self._ready(name)
                text=self.store.consume_message(event.unified_msg_origin)
                reply=await self._send_and_track(event,target,adapter,text)
            elif action=="选择" and len(args)==2:
                target=self.store.resolve(args[1]);self.store.bind(event.unified_msg_origin,target['id'])
                reply=f"已选择 {target['harness']} / {target['title']}，仅供下一条发送，5 分钟失效。"
            elif action=="发送" and len(args)>1:
                target=self.store.consume_binding(event.unified_msg_origin)
                adapter=await self._ready(target['harness'])
                reply=await self._send_and_track(event,target,adapter,command.split(None,1)[1])
            else:reply="参数不完整，发送 /助手 帮助 查看用法。"
        except ValueError as exc:
            reply=str(exc)
        except Exception as exc:
            logger.warning("[Harness] request failed: %s",type(exc).__name__)
            reply=f"查询或操作未完成（{type(exc).__name__}）。未据此宣称任何任务已完成；可稍后重试。"
        yield event.plain_result(reply)

    async def terminate(self):
        for task in self._tasks:task.cancel()
        if self._tasks:await asyncio.gather(*self._tasks,return_exceptions=True)
        await self.adapters['codex'].close()
        self.store.close()
