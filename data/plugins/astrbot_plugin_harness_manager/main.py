"""Explicit private-chat controls; no shell execution or guessed task targets."""
import json
import asyncio
from pathlib import Path
import time
from types import SimpleNamespace
import sys

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star

ROOT=Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from assistant_harness.adapters import Capability, CodexAdapter, UnavailableAdapter, format_quota
from assistant_harness.store import SessionStore
from assistant_security import redact_text,redact
from assistant_dialogue import is_harness_dispatch


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
                       "zcode":UnavailableAdapter("zcode","已发现本机 ZCode 3.16.3；运行时正式接口兼容性仍待验证。")}
        self.store=SessionStore(ROOT/"data/harness/state.sqlite3")
        self._tasks=set()

    @filter.on_astrbot_loaded()
    async def on_loaded(self):
        if self.config.get('send_setup_message',False):
            task=asyncio.create_task(self._setup_notice())
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

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

    @filter.event_message_type(filter.EventMessageType.PRIVATE_MESSAGE, priority=100)
    async def dispatch_guard(self,event:AstrMessageEvent):
        # Language generation cannot invent a write acknowledgement. Explicit
        # dispatch enters the structured selection/send path instead.
        if not self._allowed(event) or not is_harness_dispatch(event.get_message_str()):return
        event.stop_event()
        yield event.plain_result('还没有发送。先用「/助手 会话」确认目标，再用「/助手 选择 编号」和「/助手 发送 原指令」。当前桌面会话只读，独占交接未完成时会明确拒绝发送。')

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
        text=event.get_message_str().strip()
        command=text.split("助手",1)[-1].strip()
        try:
            args=command.split(maxsplit=1)
            action=args[0] if args else "帮助"
            if action in ("帮助","help"):
                reply="可用：/助手 会话 [codex]、/助手 读取 编号、/助手 额度 [codex]、/助手 选择 编号、/助手 发送 指令。选择仅供下一条发送，5 分钟失效；桌面会话未独占交接时只读。"
            elif action=="会话":
                adapter=await self._ready(args[1].lower() if len(args)>1 else "codex")
                result=await adapter.list_sessions()
                if result.state!=Capability.SUPPORTED:reply=f"{result.state}：{result.reason}"
                else:
                    self.store.save(result.data)
                    rows=self.store.list(adapter.name);shown=rows[:10]
                    reply="\n".join(f"{r['id']}. {r['title']}\n{r['project_path']}" for r in shown) or "没有找到会话。"
                    if len(rows)>len(shown):reply+=f"\n（已保存 {len(rows)} 个，只列最近 {len(shown)} 个。）"
                    reply += "\n这些是已保存会话；notLoaded 不代表桌面任务已停止。"
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
            elif action=="额度":
                adapter=await self._ready(args[1].lower() if len(args)>1 else "codex");result=await adapter.quota()
                if result.state!=Capability.SUPPORTED:reply=f"{result.state}：{result.reason}"
                else:
                    reply=format_quota(result.data)
            elif action=="选择" and len(args)==2:
                target=self.store.resolve(args[1]);self.store.bind(event.unified_msg_origin,target['id'])
                reply=f"已选择 {target['harness']} / {target['title']}，仅供下一条发送，5 分钟失效。"
            elif action=="发送" and len(args)>1:
                target=self.store.consume_binding(event.unified_msg_origin)
                adapter=await self._ready(target['harness'])
                result=await adapter.send(target['native_id'],command.split(None,1)[1])
                reply=("指令已被 Harness 接收，等待实际执行结果。" if result.state==Capability.SUPPORTED else f"{result.state}：{result.reason}")
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
