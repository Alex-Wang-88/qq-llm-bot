import asyncio
import json
from pathlib import Path
import sys
import tempfile
import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from assistant_harness.adapters import Capability,CodexAdapter,Result,UnavailableAdapter,normalize_quota,format_quota
from datetime import datetime,timezone
from assistant_harness.codex_rpc import CodexRpc,RpcError,_executable_name,resolve_codex_executable
from assistant_harness.store import AmbiguousTarget,SessionStore
from assistant_harness.zcode import ZCodeAdapter,normalize_zcode_sessions


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.store=SessionStore(Path(self.temp.name)/'state.db')
        self.store.save([{'harness':'codex','native_id':'a','project_path':'C:/one','title':'same','status':'notLoaded'},
                         {'harness':'workbuddy','native_id':'b','project_path':'C:/two','title':'same','status':'unknown'}])
    def tearDown(self):self.store.close();self.temp.cleanup()
    def test_duplicate_names_require_disambiguation(self):
        with self.assertRaises(AmbiguousTarget):self.store.resolve('same')
        self.assertEqual(self.store.resolve('same','codex')['native_id'],'a')
    def test_binding_is_actor_scoped_and_one_use(self):
        target=self.store.resolve('a');self.store.bind('owner',target['id'])
        with self.assertRaises(ValueError):self.store.consume_binding('other')
        self.assertEqual(self.store.consume_binding('owner')['native_id'],'a')
        with self.assertRaises(ValueError):self.store.consume_binding('owner')
    def test_expired_binding_cannot_send(self):
        self.store.bind('owner',self.store.resolve('a')['id'],ttl=-1)
        with self.assertRaises(ValueError):self.store.consume_binding('owner')
    def test_route_survives_restart_without_binding_every_future_message(self):
        target=self.store.resolve('a');self.store.bind('owner',target['id']);self.store.close()
        self.store=SessionStore(Path(self.temp.name)/'state.db')
        self.assertEqual(self.store.consume_binding('owner')['native_id'],'a')
    def test_notification_cannot_be_used_by_different_actor(self):
        target=self.store.resolve('a')
        with self.store.db:self.store.db.execute('INSERT INTO notifications VALUES(?,?,?,?,?,?,?)',('done-a','owner',target['id'],'qq-1','delivered','{}',0))
        self.assertIsNone(self.store.notification_target('other','qq-1'))
        self.assertEqual(self.store.notification_target('owner','qq-1')['native_id'],'a')

    def test_setup_notice_is_not_a_session_binding(self):
        with self.store.db:self.store.db.execute('INSERT INTO notifications VALUES(?,?,?,?,?,?,?)',('setup','owner',None,'qq-setup','delivered','{}',0))
        self.assertIsNone(self.store.notification_target('owner','qq-setup'))

    def test_pending_messages_are_isolated_expiring_and_one_use(self):
        self.store.stage_message('owner','第一行\n不要提交\n')
        with self.assertRaises(ValueError):self.store.consume_message('other')
        self.assertEqual(self.store.consume_message('owner'),'第一行\n不要提交\n')
        with self.assertRaises(ValueError):self.store.consume_message('owner')

    def test_turn_state_and_notification_are_deduplicated(self):
        target=self.store.resolve('a');self.store.register_turn('owner',target,'turn-1')
        self.assertIsNone(self.store.observe_turn('codex','other','turn-1','completed'))
        self.assertIsNotNone(self.store.observe_turn('codex','a','turn-1','inProgress',now=100))
        self.assertEqual(self.store.tracked('owner')[0]['started_at'],100)
        self.assertIsNotNone(self.store.observe_turn('codex','a','turn-1','completed',now=110))
        self.assertIsNone(self.store.observe_turn('codex','a','turn-1','completed',now=111))
        self.assertTrue(self.store.reserve_notification('turn-1','owner',target['id']))
        self.assertFalse(self.store.reserve_notification('turn-1','owner',target['id']))
        self.store.finish_notification('turn-1','delivery_unknown')
        self.assertIsNone(self.store.notification_target('owner','nonexistent'))
        self.store.stage_message('owner','过期',ttl=-1)
        with self.assertRaises(ValueError):self.store.consume_message('owner')

    def test_disconnect_never_leaves_a_run_marked_live(self):
        target=self.store.resolve('a');self.store.register_turn('owner',target,'turn-1')
        self.store.observe_turn('codex','a','turn-1','inProgress',now=100)
        self.store.mark_tracking_unknown()
        self.assertEqual(self.store.tracked('owner')[0]['status'],'unknown')
        self.assertIsNone(self.store.observe_turn('codex','a','turn-1','completed',now=200))

    def test_expired_message_and_interrupted_delivery_are_not_replayed(self):
        self.store.stage_message('owner','待确认',ttl=-1)
        self.store.cleanup_pending()
        with self.assertRaises(ValueError):self.store.consume_message('owner')
        self.assertTrue(self.store.reserve_notification('key','owner',1))
        self.store.mark_pending_notifications_unknown()
        self.assertEqual(self.store.db.execute("SELECT state FROM notifications WHERE event_key='key'").fetchone()['state'],'delivery_unknown')
        self.assertFalse(self.store.reserve_notification('key','owner',1))


class ZCodeTests(unittest.IsolatedAsyncioTestCase):
    def test_independent_idle_is_never_presented_as_desktop_live(self):
        rows=normalize_zcode_sessions([{'sessionId':'sess_fixture','workspace':{'workspacePath':'C:/fixture'},'title':'测试','status':'idle'}])
        self.assertEqual(rows[0]['status'],'unknown')
        self.assertEqual(rows[0]['status_scope'],'independent_app_server')
        self.assertFalse(rows[0]['can_send'])

    async def test_missing_install_and_unverified_write_are_explicit(self):
        with tempfile.TemporaryDirectory() as temp:
            adapter=ZCodeAdapter(root=temp,node='missing')
            self.assertEqual((await adapter.list_sessions()).state,Capability.UNAVAILABLE)
            self.assertEqual((await adapter.send('fixture','text')).state,Capability.UNSUPPORTED)


class AdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_terminal_answer_is_returned_instead_of_progress(self):
        client=SimpleNamespace(request=AsyncMock(side_effect=[{'thread':{'id':'a'}},{'data':[{'id':'t','status':'completed','items':[
            {'type':'agentMessage','phase':'commentary','text':'Starting tests'},
            {'type':'agentMessage','phase':'final_answer','text':'48 tests passed; no commit created'}]}]}]))
        result=await CodexAdapter(client=client).read_session('a')
        self.assertEqual(result.data['recent_turns'][0]['messages'],['48 tests passed; no commit created'])
        self.assertEqual(result.data['recent_turns'][0]['status'],'completed')
    async def test_readonly_desktop_threads_are_not_resumed(self):
        client=SimpleNamespace(request=AsyncMock())
        adapter=CodexAdapter(client=client)
        result=await adapter.send('desktop-thread','continue')
        self.assertEqual(result.state,Capability.UNSUPPORTED)
        client.request.assert_not_awaited()
    async def test_exclusive_resume_keeps_same_id_and_approval_policy(self):
        client=SimpleNamespace(request=AsyncMock(side_effect=[{'thread':{'status':{'type':'notLoaded'}}},{}, {'turn':{'id':'turn-1','status':'inProgress'}}]))
        adapter=CodexAdapter(client=client,exclusive_sessions=['owned-thread'])
        result=await adapter.send('owned-thread','do not commit')
        self.assertEqual(result.data['state'],'accepted')
        self.assertEqual(client.request.await_args_list[1].args[1]['threadId'],'owned-thread')
        self.assertEqual(client.request.await_args_list[1].args[1]['approvalPolicy'],'on-request')
        self.assertEqual(client.request.await_args_list[2].args[1]['threadId'],'owned-thread')
        self.assertNotEqual(result.data['state'],'completed')

    async def test_active_owned_session_is_not_sent_another_turn(self):
        client=SimpleNamespace(request=AsyncMock(return_value={'thread':{'status':{'type':'active'}}}))
        result=await CodexAdapter(client=client,exclusive_sessions=['owned-thread']).send('owned-thread','continue')
        self.assertEqual(result.state,Capability.UNSUPPORTED)
        self.assertEqual(client.request.await_count,1)
    def test_used_is_not_remaining_and_missing_is_not_zero(self):
        result=normalize_quota({'rateLimits':{'primary':{'usedPercent':12,'windowDurationMins':300,'resetsAt':1}}})
        self.assertEqual(result['windows'][0]['remaining_percent'],88)
        self.assertEqual(normalize_quota({})['windows'],[])
    def test_multi_bucket_quotas_keep_categories(self):
        result=normalize_quota({'rateLimitsByLimitId':{'codex':{'primary':{'usedPercent':15}},'model-a':{'primary':{'usedPercent':105}}}})
        self.assertEqual([w['remaining_percent'] for w in result['windows']],[85,0])
        self.assertEqual([w['category'] for w in result['windows']],['codex','model-a'])

    def test_quota_time_is_local_and_countdown_is_calculated(self):
        data=normalize_quota({'rateLimits':{'primary':{'usedPercent':63,'windowDurationMins':300,'resetsAt':1793085785}}},collected_at='2026-10-27T03:13:06+00:00')
        result=format_quota(data,now=datetime(2026,10,27,3,13,6,tzinfo=timezone.utc),window_minutes=300)
        self.assertIn('2026-10-27 11:13:06（北京时间）',result)
        self.assertIn('4 小时 10 分钟',result)
        self.assertNotIn('7 天',result)
        self.assertIn('已到或已过',format_quota(data,now=datetime(2026,10,28,tzinfo=timezone.utc)))


class DiscoveryTests(unittest.TestCase):
    def build_extension(self,home,version):
        binary=Path(home)/'.vscode/extensions'/f'openai.chatgpt-{version}-win32-x64'/'bin'/'windows-x86_64'/f'{_executable_name()}'
        binary.parent.mkdir(parents=True,exist_ok=True);binary.write_bytes(b'')
        return binary

    def test_bundled_editor_binary_is_found_off_path(self):
        with tempfile.TemporaryDirectory() as temp:
            newest=self.build_extension(temp,'26.1021.22027')
            self.assertEqual(resolve_codex_executable(home=Path(temp),environ={'PATH':''}),newest)

    def test_newest_installed_extension_wins(self):
        with tempfile.TemporaryDirectory() as temp:
            self.build_extension(temp,'26.1007.21434');newest=self.build_extension(temp,'26.1021.22027')
            self.assertEqual(resolve_codex_executable(home=Path(temp),environ={'PATH':''}),newest)

    def test_explicit_override_wins_over_every_probe(self):
        with tempfile.TemporaryDirectory() as temp:
            self.build_extension(temp,'26.1021.22027')
            chosen=Path(temp)/f'custom-{_executable_name()}';chosen.write_bytes(b'')
            resolved=resolve_codex_executable(home=Path(temp),environ={'PATH':'','CODEX_CLI_PATH':str(chosen)})
            self.assertEqual(resolved,chosen)

    def test_nothing_installed_returns_none(self):
        with tempfile.TemporaryDirectory() as temp:
            self.assertIsNone(resolve_codex_executable(home=Path(temp),environ={'PATH':''}))

    def test_missing_cli_error_names_the_override(self):
        with patch('assistant_harness.codex_rpc.resolve_codex_executable',return_value=None):
            with self.assertRaises(FileNotFoundError) as caught:CodexRpc()
        self.assertIn('CODEX_CLI_PATH',str(caught.exception))


class DegradationTests(unittest.IsolatedAsyncioTestCase):
    async def test_unavailable_adapter_keeps_the_lifecycle_surface(self):
        adapter=UnavailableAdapter('codex','本机 Codex CLI 不可用：missing')
        await adapter.start();await adapter.close()
        result=await adapter.list_sessions()
        self.assertEqual(result.state,Capability.UNSUPPORTED)
        self.assertIn('不可用',result.reason)
    async def test_quota_network_failure_is_a_reason_not_a_crash(self):
        client=SimpleNamespace(request=AsyncMock(side_effect=RpcError({'code':-1,'message':'error sending request'})))
        result=await CodexAdapter(client=client).quota()
        self.assertEqual(result.state,Capability.UNAVAILABLE)
        self.assertIn('chatgpt.com',result.reason)
        self.assertIsNone(result.data)


FAKE_SERVER='''import sys,json
for line in sys.stdin:
 m=json.loads(line)
 if 'id' not in m:continue
 if m['method']=='failure':r={'error':{'code':-1,'message':'fixture error'}}
 else:r={'result':{'method':m['method']}}
 print(json.dumps({'id':m['id'],**r}),flush=True)
'''


class TransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_handshake_concurrent_start_and_requests(self):
        client=CodexRpc([sys.executable,'-u','-c',FAKE_SERVER],timeout=3)
        try:
            await asyncio.gather(client.start(),client.start())
            self.assertTrue(client.ready)
            values=await asyncio.gather(client.request('thread/list'),client.request('account/rateLimits/read'))
            self.assertEqual([v['method'] for v in values],['thread/list','account/rateLimits/read'])
            self.assertTrue(client.ready)
            old_pid=client.proc.pid
            await client.start()
            self.assertEqual(client.proc.pid,old_pid)
            with self.assertRaises(RpcError):await client.request('failure')
        finally:await client.close()


class AccessTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).resolve().parents[1]/'data/plugins/astrbot_plugin_harness_manager/main.py'
        spec=importlib.util.spec_from_file_location('harness_plugin_test',path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        cls.plugin_class=module.HarnessManager
        cls.module=module

    def event(self,umo,text):
        return SimpleNamespace(unified_msg_origin=umo,get_message_str=lambda:text,plain_result=lambda text:text)

    async def test_followup_queries_current_quota_and_bypasses_model(self):
        plugin=self.plugin_class.__new__(self.plugin_class)
        plugin.config={'allowed_private_sessions':['qq:FriendMessage:owner']}
        plugin._quota_recent={}
        data=normalize_quota({'rateLimits':{'primary':{'usedPercent':63,'windowDurationMins':300,'resetsAt':1793085785}}})
        adapter=SimpleNamespace(quota=AsyncMock(return_value=Result(Capability.SUPPORTED,data)))
        plugin._ready=AsyncMock(return_value=adapter)
        event=self.event('qq:FriendMessage:owner','下一次5小时重置还剩多久？')
        event.send=AsyncMock();event.stop_event=lambda:setattr(event,'stopped',True)
        req=SimpleNamespace(prompt=event.get_message_str(),contexts=[{'role':'user','content':[{'type':'text','text':'帮我看一下我电脑上codex额度还有多少'}]}])
        await plugin.live_quota_request(event,req)
        adapter.quota.assert_awaited_once();event.send.assert_awaited_once();self.assertTrue(event.stopped)
        self.assertIn('5 小时窗口',event.send.await_args.args[0].get_plain_text())

    async def test_quota_failure_never_reuses_history_and_unauthorized_never_queries(self):
        plugin=self.plugin_class.__new__(self.plugin_class);plugin.config={'allowed_private_sessions':['qq:FriendMessage:owner']}
        plugin._quota_recent={}
        adapter=SimpleNamespace(quota=AsyncMock(return_value=Result(Capability.UNAVAILABLE,reason='连接失败')))
        plugin._ready=AsyncMock(return_value=adapter)
        req=SimpleNamespace(prompt='查一下 Codex 剩余额度',contexts=[])
        stranger=self.event('qq:FriendMessage:stranger',req.prompt);stranger.send=AsyncMock()
        await plugin.live_quota_request(stranger,req);plugin._ready.assert_not_awaited()
        event=self.event('qq:FriendMessage:owner',req.prompt);event.send=AsyncMock();event.stop_event=lambda:setattr(event,'stopped',True)
        await plugin.live_quota_request(event,req)
        self.assertIn('实时额度查询未完成',event.send.await_args.args[0].get_plain_text())
        self.assertNotIn('%',event.send.await_args.args[0].get_plain_text())

    def test_followup_requires_context_and_does_not_capture_unrelated_reset(self):
        match=self.module.is_live_codex_quota_question
        self.assertFalse(match('下一次5小时重置还剩多久？',[]))
        self.assertFalse(match('下一次5小时重置还剩多久？',[{'role':'user','content':'游戏副本刷新时间是什么'}]))
        self.assertTrue(match('下一次5小时重置还剩多久？',[{'role':'user','content':'帮我看一下我电脑上codex额度还有多少'}]))
        self.assertTrue(match('下一次5小时重置还剩多久？',[],True))
        self.assertTrue(match('现在还剩多少？',[],True))
        self.assertFalse(match('现在还剩多少？',[],False))
        self.assertFalse(match('你好',[],True))

    async def test_unauthorized_actor_never_contacts_harness(self):
        plugin=self.plugin_class.__new__(self.plugin_class)
        plugin.config={'allowed_private_sessions':['qq:FriendMessage:owner']}
        plugin._ready=AsyncMock()
        values=[value async for value in plugin.manage(self.event('qq:FriendMessage:stranger','/助手 额度'))]
        self.assertIn('未获',values[0]);plugin._ready.assert_not_awaited()

    async def test_group_is_denied_even_if_misconfigured_in_allowlist(self):
        plugin=self.plugin_class.__new__(self.plugin_class)
        plugin.config={'allowed_private_sessions':['qq:GroupMessage:group']}
        plugin._ready=AsyncMock()
        values=[value async for value in plugin.manage(self.event('qq:GroupMessage:group','/助手 会话'))]
        self.assertIn('群聊',values[0]);plugin._ready.assert_not_awaited()

    async def test_llm_tool_has_same_private_authorization_gate(self):
        plugin=self.plugin_class.__new__(self.plugin_class)
        plugin.config={'allowed_private_sessions':['qq:FriendMessage:owner']}
        plugin._ready=AsyncMock()
        result=json.loads(await plugin.query_tool(self.event('qq:GroupMessage:group','查额度'),'quota','codex'))
        self.assertEqual(result['state'],'Unsupported');plugin._ready.assert_not_awaited()

    async def test_unavailable_backend_reports_reason_instead_of_removing_plugin(self):
        plugin=self.plugin_class.__new__(self.plugin_class)
        plugin.config={'allowed_private_sessions':['qq:FriendMessage:owner']}
        plugin.adapters={'codex':UnavailableAdapter('codex','本机 Codex CLI 不可用：missing')}
        values=[value async for value in plugin.manage(self.event('qq:FriendMessage:owner','/助手 会话'))]
        self.assertIn('不可用',values[0])

    async def test_natural_dispatch_stops_before_language_model_acknowledgement(self):
        plugin=self.plugin_class.__new__(self.plugin_class)
        plugin.config={'allowed_private_sessions':['qq:FriendMessage:owner']}
        event=self.event('qq:FriendMessage:owner','把“继续修复”交给 Codex。现在有两个会话。')
        event.stop_event=lambda:setattr(event,'stopped',True)
        with tempfile.TemporaryDirectory() as temp:
            plugin.store=SessionStore(Path(temp)/'state.db')
            values=[value async for value in plugin.dispatch_guard(event)]
            self.assertTrue(event.stopped)
            self.assertIn('还没有发送',values[0])
            self.assertEqual(plugin.store.consume_message(event.unified_msg_origin),event.get_message_str())
            plugin.store.close()

    async def test_status_question_does_not_fabricate_running_desktop(self):
        plugin=self.plugin_class.__new__(self.plugin_class)
        plugin.config={'allowed_private_sessions':['qq:FriendMessage:owner']}
        event=self.event('qq:FriendMessage:owner','Codex 任务跑了多久，剩多少额度？')
        event.send=AsyncMock();event.stop_event=lambda:setattr(event,'stopped',True)
        plugin._tracked_status=AsyncMock(return_value='本服务没有正在跟踪的回合；其他桌面任务的实时状态未知。')
        await plugin.live_task_status(event,SimpleNamespace(prompt=event.get_message_str()))
        self.assertTrue(event.stopped)
        self.assertIn('状态未知',event.send.await_args.args[0].get_plain_text())

    async def test_raw_instruction_quotes_and_newlines_are_preserved(self):
        plugin=self.plugin_class.__new__(self.plugin_class)
        plugin.config={'allowed_private_sessions':['qq:FriendMessage:owner']}
        adapter=SimpleNamespace(send=AsyncMock(return_value=Result(Capability.SUPPORTED,{'state':'accepted','turn':{'id':'turn-1'}})),client=SimpleNamespace(ready=False,events_lost=False))
        plugin._tracking_lock=asyncio.Lock();plugin._codex_watcher=None;plugin._tasks=set()
        plugin._ready=AsyncMock(return_value=adapter)
        with tempfile.TemporaryDirectory() as temp:
            plugin.store=SessionStore(Path(temp)/'state.db')
            plugin.store.save([{'harness':'codex','native_id':'thread-a','project_path':'C:/a','title':'a'}])
            plugin.store.bind('qq:FriendMessage:owner',plugin.store.resolve('thread-a')['id'])
            values=[value async for value in plugin.manage(self.event('qq:FriendMessage:owner',"/助手 发送 don't commit\n先看原因"))]
            adapter.send.assert_awaited_once_with('thread-a',"don't commit\n先看原因")
            self.assertIn('接收',values[0]);self.assertNotIn('已完成',values[0])
            await plugin._codex_watcher
            plugin.store.close()

    async def test_continuation_requires_harness_and_session_before_one_send(self):
        plugin=self.plugin_class.__new__(self.plugin_class)
        plugin.config={'allowed_private_sessions':['qq:FriendMessage:owner']}
        adapter=SimpleNamespace(name='codex',send=AsyncMock(return_value=Result(Capability.SUPPORTED,{'state':'accepted','turn':{'id':'turn-1'}})),client=SimpleNamespace(ready=False,events_lost=False))
        plugin._tracking_lock=asyncio.Lock();plugin._codex_watcher=None;plugin._tasks=set()
        plugin.adapters={'codex':adapter};plugin._ready=AsyncMock(return_value=adapter)
        with tempfile.TemporaryDirectory() as temp:
            plugin.store=SessionStore(Path(temp)/'state.db')
            plugin.store.save([{'harness':'codex','native_id':'thread-a','project_path':'C:/a','title':'a'}])
            staged=[x async for x in plugin.manage(self.event('qq:FriendMessage:owner','/助手 继续 保留引号\n不要提交\n'))]
            self.assertIn('尚未发送',staged[0]);adapter.send.assert_not_awaited()
            wrong=[x async for x in plugin.manage(self.event('qq:FriendMessage:owner','/助手 确认 codex 99'))]
            self.assertIn('未找到',wrong[0]);adapter.send.assert_not_awaited()
            done=[x async for x in plugin.manage(self.event('qq:FriendMessage:owner','/助手 确认 codex 1'))]
            adapter.send.assert_awaited_once_with('thread-a','保留引号\n不要提交\n')
            self.assertIn('接收',done[0]);self.assertNotIn('已完成',done[0])
            again=[x async for x in plugin.manage(self.event('qq:FriendMessage:owner','/助手 确认 codex 1'))]
            self.assertIn('不存在或已过期',again[0]);adapter.send.assert_awaited_once()
            await plugin._codex_watcher
            plugin.store.close()

    async def test_real_event_shape_tracks_only_owned_turn_and_notifies_once(self):
        plugin=self.plugin_class.__new__(self.plugin_class)
        plugin._tracking_lock=asyncio.Lock()
        received=asyncio.Event()
        plugin._notify_codex_turn=AsyncMock(side_effect=lambda *args:received.set())
        client=SimpleNamespace(ready=True,events_lost=False,events=asyncio.Queue())
        adapter=SimpleNamespace(client=client)
        with tempfile.TemporaryDirectory() as temp:
            plugin.store=SessionStore(Path(temp)/'state.db')
            plugin.store.save([{'harness':'codex','native_id':'thread-a','title':'a'}])
            plugin.store.register_turn('owner',plugin.store.resolve('thread-a'),'turn-1')
            task=asyncio.create_task(plugin._watch_codex(adapter))
            for method,thread,turn,status in [('turn/started','thread-a','turn-1','inProgress'),('turn/completed','other','turn-1','completed'),('turn/completed','thread-a','turn-1','completed'),('turn/completed','thread-a','turn-1','completed')]:
                await client.events.put({'method':method,'params':{'threadId':thread,'turn':{'id':turn,'status':status}}})
            await asyncio.wait_for(received.wait(),2)
            await asyncio.sleep(0)
            self.assertEqual(plugin.store.tracked('owner')[0]['status'],'completed')
            plugin._notify_codex_turn.assert_awaited_once()
            task.cancel();await asyncio.gather(task,return_exceptions=True)
            plugin.store.close()

    async def test_completion_notification_uses_final_answer_and_deduplicates(self):
        plugin=self.plugin_class.__new__(self.plugin_class)
        plugin.config={'allowed_private_sessions':['qq:FriendMessage:owner']}
        platform=SimpleNamespace(client=object(),meta=lambda:SimpleNamespace(name='qq_official'))
        plugin.context=SimpleNamespace(get_platform_inst=lambda name:platform)
        adapter=SimpleNamespace(read_session=AsyncMock(return_value=Result(Capability.SUPPORTED,{'recent_turns':[{'turn_id':'turn-1','output_phase':'final_answer','messages':['已完成测试。']}]})))
        with tempfile.TemporaryDirectory() as temp:
            plugin.store=SessionStore(Path(temp)/'state.db')
            plugin.store.save([{'harness':'codex','native_id':'thread-a','title':'a'}])
            target=plugin.store.resolve('thread-a')
            tracked={'actor':'qq:FriendMessage:owner','session_id':target['id']}
            with patch('astrbot.core.platform.sources.qqofficial.qqofficial_message_event.QQOfficialMessageEvent.post_c2c_message',new_callable=AsyncMock,return_value={'id':'receipt'}) as post:
                await plugin._notify_codex_turn(adapter,tracked,'thread-a','turn-1','completed')
                await plugin._notify_codex_turn(adapter,tracked,'thread-a','turn-1','completed')
                post.assert_awaited_once()
                self.assertIn('Codex：已完成',post.await_args.kwargs['content'])
                self.assertIn('已完成测试。',post.await_args.kwargs['content'])
                self.assertEqual(plugin.store.notification_target(tracked['actor'],'receipt')['native_id'],'thread-a')
            plugin.store.close()

    async def test_unconfirmed_notification_is_never_blindly_resent(self):
        plugin=self.plugin_class.__new__(self.plugin_class)
        plugin.config={'allowed_private_sessions':['qq:FriendMessage:owner']}
        platform=SimpleNamespace(client=object(),meta=lambda:SimpleNamespace(name='qq_official'))
        plugin.context=SimpleNamespace(get_platform_inst=lambda name:platform)
        adapter=SimpleNamespace(read_session=AsyncMock(return_value=Result(Capability.SUPPORTED,{'recent_turns':[]})))
        with tempfile.TemporaryDirectory() as temp:
            plugin.store=SessionStore(Path(temp)/'state.db')
            plugin.store.save([{'harness':'codex','native_id':'thread-a','title':'a'}]);target=plugin.store.resolve('thread-a')
            tracked={'actor':'qq:FriendMessage:owner','session_id':target['id']}
            with patch('astrbot.core.platform.sources.qqofficial.qqofficial_message_event.QQOfficialMessageEvent.post_c2c_message',new_callable=AsyncMock,side_effect=TimeoutError) as post:
                await plugin._notify_codex_turn(adapter,tracked,'thread-a','turn-1','completed')
                await plugin._notify_codex_turn(adapter,tracked,'thread-a','turn-1','completed')
                post.assert_awaited_once()
            state=plugin.store.db.execute("select state from notifications where event_key='codex-turn:turn-1'").fetchone()['state']
            self.assertEqual(state,'delivery_unknown')
            plugin.store.close()

    async def test_event_loss_makes_running_status_unknown(self):
        plugin=self.plugin_class.__new__(self.plugin_class)
        plugin._tracking_lock=asyncio.Lock();plugin._notify_codex_turn=AsyncMock()
        client=SimpleNamespace(ready=True,events_lost=True,events=asyncio.Queue())
        with tempfile.TemporaryDirectory() as temp:
            plugin.store=SessionStore(Path(temp)/'state.db')
            plugin.store.save([{'harness':'codex','native_id':'thread-a','title':'a'}])
            plugin.store.register_turn('owner',plugin.store.resolve('thread-a'),'turn-1')
            plugin.store.observe_turn('codex','thread-a','turn-1','inProgress')
            await plugin._watch_codex(SimpleNamespace(client=client))
            self.assertEqual(plugin.store.tracked('owner')[0]['status'],'unknown')
            plugin._notify_codex_turn.assert_not_awaited()
            plugin.store.close()

    async def test_send_reports_event_loss_without_claiming_monitoring(self):
        plugin=self.plugin_class.__new__(self.plugin_class);plugin._tracking_lock=asyncio.Lock()
        event=self.event('qq:FriendMessage:owner','send')
        adapter=SimpleNamespace(client=SimpleNamespace(events_lost=True),send=AsyncMock(return_value=Result(Capability.SUPPORTED,{'turn':{'id':'turn-1'}})))
        with tempfile.TemporaryDirectory() as temp:
            plugin.store=SessionStore(Path(temp)/'state.db')
            plugin.store.save([{'harness':'codex','native_id':'thread-a','title':'a'}])
            reply=await plugin._send_and_track(event,plugin.store.resolve('thread-a'),adapter,'继续')
            self.assertIn('状态未知',reply)
            self.assertEqual(plugin.store.tracked(event.unified_msg_origin)[0]['status'],'unknown')
            plugin.store.close()


if __name__=='__main__':unittest.main()
