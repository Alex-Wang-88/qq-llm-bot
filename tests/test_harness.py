import asyncio
import json
from pathlib import Path
import sys
import tempfile
import importlib.util
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from assistant_harness.adapters import Capability,CodexAdapter,Result,UnavailableAdapter,normalize_quota
from assistant_harness.codex_rpc import CodexRpc,RpcError,_executable_name,resolve_codex_executable
from assistant_harness.store import AmbiguousTarget,SessionStore


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
        client=SimpleNamespace(request=AsyncMock(side_effect=[{}, {'turn':{'id':'turn-1','status':'inProgress'}}]))
        adapter=CodexAdapter(client=client,exclusive_sessions=['owned-thread'])
        result=await adapter.send('owned-thread','do not commit')
        self.assertEqual(result.data['state'],'accepted')
        self.assertEqual(client.request.await_args_list[0].args[1]['threadId'],'owned-thread')
        self.assertEqual(client.request.await_args_list[0].args[1]['approvalPolicy'],'on-request')
        self.assertEqual(client.request.await_args_list[1].args[1]['threadId'],'owned-thread')
        self.assertNotEqual(result.data['state'],'completed')
    def test_used_is_not_remaining_and_missing_is_not_zero(self):
        result=normalize_quota({'rateLimits':{'primary':{'usedPercent':12,'windowDurationMins':300,'resetsAt':1}}})
        self.assertEqual(result['windows'][0]['remaining_percent'],88)
        self.assertEqual(normalize_quota({})['windows'],[])
    def test_multi_bucket_quotas_keep_categories(self):
        result=normalize_quota({'rateLimitsByLimitId':{'codex':{'primary':{'usedPercent':15}},'model-a':{'primary':{'usedPercent':105}}}})
        self.assertEqual([w['remaining_percent'] for w in result['windows']],[85,0])
        self.assertEqual([w['category'] for w in result['windows']],['codex','model-a'])


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

    def event(self,umo,text):
        return SimpleNamespace(unified_msg_origin=umo,get_message_str=lambda:text,plain_result=lambda text:text)

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
        values=[value async for value in plugin.dispatch_guard(event)]
        self.assertTrue(event.stopped)
        self.assertIn('还没有发送',values[0])

    async def test_raw_instruction_quotes_and_newlines_are_preserved(self):
        plugin=self.plugin_class.__new__(self.plugin_class)
        plugin.config={'allowed_private_sessions':['qq:FriendMessage:owner']}
        adapter=SimpleNamespace(send=AsyncMock(return_value=Result(Capability.SUPPORTED,{'state':'accepted'})))
        plugin._ready=AsyncMock(return_value=adapter)
        with tempfile.TemporaryDirectory() as temp:
            plugin.store=SessionStore(Path(temp)/'state.db')
            plugin.store.save([{'harness':'codex','native_id':'thread-a','project_path':'C:/a','title':'a'}])
            plugin.store.bind('qq:FriendMessage:owner',plugin.store.resolve('thread-a')['id'])
            values=[value async for value in plugin.manage(self.event('qq:FriendMessage:owner',"/助手 发送 don't commit\n先看原因"))]
            adapter.send.assert_awaited_once_with('thread-a',"don't commit\n先看原因")
            self.assertIn('接收',values[0]);self.assertNotIn('已完成',values[0])
            plugin.store.close()


if __name__=='__main__':unittest.main()
