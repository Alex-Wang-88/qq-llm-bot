import asyncio
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock

import assistant_hapi_local as local

ROOT=Path(__file__).resolve().parents[1]
p=ROOT/'data/plugins/astrbot_plugin_hapi_connector'
spec=importlib.util.spec_from_file_location('hapi_readonly_test',p/'__init__.py',submodule_search_locations=[str(p)])
package=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=package
spec.loader.exec_module(package)
from hapi_readonly_test.main import HapiConnectorPlugin


class Event:
    unified_msg_origin='fixture:FriendMessage:1'
    message_str='/hapi create'
    def is_private_chat(self): return True
    def is_admin(self): return True
    def plain_result(self, text): return text


class HapiBoundaryTests(unittest.IsolatedAsyncioTestCase):
    def plugin(self):
        return object.__new__(HapiConnectorPlugin)

    @mock.patch('pathlib.Path.read_text', return_value='{"allowed_private_sessions":["fixture:FriendMessage:1"]}')
    async def test_private_allowlist_and_no_mutating_commands(self, _):
        p=self.plugin()
        event=Event()
        self.assertTrue(p._is_admin(event))
        denied=[x async for x in p.handle_hapi(event,'create')]
        self.assertTrue(any('仅允许只读' in x for x in denied))
        event.is_private_chat=lambda:False
        self.assertFalse(p._is_admin(event))
        event.is_private_chat=lambda:True
        event.is_admin=lambda:False
        self.assertFalse(p._is_admin(event))

    @mock.patch('pathlib.Path.read_text', side_effect=FileNotFoundError)
    async def test_missing_allowlist_fails_closed(self, _):
        self.assertFalse(self.plugin()._is_admin(Event()))

    @mock.patch('assistant_hapi_local.psutil.net_connections')
    @mock.patch('assistant_hapi_local.subprocess.Popen')
    @mock.patch('assistant_hapi_local.verified_hub', return_value=False)
    def test_does_not_start_on_occupied_port(self, _, spawn, connections):
        connections.return_value=[type('C',(),{'status':local.psutil.CONN_LISTEN,'laddr':type('A',(),{'port':3006})()})()]
        with mock.patch.object(local.Path,'is_file',return_value=True):
            self.assertFalse(local.ensure_hub())
        spawn.assert_not_called()
