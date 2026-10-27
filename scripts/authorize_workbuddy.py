"""One-time local authorization using the installed plugin's OAuth flow.

The URL is shown to the local user; tokens are saved only by the plugin's
encrypted CredentialStore. No message is sent through QQ.
"""
import asyncio
import ctypes
import importlib
import os
from pathlib import Path
import sys
import winreg

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'data/plugins'))
if os.name == 'nt':
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x00004000)
with winreg.OpenKey(winreg.HKEY_CURRENT_USER,'Environment') as key:
    master,_=winreg.QueryValueEx(key,'WORKBUDDY_MASTER_KEY')

import aiohttp
from astrbot_plugin_workbuddy.core.auth import AuthManager
from astrbot_plugin_workbuddy.core.storage import CredentialStore

async def main():
    store=CredentialStore(ROOT/'data/plugin_data/astrbot_plugin_workbuddy',master)
    if store.load():
        print('ALREADY_AUTHORIZED',flush=True)
        return
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
        auth=AuthManager(session,store,chat_base='https://copilot.tencent.com',poll_interval=3,login_timeout=600)
        try:
            pending=await auth.begin_login()
            print('AUTH_URL '+pending['authUrl'],flush=True)
            await auth.complete_login(pending['state'])
            print('AUTHORIZED',flush=True)
        except Exception as exc:
            print('AUTH_FAILED '+type(exc).__name__,flush=True)
            raise SystemExit(1)

if __name__=='__main__': asyncio.run(main())
