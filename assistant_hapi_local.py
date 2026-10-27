"""Launch and identify the project-local HAPI hub without exposing its token."""
import json
import os
from pathlib import Path
import subprocess
import time

import psutil

ROOT = Path(__file__).resolve().parent
EXE = ROOT / 'data/hapi-runtime/node_modules/@twsxtd/hapi-win32-x64/bin/hapi.exe'
SETTINGS = Path.home() / '.hapi/settings.json'
PORT = 3006


def verified_hub():
    try:
        for c in psutil.net_connections(kind='tcp'):
            if c.status != psutil.CONN_LISTEN or not c.laddr or c.laddr.port != PORT:
                continue
            if c.laddr.ip != '127.0.0.1' or c.pid is None:
                return False
            p=psutil.Process(c.pid)
            args=p.cmdline()
            return Path(p.exe()).resolve() == EXE.resolve() and 'hub' in args and '--no-relay' in args
    except (psutil.Error, OSError):
        return False
    return False


def ensure_hub():
    if verified_hub():
        return True
    try:
        if not EXE.is_file():
            return False
        # Never start a duplicate or use an unrelated service on this port.
        if any(c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == PORT for c in psutil.net_connections(kind='tcp')):
            return False
        flags = getattr(subprocess, 'DETACHED_PROCESS', 0) | getattr(subprocess, 'CREATE_NEW_PROCESS_GROUP', 0) | getattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS', 0)
        subprocess.Popen([str(EXE),'hub','--no-relay'],cwd=str(ROOT),stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,creationflags=flags)
        for _ in range(20):
            time.sleep(.25)
            if verified_hub(): return True
    except (psutil.Error, OSError):
        pass
    return False


def local_access_token():
    if not verified_hub(): return ''
    try:
        data=json.loads(SETTINGS.read_text(encoding='utf-8'))
        token=data.get('cliApiToken','')
        return token if isinstance(token,str) else ''
    except (OSError, ValueError):
        return ''
