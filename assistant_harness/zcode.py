"""Read-only ZCode session discovery; never attach or resume desktop sessions."""
import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess

from .adapters import Capability, Result


def normalize_zcode_sessions(raw, limit=20):
    rows=[]
    for entry in raw[:limit]:
        if not isinstance(entry,dict) or not isinstance(entry.get('sessionId'),str):
            continue
        workspace=entry.get('workspace')
        rows.append({'harness':'zcode','native_id':entry['sessionId'],
                     'project_path':workspace.get('workspacePath') if isinstance(workspace,dict) else None,
                     'title':entry.get('title') or '未命名会话',
                     # A separate CLI's idle status does not establish desktop liveness.
                     'status':'unknown','status_scope':'independent_app_server',
                     'can_send':False})
    return rows


class ZCodeAdapter:
    name='zcode'

    def __init__(self, root=None, node=None):
        self.root=Path(root or Path(os.environ.get('ProgramFiles',r'C:\Program Files'))/'ZCode/resources')
        self.node=node or shutil.which('node')
        self._lock=asyncio.Lock()

    async def start(self): pass
    async def close(self): pass

    async def list_sessions(self,limit=20):
        cli=self.root/'glm/zcode.cjs'
        entry=self.root/'config/default.json'
        provider=self.root/'config/provider/zcode-builtin.json'
        if not self.node or not all(path.is_file() for path in (cli,entry,provider)):
            return Result(Capability.UNAVAILABLE,reason='本机 ZCode CLI、Node 或官方内置配置不可用。')
        # The packaged CLI expects its provider config alongside its entrypoint;
        # its bundled location differs. Supply the actual installed config path.
        bootstrap='process.argv=[process.execPath,'+json.dumps(str(entry))+',"app-server"];require('+json.dumps(str(cli))+')'
        env=dict(os.environ)
        for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
            env[name]='4'
        flags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name=='nt' else 0
        async with self._lock:
            proc=None
            try:
                proc=await asyncio.create_subprocess_exec(self.node,'-e',bootstrap,stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.DEVNULL,env=env,creationflags=flags)
                proc.stdin.write(b'{"id":1,"method":"session/list","params":{}}\n')
                await proc.stdin.drain()
                async def response():
                    while line:=await proc.stdout.readline():
                        message=json.loads(line)
                        if message.get('id')==1:
                            if 'error' in message:raise ValueError('session/list rejected')
                            data=message.get('result')
                            if not isinstance(data,dict) or not isinstance(data.get('sessions'),list):raise ValueError('unexpected session/list result')
                            return data['sessions']
                    raise ConnectionError('ZCode app-server closed')
                rows=await asyncio.wait_for(response(),10)
                return Result(Capability.SUPPORTED,normalize_zcode_sessions(rows,max(1,min(limit,50))))
            except (OSError,ValueError,ConnectionError,asyncio.TimeoutError):
                return Result(Capability.UNAVAILABLE,reason='ZCode 只读会话发现失败；桌面实时状态未知。')
            finally:
                if proc and proc.returncode is None:
                    proc.terminate()
                    try:await asyncio.wait_for(proc.wait(),3)
                    except asyncio.TimeoutError:
                        proc.kill();await proc.wait()

    async def read_session(self,identity):
        return Result(Capability.UNSUPPORTED,reason='尚未连接桌面会话的权威实时通道，不能读取实时状态。')

    async def quota(self):
        return Result(Capability.UNSUPPORTED,reason='尚未确认可用的 ZCode 额度接口。')

    async def send(self,identity,text):
        return Result(Capability.UNSUPPORTED,reason='尚未连接该桌面会话的权威控制通道，不能发送或并发接管。')
