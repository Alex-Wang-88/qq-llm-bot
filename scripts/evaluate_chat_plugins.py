"""Isolated, synthetic A/B sample for candidate chat plugins; no QQ or DB writes."""
import asyncio
import ctypes
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import urllib.request

if os.name == 'nt':
    ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x00004000)
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='4'

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import verify_chat_tuning as baseline

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module

enhancer = load('candidate_enhancer', ROOT / 'data/plugin-evaluation/enhancer/main.py')
sys.path.insert(0, str(ROOT / 'data/plugin-evaluation/lingxi'))
from lx.prompt.assembler import DefaultAssembler
from lx.contracts import SessionKey, Scope, Source, Trigger


class FixtureEvent:
    unified_msg_origin = 'fixture:FriendMessage:synthetic'
    message_str = ''
    def get_sender_id(self): return 'synthetic-user'
    def get_sender_name(self): return '虚构测试者'
    def get_group_id(self): return ''
    def is_private_chat(self): return True


CASES = [
    ('疲惫闲聊', [{'role':'user','content':'今天好累，连着开了四个会。'}]),
    ('多轮纠错', [{'role':'user','content':'我喜欢无糖咖啡。'}, {'role':'assistant','content':'你喜欢少糖咖啡对吧？'}, {'role':'user','content':'不是少糖，是无糖。'}]),
    ('结束边界', [{'role':'user','content':'今天有点累。'}, {'role':'assistant','content':'辛苦了。'}, {'role':'user','content':'我先躺会儿，不想听建议，也别再问了。'}]),
]


def infer(system, messages):
    payload = {'model':'qwen3.5:4b-qq-v2','messages':[{'role':'system','content':system},*messages],
               'stream':False,'think':False,'keep_alive':'90s',
               'options':{'num_thread':4,'num_ctx':8192,'num_predict':180,'temperature':0.4,'seed':42}}
    req = urllib.request.Request('http://127.0.0.1:11434/api/chat',
                                 data=json.dumps(payload,ensure_ascii=False).encode('utf-8'),
                                 headers={'Content-Type':'application/json'})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req,timeout=120) as response:
        result=json.load(response)
    return result['message']['content']


async def main():
    samples=[]
    safe='--safe' in sys.argv
    with tempfile.TemporaryDirectory(prefix='qianwen-fixtures-') as tmp:
        baseline.module.SESSION_STATE=Path(tmp)/'synthetic-session.json'
        for case,messages in CASES:
            req=await baseline.build_request(messages[-1]['content'])
            base=req.system_prompt
            for mode in (('enhancer_safe','lingxi_safe') if safe else ('baseline','enhancer','lingxi')):
                prompt=base
                if mode in ('enhancer','enhancer_safe'):
                    p=enhancer.ChatEnhancerPlugin(None, {'auto_purge_history':False, 'debug_dump_request':False})
                    if mode=='enhancer_safe': p._apply_identity_header=lambda *args: None
                    fake=SimpleNamespace(system_prompt=prompt,contexts=[dict(m) for m in messages])
                    await p.on_llm_request(FixtureEvent(),fake)
                    prompt=fake.system_prompt
                elif mode in ('lingxi','lingxi_safe'):
                    key=SessionKey(Scope.PRIVATE,'fixture','fixture')
                    trigger=Trigger(source=Source.DIRECT,reason='虚构测试对话',silent_ok=False,meta={})
                    options={'trigger.next_reply_enabled':False,'context.status_note_enabled':False} if mode=='lingxi_safe' else {}
                    prompt+='\n'+DefaultAssembler()._build_system_tail(key,trigger,options)
                result=await asyncio.to_thread(infer,prompt,messages)
                samples.append({'case':case,'variant':mode,'system_chars':len(prompt),'response':result})
                print(f'{case}: {mode} complete',flush=True)
    out=ROOT/'data/plugin-evaluation'/('chat-ab-safe.json' if safe else 'chat-ab.json')
    out.write_text(json.dumps({'note':'全部对话均为虚构测试；对比提示与输出，不等同完整插件或 QQ 端验收。','samples':samples},ensure_ascii=False,indent=2),encoding='utf-8')
    print('Results:',out,flush=True)

if __name__=='__main__': asyncio.run(main())
