"""Same-fixture native/OAI conformance benchmark; no QQ or Harness writes."""
import argparse
import base64
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
API='http://127.0.0.1:11434'


def request_json(path,payload):
    req=urllib.request.Request(API+path,data=json.dumps(payload,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=120) as response:return json.load(response)


def gpu_memory():
    try:
        flags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess,'CREATE_NO_WINDOW') else 0
        text=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],creationflags=flags,timeout=3,text=True)
        return int(text.strip().splitlines()[0])
    except Exception:return None


def generate_fixture():
    from PIL import Image,ImageDraw,ImageFont
    image=Image.new('RGB',(400,260),'white');draw=ImageDraw.Draw(image)
    for x in (25,135,245):draw.ellipse((x,30,x+65,95),fill='blue')
    draw.rectangle((150,130,225,205),fill='red')
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',28)
    draw.text((15,220),'CODE 42',fill='black',font=font)
    path=ROOT/'data/harness/vision-fixture.png';image.save(path)
    return base64.b64encode(path.read_bytes()).decode()


FIXTURES=[
    ('greeting',[{'role':'user','content':'在吗，今天过得咋样'}]),
    ('banter',[{'role':'user','content':'我的待办清单又繁殖了，离谱'}]),
    ('multi_turn',[{'role':'user','content':'我在调 QQ 机器人，一直卡在登录重连'},{'role':'assistant','content':'重连那块还没稳住？'},{'role':'user','content':'刚才终于好了'}]),
    ('reference',[{'role':'user','content':'我说的那个项目叫 REA，先只查问题，不改代码'},{'role':'assistant','content':'明白，先看问题。'},{'role':'user','content':'上次那个接下来怎么处理？'}]),
    ('short_stop',[{'role':'user','content':'这个报错烦死了'},{'role':'assistant','content':'想先放一会儿吗？'},{'role':'user','content':'算了，不聊这个了'}]),
    ('correction',[{'role':'user','content':'我说的是无糖，不是少糖'}]),
    ('computer_status',[{'role':'user','content':'刚才让你开的软件打开了吗？'}]),
    ('harness_ambiguity',[{'role':'user','content':'把“继续修复”交给 Codex。现在有两个 Codex 会话，REA 和天体模拟器。'}]),
    ('link_unavailable',[{'role':'user','content':'这个网页讲什么 https://example.invalid/unavailable'}]),
]


def native_call(profile,messages,tools=None,format_json=False):
    options={k:v for k,v in profile['generation'].items() if k!='max_tokens'}
    options.update(num_ctx=8192,num_thread=4,num_predict=384)
    payload={'model':profile['model'],'messages':messages,'stream':True,'think':False,'options':options,'keep_alive':'2m'}
    if tools:payload['tools']=tools
    if format_json:payload['format']='json'
    started=time.monotonic();first=None;text='';thinking='';calls=[];last={}
    req=urllib.request.Request(API+'/api/chat',data=json.dumps(payload,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=120) as response:
        for line in response:
            chunk=json.loads(line);message=chunk.get('message',{})
            if first is None and (message.get('content') or message.get('tool_calls')):first=time.monotonic()-started
            text+=message.get('content','');thinking+=message.get('thinking','')
            calls.extend(message.get('tool_calls',[]));last=chunk
    duration=time.monotonic()-started
    return {'text':text,'tool_calls':calls,'thinking_chars':len(thinking),'seconds':round(duration,3),'ttft_seconds':round(first,3) if first else None,
            'eval_count':last.get('eval_count'),'tokens_per_second':round(last.get('eval_count',0)*1e9/last['eval_duration'],2) if last.get('eval_duration') else None,
            'load_seconds':round(last.get('load_duration',0)/1e9,3),'gpu_used_mb_after':gpu_memory()}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--profile',choices=['qwen3','qwen35'],required=True);args=parser.parse_args()
    profile=json.loads((ROOT/'config/model_profiles.json').read_text(encoding='utf-8'))[args.profile]
    report={'profile':args.profile,'model':profile['model'],'collected_at':datetime.now(timezone.utc).isoformat(),'context_tokens':8192,'results':[]}
    metadata=request_json('/api/show',{'model':profile['model']});report['model_details']=metadata['details'];report['capabilities']=metadata.get('capabilities',[])
    system=(ROOT/profile['persona_file']).read_text(encoding='utf-8')
    system+='\n本测试没有电脑操作工具，也没有实际执行过电脑操作。没有提供的网页和图片内容不可猜测。'
    for name,history in FIXTURES:
        try:
            result=native_call(profile,[{'role':'system','content':system},*history]);result['case']=name
            result['checks']={'nonempty':bool(result['text']),'thinking_disabled':result['thinking_chars']==0,'no_default_report':not result['text'].lstrip().startswith(('#','1.','首先'))}
        except Exception as exc:result={'case':name,'error_type':type(exc).__name__,'error':str(exc)[:180]}
        report['results'].append(result);print(json.dumps(result,ensure_ascii=False),flush=True)
    tool={'type':'function','function':{'name':'harness_query','description':'查询编程 Harness 的额度或已有会话，只读',
          'parameters':{'type':'object','properties':{'operation':{'type':'string','enum':['quota','sessions']},'harness':{'type':'string','enum':['codex','workbuddy','zcode']}},'required':['operation','harness']}}}
    for name,prompt,tools,formatted in [('tool_routing','查一下 Codex 的剩余额度',[tool],False),('json','只返回 JSON：harness 为 codex，operation 为 quota。',None,True)]:
        try:
            result=native_call(profile,[{'role':'system','content':system},{'role':'user','content':prompt}],tools,formatted);result['case']=name
            if formatted:result['parsed_json']=json.loads(result['text'])
            result['checks']={'thinking_disabled':result['thinking_chars']==0,'valid_tool_call':bool(result['tool_calls']) if tools else True}
        except Exception as exc:result={'case':name,'error_type':type(exc).__name__,'error':str(exc)[:180]}
        report['results'].append(result);print(json.dumps(result,ensure_ascii=False),flush=True)
    if 'vision' in report['capabilities']:
        encoded=generate_fixture()
        prompt='只返回 JSON，统计图片：blue_circles 是蓝色圆形数量，red_squares 是红色方块数量，code 是图片中的数字。'
        try:
            result=native_call(profile,[{'role':'user','content':prompt,'images':[encoded]}],format_json=True);result['case']='native_vision';result['parsed_json']=json.loads(result['text'])
            result['checks']={'blue_circles':result['parsed_json'].get('blue_circles')==3,'red_squares':result['parsed_json'].get('red_squares')==1,'code':str(result['parsed_json'].get('code'))=='42'}
        except Exception as exc:result={'case':'native_vision','error_type':type(exc).__name__,'error':str(exc)[:180]}
        report['results'].append(result);print(json.dumps(result,ensure_ascii=False),flush=True)
        payload={'model':profile['model'],'messages':[{'role':'user','content':[{'type':'text','text':prompt},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+encoded}},{'type':'text','text':'当前是隔离验证场景。'}]}],
                 'stream':False,'reasoning_effort':'none','response_format':{'type':'json_object'},'max_tokens':384,'temperature':0.7,'top_p':0.8}
        try:
            request=urllib.request.Request('http://127.0.0.1:8899/v1/chat/completions',data=json.dumps(payload,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
            with urllib.request.urlopen(request,timeout=120) as response:result=json.load(response)
            text=result['choices'][0]['message']['content'];parsed=json.loads(text)
            result={'case':'openai_proxy_vision','text':text,'parsed_json':parsed,'passed':parsed.get('blue_circles')==3 and parsed.get('red_squares')==1 and str(parsed.get('code'))=='42'}
        except Exception as exc:result={'case':'openai_proxy_vision','error_type':type(exc).__name__,'error':str(exc)[:180]}
        report['results'].append(result);print(json.dumps(result,ensure_ascii=False),flush=True)
    else:report['results'].append({'case':'vision','state':'Unsupported','reason':'Text-only model; image not sent and content not invented'})
    report['loaded_models']=json.load(urllib.request.urlopen(API+'/api/ps',timeout=5))
    output=ROOT/f'data/logs/benchmark-{args.profile}.json';output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('SAVED',output,flush=True)


if __name__=='__main__':main()
