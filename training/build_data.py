"""Build reproducible, explicitly synthetic data; split semantic families before variants."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import random
from common import LOCAL, REPO, save_json

SYSTEM = '你是千问，用户电脑里的私人 AI 助手。在 QQ 像熟人一样聊天，温和机灵，有判断。承接具体上下文，按需展开，允许结束。只依据对话和真实工具结果说话；当前只可搜索、读网页和只读查询，执行状态如实说明。'
TOOLS = [
    {'type':'function','function':{'name':'web_search','description':'搜索实时网页信息','parameters':{'type':'object','properties':{'query':{'type':'string'}},'required':['query'],'additionalProperties':False}}},
    {'type':'function','function':{'name':'fetch_web_page','description':'读取指定网页正文','parameters':{'type':'object','properties':{'url':{'type':'string'}},'required':['url'],'additionalProperties':False}}},
    {'type':'function','function':{'name':'harness_query','description':'只读查询已有会话或额度，不能发送任务','parameters':{'type':'object','properties':{'operation':{'type':'string','enum':['sessions','quota']},'harness':{'type':'string','enum':['codex','workbuddy','zcode']}},'required':['operation','harness'],'additionalProperties':False}}},
]


def dialogue(category, family, turns, tools=None):
    result={'id':family,'family':family,'category':category,'source':'original_synthetic_v1',
            'messages':[{'role':'system','content':SYSTEM},*[{'role':('user' if i%2==0 else 'assistant'),'content':s} for i,s in enumerate(turns)]]}
    if tools: result['tools']=tools
    return result


def make_records():
    rows=[]
    for line in (REPO/'training/scenarios.tsv').read_text(encoding='utf-8').splitlines():
        if not line or line.startswith('#'): continue
        category,*turns=line.split('|')
        assert len(turns)==4, line
        turns=[s.replace('\\n','\n') for s in turns]
        rows.append(dialogue(category,f'curated-{len(rows):03}',turns))
    # Structured variations use concrete, checked targets. All variants of one
    # family remain in one split; name/number changes never count as new families.
    names=['小陆','小孟','小乔','小周','小夏','小丁','小姚','小田','小范','小吴','小叶','小余']
    projects=['青柚','微光','岸桥','书页','星河','晴川','山岚','拾光','蓝帆','橙子','小溪','远帆']
    for i,(name,project) in enumerate(zip(names,projects)):
        cases=[
            ('context', 'project-scope', [f'项目叫{project}，这次只检查日志，不动程序。',f'{project}先只看日志。',f'{name}说那个还没好，能直接重启吗？',f'先别直接重启，{project}这轮定的是只检查日志。先看最新报错和时间，再决定下一步。']),
            ('correction', 'number-correction', [f'我要订{i+3}份饭。',f'先按{i+3}份算。',f'有两人不来了，按现在人数说。',f'那就是{i+1}份，前面的数量要减掉两份。']),
            ('fact','arithmetic-chain',[f'每盒{i+4}支笔，我有三盒，一共多少？',f'一共{(i+4)*3}支。',f'给{name}五支，还剩多少？',f'还剩{(i+4)*3-5}支。']),
            ('context','order-correction',[f'{name}要热茶，我要冰茶。',f'{name}热茶，你冰茶。','我那份改常温，另一份不变。',f'现在你是常温茶，{name}仍然热茶。']),
            ('honesty','dispatch-boundary',[f'把“检查{project}的日志”交给工具做。',f'我当前只有只读查询工具，没有发送任务的工具，暂时不能替你发出。','那能不能先列一下已有会话？','能用只读查询确认有哪些会话；列出会话也不代表任务已经发出。']),
            ('format','exact-name',[f'记住本轮文件名是 {project}_r{i+1}.txt。',f'本轮按 {project}_r{i+1}.txt 来。','现在只返回这个文件名。',f'{project}_r{i+1}.txt']),
            ('emotion','effort-specific',[f'给{project}整理了{i+2}小时资料，结果会议改期了。','准备都做完才改期，确实有点泄气。','不用安慰，帮我说句客观的话。',f'{project}的资料已经整理完成，会议改期，准备工作可以留待下次使用。']),
            ('context','actor-isolation',[f'{name}负责测试，我负责写文档，{project}的代码另有人改。','测试、文档和代码是三个分工。','所以我现在该去改代码吗？',f'按刚才的分工，你负责{project}的文档；不能把另一个人的改代码任务算到你头上。']),
            ('fact','time-conversion',[f'计时器设了{i+2}分钟，是多少秒？',f'{(i+2)*60}秒。','已经过了半分钟，还剩几秒？',f'还剩{(i+2)*60-30}秒。']),
            ('format','literal-instruction',[f'把这句改口语：关于{project}的相关事宜，后续将进一步讨论。',f'{project}的事，后面再聊。','别把不确定的时间写成明天。',f'{project}的事，之后再讨论，具体时间还没定。']),
            ('context','scope-expansion',[f'{project}只改标题，正文先不动。','先只改标题。','现在我明确同意第二段也改，第一段仍不动。','范围更新了：标题和第二段可以改，第一段保持原样。']),
            ('honesty','memory-limit',[f'我这次出差的代号是{project}，只在这轮用。',f'这轮用{project}这个代号。','你已经存进永久记忆了？','没有长期记忆写入结果；刚才你也限定了只在这轮使用。']),
        ]
        for category,family,turns in cases:
            row=dialogue(category,f'generated-{family}',turns)
            row['id']+=f'-{i:02}'
            rows.append(row)
    # Tool calls are structured messages rendered by the official Qwen template.
    for i in range(12):
        call_id=f'synthetic-{i}'
        query=f'测试应用 LeafDemo{i} 最新发布说明'
        call={'role':'assistant','content':'','tool_calls':[{'id':call_id,'type':'function','function':{'name':'web_search','arguments':{'query':query}}}]}
        result={'role':'tool','tool_call_id':call_id,'name':'web_search','content':json.dumps({'fixture':True,'status':'ok','version':f'2.{i}','changes':['修复离线保存','改善列表滚动'],'url':f'https://example.invalid/leafdemo{i}'},ensure_ascii=False)}
        row=dialogue('tool',f'tool-success-{i}',[])
        row['family']='tool-success'
        row['tools']=TOOLS
        row['messages'] += [{'role':'user','content':f'这是固定虚构资料测试，查 LeafDemo{i} 的版本和主要改动。'},call,result,{'role':'assistant','content':f'这份测试资料里的版本是 2.{i}，主要修复离线保存、改善列表滚动。来源是给定的虚构发布页，不能当真实版本信息。'}]
        rows.append(row)
    for i,operation in enumerate(['quota','sessions']*6):
        name=f'测试服务 RiverDemo{i}'
        row=dialogue('tool',f'tool-failure-{i}',[])
        row['family']='tool-failure'
        row['tools']=TOOLS
        row['messages'] += [{'role':'user','content':f'查一下{name}目前能否访问，这是固定失败测试。'},
            {'role':'assistant','content':'','tool_calls':[{'id':f'fail-{i}','type':'function','function':{'name':'web_search','arguments':{'query':f'{name} 当前状态'}}}]},
            {'role':'tool','name':'web_search','tool_call_id':f'fail-{i}','content':'{"fixture":true,"status":"error","error":"timeout","results":[]}'},
            {'role':'assistant','content':f'这次查询超时了，还不能确认{name}的状态。查询失败本身不能证明服务停了。'}]
        rows.append(row)
    for i,harness in enumerate(['codex','workbuddy','zcode']*4):
        operation='quota' if i%2 else 'sessions'
        row=dialogue('tool',f'harness-{i}',[])
        row['family']=f'harness-{operation}'
        row['tools']=TOOLS
        row['messages'] += [{'role':'user','content':f'这是模拟测试，只读查询 {harness} 的'+('额度。' if operation=='quota' else '已有会话。')},
            {'role':'assistant','content':'','tool_calls':[{'id':f'query-{i}','type':'function','function':{'name':'harness_query','arguments':{'operation':operation,'harness':harness}}}]},
            {'role':'tool','tool_call_id':f'query-{i}','name':'harness_query','content':json.dumps({'fixture':True,'status':'unavailable','reason':'测试环境没有连接'},ensure_ascii=False)},
            {'role':'assistant','content':f'测试环境没有连接到 {harness}，这次没取得'+('额度数据。' if operation=='quota' else '会话列表。')}]
        rows.append(row)
    return rows


def main():
    rows=make_records()
    # Curated examples stratified by category. Generated variants stay together.
    split_for={}
    for category in sorted({r['category'] for r in rows if r['family'].startswith('curated-')}):
        families=[r['family'] for r in rows if r['category']==category and r['family'].startswith('curated-')]
        random.Random('3407-'+category).shuffle(families)
        for j,family in enumerate(families): split_for[family]='test' if j==0 else 'validation' if j==1 else 'train'
    # Generated/tool families train only; independent curated tool tests live in eval fixtures.
    output=LOCAL/'data/experimental-v1'
    hashes={}
    counts={}
    for split in ('train','validation','test'):
        selected=[r for r in rows if split_for.get(r['family'],'train')==split]
        data=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in selected)
        output.mkdir(parents=True,exist_ok=True)
        (output/f'{split}.jsonl').write_text(data,encoding='utf-8')
        hashes[split]=hashlib.sha256(data.encode()).hexdigest()
        counts[split]={'records':len(selected),'categories':dict(Counter(r['category'] for r in selected))}
    manifest={'name':'experimental-v1','seed':3407,'total':len(rows),'counts':counts,'sha256':hashes,
        'sources':['config/qianwen35_persona.txt (behavior specification)','活人感诊断.md (scenario requirements)','training/scenarios.tsv (original synthetic)','training/build_data.py (deterministic original synthetic variations)'],
        'private_chat_accessed':False,'real_chat_validated':False,'human_reviewed':False,
        'limitations':['Synthetic pilot only; no real QQ history','Generated/tool families currently train only; evaluate tools separately','Reasoning traces absent; reasoning preservation is unverified','Curated split is small, synthetic and authored by same generator; not a production benchmark']}
    save_json(output/'manifest.json',manifest)
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
