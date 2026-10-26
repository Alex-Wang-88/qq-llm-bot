"""Controlled A/B/C generation. Heuristics flag review; they are not human ratings."""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import time
from common import LOCAL,REPO,append_jsonl,configure,memory,save_json
from build_data import TOOLS
from data_utils import read_rows
from model_utils import load


def fixtures():
    spec=importlib.util.spec_from_file_location('original_benchmark',REPO/'scripts/benchmark_models.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=[{'id':name,'messages':history,'source':'scripts/benchmark_models.py'} for name,history in module.FIXTURES]
    # Test targets never appear in training. Expected values are mechanical checks,
    # while naturalness/persona/factual grounding require reviewing raw outputs.
    rows += [{'id':'heldout-'+r['id'],'messages':r['messages'][1:-1],'source':'heldout_synthetic_test','reference':r['messages'][-1]['content']} for r in read_rows(LOCAL/'data/experimental-v1/test.jsonl')]
    rows += [
      {'id':'detail_streaming','messages':[{'role':'user','content':'请详细解释分段发送和流式输出的区别，并说怎么验证。'}]},
      {'id':'fact_math','messages':[{'role':'user','content':'只算结果：23 × 7 - 18。'}],'expected':['143']},
      {'id':'fact_code','messages':[{'role':'user','content':'Python 中 a=[1]; b=a; a.append(2)，此时 b 是什么？'}],'expected':['1','2']},
      {'id':'no_image','messages':[{'role':'user','content':'我还没上传照片，你说说照片里的窗帘是什么颜色。'}]},
      {'id':'tool_quota','messages':[{'role':'user','content':'查一下 Codex 的剩余额度。'}],'tools':TOOLS,'expected_tool':{'name':'harness_query','arguments':{'harness':'codex','operation':'quota'}}},
      {'id':'tool_search','messages':[{'role':'user','content':'测试情境：联网查一下虚构应用 MossChat 的最新版本。'}],'tools':TOOLS,'expected_tool':{'name':'web_search'}},
    ]
    for status in ('success','failure'):
        result={'fixture':True,'version':'8.7','changes':['修复复制失败','新增草稿恢复'],'url':'https://example.invalid/mosschat/8.7'} if status=='success' else {'fixture':True,'error':'timeout','results':[]}
        rows.append({'id':'tool_result_'+status,'tools':TOOLS,'messages':[
          {'role':'user','content':'固定虚构资料测试：MossChat 最新版改了什么？'},
          {'role':'assistant','content':'','tool_calls':[{'id':'eval-fixed','type':'function','function':{'name':'web_search','arguments':{'query':'MossChat 最新版'}}}]},
          {'role':'tool','tool_call_id':'eval-fixed','name':'web_search','content':json.dumps(result,ensure_ascii=False)}],
          **({'expected':['8.7','复制','草稿']} if status=='success' else {})})
    return rows


def parse_calls(text):
    calls=[]
    for block in re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>',text,re.S):
        fn=re.fullmatch(r'<function=([\w]+)>\s*(.*?)\s*</function>',block,re.S)
        if not fn:continue
        arguments={k:v.strip() for k,v in re.findall(r'<parameter=([\w]+)>\s*(.*?)\s*</parameter>',fn[2],re.S)}
        calls.append({'name':fn[1],'arguments':arguments})
    return calls


def checks(text,case):
    calls=parse_calls(text)
    result={'nonempty':bool(text.strip()),'characters':len(text),'question_marks':text.count('？')+text.count('?'),
      'ends_with_question':bool(re.search(r'[？?][”"]?\s*$',text)),
      'stock_phrase':bool(re.search(r'有什么(可以|需要).*帮|很高兴为您|希望(以上|这些).*帮助|随时(告诉|联系|找)我|还有(其他|什么).*问题',text)),
      'tool_calls':calls,'tool_announcement':bool(calls and text.split('<tool_call>')[0].strip())}
    if 'expected' in case:result['mechanical_fact_check']=all(s in text for s in case['expected'])
    if 'expected_tool' in case:
        expected=case['expected_tool']
        result['tool_format_and_route']=any(c['name']==expected['name'] and all(c['arguments'].get(k)==v for k,v in expected.get('arguments',{}).items()) for c in calls)
    result['requires_human_review']=['naturalness','context_grounding','persona','unnecessary_question','unsupported_actions_or_experiences','factual_accuracy']
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--size',default='4B');p.add_argument('--cpu-layers',type=int,default=0)
    p.add_argument('--adapter',required=True);p.add_argument('--output',default='comparison-v1');p.add_argument('--limit',type=int);p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    destination=LOCAL/'evaluation'/args.output
    if (destination/'outputs.jsonl').exists() and not args.resume:raise RuntimeError('Evaluation exists; use --resume or a new output name')
    torch=configure();model,tokenizer,_=load(torch,args.size,args.cpu_layers,adapter=args.adapter)
    model.eval();model.config.use_cache=True
    persona=(REPO/'config/qianwen35_persona.txt').read_text(encoding='utf-8')
    cases=fixtures()
    if args.limit:cases=cases[:args.limit]
    destination.mkdir(parents=True,exist_ok=True)
    save_json(destination/'fixtures.json',cases)
    config={'base_size':args.size,'adapter':args.adapter,'seed':3407,'temperature':.7,'top_p':.8,'top_k':20,
      'max_new_tokens':256,'persona_sha256':hashlib.sha256(persona.encode()).hexdigest(),
      'note':'A no persona/base; B same base + shipped persona; C adapter + identical persona. No real tool calls are executed.'}
    if args.resume and (destination/'config.json').exists():
        previous=json.loads((destination/'config.json').read_text(encoding='utf-8'))
        if previous!=config:raise RuntimeError('Evaluation config changed; refusing mixed results')
    save_json(destination/'config.json',config)
    rows=read_rows(destination/'outputs.jsonl') if args.resume and (destination/'outputs.jsonl').exists() else []
    completed={(r['variant'],r['case']) for r in rows}
    for variant in ('A','B','C'):
        for index,case in enumerate(cases):
            if (variant,case['id']) in completed:continue
            torch.manual_seed(3407+index);torch.cuda.manual_seed_all(3407+index)
            messages=([{'role':'system','content':persona}] if variant!='A' else [])+case['messages']
            ids=tokenizer.apply_chat_template(messages,tools=case.get('tools'),add_generation_prompt=True,
                enable_thinking=False,tokenize=True,return_tensors='pt',return_dict=False).to('cuda')
            context=nullcontext() if variant=='C' else model.disable_adapter()
            torch.cuda.reset_peak_memory_stats();tick=time.perf_counter()
            with context,torch.inference_mode():
                generated=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=384 if 'detail' in case['id'] else 256,
                    do_sample=True,temperature=.7,top_p=.8,top_k=20,use_cache=True,pad_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize();seconds=time.perf_counter()-tick
            output=generated[0,ids.shape[1]:];text=tokenizer.decode(output,skip_special_tokens=True)
            row={'variant':variant,'case':case['id'],'text':text,'seconds':seconds,'input_tokens':ids.shape[1],
              'output_tokens':len(output),'tokens_per_second':len(output)/seconds,'memory':memory(torch),'checks':checks(text,case),
              'hit_token_limit':len(output)>=(384 if 'detail' in case['id'] else 256)}
            append_jsonl(destination/'outputs.jsonl',row);rows.append(row)
            print(json.dumps(row,ensure_ascii=False),flush=True)
    summary={}
    for variant in ('A','B','C'):
        subset=[r for r in rows if r['variant']==variant]
        summary[variant]={'cases':len(subset),'stock_phrase_rate':sum(r['checks']['stock_phrase'] for r in subset)/len(subset),
            'ending_question_rate':sum(r['checks']['ends_with_question'] for r in subset)/len(subset),
            'average_characters':sum(len(r['text']) for r in subset)/len(subset),
            'tool_routing_passed':sum(r['checks'].get('tool_format_and_route',False) for r in subset),
            'tool_routing_total':sum('tool_format_and_route' in r['checks'] for r in subset),
            'mechanical_fact_passed':sum(r['checks'].get('mechanical_fact_check',False) for r in subset),
            'mechanical_fact_total':sum('mechanical_fact_check' in r['checks'] for r in subset),
            'truncated':sum(r['hit_token_limit'] for r in subset)}
    save_json(destination/'summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
