"""Validate roles/tools, duplicates, family leakage, token masks and length."""
import argparse
from collections import Counter
import hashlib
import json
import re
from common import LOCAL, configure, save_json
from data_utils import read_rows, encode


def fingerprint(row):
    return hashlib.sha256(json.dumps({'messages':row['messages'],'tools':row.get('tools')},sort_keys=True,ensure_ascii=False).encode()).hexdigest()


def validate(splits,tokenizer=None,max_length=1024):
    seen={};families={};ids=set();report={};errors=[]
    for split,rows in splits.items():
        lengths=[];targets=[];duplicate_responses=Counter()
        for row in rows:
            key=row['id']
            if key in ids:errors.append(f'Duplicate id: {key}')
            ids.add(key)
            h=fingerprint(row)
            if h in seen:errors.append(f'Duplicate dialogue: {key}, {seen[h]}')
            seen[h]=(split,key)
            family=row['family']
            if family in families and families[family]!=split:errors.append(f'Family leakage: {family}')
            families[family]=split
            messages=row['messages']
            if messages[0]['role']!='system' or messages[-1]['role']!='assistant':errors.append(f'Bad boundaries {key}')
            pending={};used_tools={t['function']['name']:t['function'] for t in row.get('tools',[])}
            for i,msg in enumerate(messages):
                if msg['role'] not in {'system','user','assistant','tool'}:errors.append(f'Bad role {key}')
                if re.search(r'<\|im_(?:start|end)\|>|<tool_call>|<tool_response>',msg.get('content','')):errors.append(f'Reserved token injection {key}')
                if msg['role']=='system' and i:errors.append(f'System in middle {key}')
                if msg['role']=='assistant':
                    duplicate_responses[msg['content']]+=1
                    for call in msg.get('tool_calls',[]):
                        fn=call['function'];schema=used_tools.get(fn['name'])
                        if not schema:errors.append(f'Unknown tool {key}');continue
                        arguments=fn['arguments']
                        if not isinstance(arguments,dict):errors.append(f'Non-object arguments {key}');continue
                        if not set(schema['parameters'].get('required',[]))<=arguments.keys():errors.append(f'Missing argument {key}')
                        pending[call['id']]=fn['name']
                elif msg['role']=='tool':
                    if pending.pop(msg['tool_call_id'],None)!=msg.get('name'):errors.append(f'Unpaired tool {key}')
                elif pending:errors.append(f'Unresolved tool before next turn {key}')
            if pending:errors.append(f'Unresolved tools {key}')
            if tokenizer:
                try:
                    item=encode(tokenizer,row,max_length)
                    lengths.append(len(item['input_ids']));targets.append(sum(v!=-100 for v in item['labels']))
                except Exception as exc:errors.append(str(exc))
        report[split]={'records':len(rows),'families':len({r['family'] for r in rows}),
            'max_tokens':max(lengths,default=None),'mean_tokens':sum(lengths)/len(lengths) if lengths else None,
            'supervised_tokens':sum(targets),'repeated_assistant_strings':[(s,n) for s,n in duplicate_responses.most_common(10) if n>1 and s]}
    report['passed']=not errors;report['errors']=errors
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--size',default='9B');p.add_argument('--data',default='experimental-v1');args=p.parse_args()
    configure()
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(LOCAL/'models'/f'Qwen3.5-{args.size}',local_files_only=True)
    directory=LOCAL/'data'/args.data
    splits={s:read_rows(directory/f'{s}.jsonl') for s in ('train','validation','test')}
    report=validate(splits,tokenizer)
    report['template_sha256']=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
    save_json(directory/'validation.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
