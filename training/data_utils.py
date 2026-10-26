"""Official template rendering with assistant-only supervision and no truncation."""
import json
from pathlib import Path


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def encode(tokenizer, row, max_length=1024):
    text=tokenizer.apply_chat_template(row['messages'],tools=row.get('tools'),tokenize=False,
        add_generation_prompt=False,enable_thinking=False)
    spans=[]
    cursor=0
    header='<|im_start|>assistant\n'
    while True:
        start=text.find(header,cursor)
        if start<0:break
        start+=len(header)
        end=text.find('<|im_end|>',start)
        if end<0:raise ValueError('Unterminated assistant turn')
        end+=len('<|im_end|>')
        spans.append((start,end))
        cursor=end
    encoded=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)
    if len(encoded['input_ids'])>max_length:
        raise ValueError(f"{row['id']} has {len(encoded['input_ids'])} tokens > {max_length}; no silent truncation")
    labels=[token if any(start<=a and b<=end and b>a for start,end in spans) else -100
            for token,(a,b) in zip(encoded['input_ids'],encoded['offset_mapping'])]
    assert sum(t!=-100 for t in labels)>1,'No supervised tokens'
    return {'input_ids':encoded['input_ids'],'attention_mask':encoded['attention_mask'],'labels':labels}


def batch(torch, encoded, device='cuda'):
    return {k:torch.tensor([v],dtype=torch.long,device=device) for k,v in encoded.items()}
