"""Merge only after a successful completed run; preserve original and adapter."""
import argparse
import json
from pathlib import Path
import shutil
import time
from common import LOCAL,configure,memory,save_json,stamp
from model_utils import load


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--adapter');args=p.parse_args()
    run=Path(args.run).resolve();report=json.loads((run/'summary.json').read_text(encoding='utf-8'))
    if report['status']!='completed':raise RuntimeError('Refusing to merge an incomplete run')
    if not report.get('adapter_parameter_delta_l2_sum',0)>0:raise RuntimeError('No proved parameter update')
    config=report['config'];target=run/'merged-bf16'
    if target.exists():raise RuntimeError('Output already exists; not overwriting')
    if shutil.disk_usage(run).free<25*2**30:raise RuntimeError('Need at least 25 GiB free disk')
    torch=configure(config['seed']);tick=time.perf_counter()
    adapter=Path(args.adapter).resolve() if args.adapter else run/'adapter'
    model,tokenizer,_=load(torch,config['size'],config['cpu_layers'],adapter=adapter)
    model.eval();ids=torch.tensor([tokenizer.encode('准确回答：七乘九等于多少？')],device='cuda')
    with torch.no_grad():before=model(input_ids=ids,use_cache=False,logits_to_keep=1).logits.float().cpu()
    merged=model.merge_and_unload(safe_merge=True)
    with torch.no_grad():after=merged(input_ids=ids,use_cache=False,logits_to_keep=1).logits.float().cpu()
    error=float((before-after).norm()/before.norm())
    result={'started':stamp(),'base_model':report['base_model']['model'],'revision':report['base_model']['revision'],
      'adapter':str(adapter),'merged_path':str(target),'merge_relative_logits_l2':error,'memory':memory(torch)}
    if error>.03:save_json(run/'merge-report.json',result);raise RuntimeError(f'Merge error exceeds bound: {error}')
    merged.save_pretrained(target,safe_serialization=True,max_shard_size='4GB');tokenizer.save_pretrained(target)
    source=LOCAL/'models'/f"Qwen3.5-{config['size']}"
    for name in ('preprocessor_config.json','video_preprocessor_config.json','chat_template.jinja'):
        if (source/name).exists():shutil.copy2(source/name,target/name)
    result.update(saved=True,seconds=time.perf_counter()-tick,finished=stamp())
    save_json(run/'merge-report.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
