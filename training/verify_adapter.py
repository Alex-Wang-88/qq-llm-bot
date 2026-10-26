"""Fresh-process adapter loading, changed logits and real generation acceptance."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from common import configure,memory,save_json,stamp
from build_data import SYSTEM
from model_utils import load


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--adapter');args=parser.parse_args()
    run=Path(args.run).resolve();report=json.loads((run/'summary.json').read_text(encoding='utf-8'))
    config=report['config'];torch=configure(config['seed']);tick=time.perf_counter()
    assert report.get('optimizer_steps',0)>0 and report.get('adapter_parameter_delta_l2_sum',0)>0
    assert math.isfinite(report['final_validation_loss'])
    adapter=Path(args.adapter).resolve() if args.adapter else run/'adapter'
    model,tokenizer,info=load(torch,config['size'],config['cpu_layers'],adapter=adapter)
    model.eval()
    ids=tokenizer.apply_chat_template([{'role':'system','content':SYSTEM},{'role':'user','content':'晚上好，今天有点累。'}],
        tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors='pt',return_dict=False).to('cuda')
    with torch.no_grad():
        adapted=model(input_ids=ids,use_cache=False,logits_to_keep=1).logits.float()
        with model.disable_adapter():base=model(input_ids=ids,use_cache=False,logits_to_keep=1).logits.float()
        delta=float((adapted-base).norm().cpu()/base.norm().cpu())
        assert delta>1e-5 and torch.isfinite(adapted).all()
        output=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=96,do_sample=False,use_cache=True,pad_token_id=tokenizer.eos_token_id)
    text=tokenizer.decode(output[0,ids.shape[1]:],skip_special_tokens=True)
    assert text.strip()
    with (adapter/'adapter_model.safetensors').open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    proof={'passed':True,'timestamp':stamp(),'fresh_process_load':True,'adapter':str(adapter),'adapter_sha256':digest,
        'adapter_vs_base_relative_logits_l2':delta,'text':text,'seconds':time.perf_counter()-tick,'memory':memory(torch),'loading_info':info}
    save_json(run/'adapter-verification.json',proof)
    expected=config.get('steps') or math.ceil(len(json.loads((run/'training-record-ids.json').read_text()))*config.get('epochs',1)/config['gradient_accumulation'])
    if report['optimizer_steps']==expected:
        if report.get('error'):report.setdefault('recovered_errors',[]).append(report.pop('error'))
        report.setdefault('final_adapter_inference_text',report.get('inference_text'))
        report.update(status='completed',inference_text=text,fresh_adapter_verification=proof,finished=stamp())
        save_json(run/'summary.json',report)
    print(json.dumps(proof,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
