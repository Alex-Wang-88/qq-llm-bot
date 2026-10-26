"""Bounded SFT with finite checks, independent validation and resumable checkpoints."""
import argparse
import json
import math
from pathlib import Path
import random
import time
import traceback
from common import LOCAL, append_jsonl, configure, memory, save_json, stamp
from data_utils import batch,encode,read_rows
from model_utils import load,placement_summary


def validation(torch,model,rows):
    model.eval();total=0.;tokens=0
    with torch.no_grad():
        for row in rows:
            n=sum(t!=-100 for t in row['labels'][1:])
            loss=model(**batch(torch,row),use_cache=False).loss
            if not torch.isfinite(loss):raise RuntimeError('Non-finite validation loss')
            total+=float(loss.cpu())*n;tokens+=n
    model.train()
    return total/tokens


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--resume');p.add_argument('--data',default='experimental-v1');args=p.parse_args()
    config=json.loads(Path(args.config).read_text(encoding='utf-8'))
    torch=configure(config['seed'])
    if config.get('gpu_memory_fraction'):
        torch.cuda.set_per_process_memory_fraction(config['gpu_memory_fraction'])
    smoke=json.loads((LOCAL/'gpu-smoke.json').read_text(encoding='utf-8'))
    assert smoke['passed'] and any(t['dtype']=='torch.bfloat16' and t['passed'] for t in smoke['tests'])
    output=LOCAL/'runs'/config['run_name'];output.mkdir(parents=True,exist_ok=True)
    if (output/'summary.json').exists() and not args.resume:raise RuntimeError('Run already exists; choose a new run_name or --resume')
    started=time.perf_counter()
    report={'started':stamp(),'config':config,'status':'running','optimizer_steps':0,'history':[]}
    if args.resume:
        previous_report=json.loads((output/'summary.json').read_text(encoding='utf-8')) if (output/'summary.json').exists() else {}
        report['original_started']=previous_report.get('original_started',previous_report.get('started',report['started']))
        report['resumed_from']=str(Path(args.resume).resolve())
        if (output/'metrics.jsonl').exists():report['history']=read_rows(output/'metrics.jsonl')
    save_json(output/'summary.json',report);save_json(output/'config.json',config)
    try:
        print('STAGE loading official pretrained weights',flush=True)
        model,tokenizer,loading_info=load(torch,config['size'],config['cpu_layers'],config['rank'],config['alpha'])
        report['loading_info']=loading_info
        report['placement']=placement_summary(model)
        report['memory_after_load']=memory(torch)
        print(json.dumps({'stage':'loaded','placement':report['placement'],'memory':report['memory_after_load']}),flush=True)
        manifest=json.loads((LOCAL/f"model-{config['size']}.json").read_text(encoding='utf-8'))
        report['base_model']=manifest
        save_json(output/'base-model.json',manifest)
        data=LOCAL/'data'/args.data
        data_manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
        save_json(output/'data-manifest.json',data_manifest)
        train_raw=read_rows(data/'train.jsonl');valid_raw=read_rows(data/'validation.jsonl')
        if config.get('overfit_records'):train_raw=train_raw[:config['overfit_records']]
        elif 'epochs' not in config:train_raw=[r for r in train_raw if not r.get('tools')][:4]
        train=[encode(tokenizer,r,config['max_length']) for r in train_raw]
        valid=[encode(tokenizer,r,config['max_length']) for r in valid_raw]
        save_json(output/'training-record-ids.json',[r['id'] for r in train_raw])
        if config.get('cpu_reference'):
            print('STAGE pretrained mixed-placement / CPU BF16 numerical reference',flush=True)
            # Numerical reference on the pretrained model, not merely a tiny MLP.
            # CPU FP32 all-weight reference would exceed the current memory reserve;
            # compare all-CPU BF16 against the exact same static mixed placement.
            raw=model.base_model.model
            probe=torch.tensor([tokenizer.encode('数值核对：三加四等于几？')],device='cuda')
            model.eval()
            with torch.no_grad():mixed=model(input_ids=probe,use_cache=False,logits_to_keep=1).logits.float().cpu()
            handles=[]
            # Remove only our own hooks, then place all modules on CPU.
            for module in raw.modules():
                for key,fn in list(module._forward_pre_hooks.items()):
                    if getattr(fn,'__name__','')=='pre_hook':
                        del module._forward_pre_hooks[key]
                        module._forward_pre_hooks_with_kwargs.pop(key,None)
            raw.to('cpu');torch.cuda.empty_cache()
            with torch.no_grad():reference=model(input_ids=probe.cpu(),use_cache=False,logits_to_keep=1).logits.float().cpu()
            error=float((mixed-reference).norm()/reference.norm())
            report['pretrained_cpu_bf16_reference_relative_l2']=error
            print(json.dumps({'pretrained_cpu_bf16_reference_relative_l2':error}),flush=True)
            if error>.12:raise RuntimeError(f'Pretrained CPU/GPU logits mismatch: {error}')
            from model_utils import place
            place(torch,raw,config['cpu_layers'])
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        model.enable_input_require_grads();model.config.use_cache=False
        parameters=[p for p in model.parameters() if p.requires_grad]
        assert all(p.dtype==torch.float32 for p in parameters)
        optimizer=torch.optim.AdamW(parameters,lr=config['learning_rate'],eps=1e-8,foreach=False,fused=False)
        before={n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
        rng=random.Random(config['seed']);order=list(range(len(train)));rng.shuffle(order)
        accumulation=config['gradient_accumulation']
        total_steps=config.get('steps') or math.ceil(len(train)*config.get('epochs',1)/accumulation)
        start_step=0;cursor=0
        if args.resume:
            saved_config=json.loads((Path(args.resume)/'run-config.json').read_text(encoding='utf-8'))
            report['resume_original_config']=saved_config
            for name in ('size','rank','alpha','max_length','seed','gradient_accumulation','learning_rate'):
                if saved_config[name]!=config[name]:raise RuntimeError(f'Resume config mismatch: {name}')
            state=torch.load(Path(args.resume)/'training-state.pt',map_location='cpu',weights_only=False)
            from peft import set_peft_model_state_dict
            from safetensors.torch import load_file
            set_peft_model_state_dict(model,load_file(str(Path(args.resume)/'adapter_model.safetensors')))
            optimizer.load_state_dict(state['optimizer']);start_step=state['step'];cursor=state['cursor'];order=state['order'];rng.setstate(state['random_state'])
            torch.set_rng_state(state['torch_rng']);torch.cuda.set_rng_state_all(state['cuda_rng'])
            discarded=[r for r in report['history'] if r['step']>start_step]
            if discarded:save_json(output/f'recovery-discarded-after-{start_step}.json',discarded)
            report['history']=[r for r in report['history'] if r['step']<=start_step]
            (output/'metrics.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in report['history']),encoding='utf-8')
            report['optimizer_steps']=start_step
        print('STAGE initial held-out validation',flush=True)
        report['initial_validation_loss']=validation(torch,model,valid)
        if args.resume:
            report['validation_loss_at_resume']=report['initial_validation_loss']
            if 'initial_validation_loss' in state:report['initial_validation_loss']=state['initial_validation_loss']
        print(json.dumps({'initial_validation_loss':report['initial_validation_loss']}),flush=True)
        if config.get('overfit_records'):report['initial_overfit_loss']=validation(torch,model,train)
        def checkpoint(step):
            dest=output/f'checkpoint-{step:04}'
            model.save_pretrained(dest,safe_serialization=True);tokenizer.save_pretrained(dest)
            torch.save({'optimizer':optimizer.state_dict(),'step':step,'cursor':cursor,'order':order,
                'random_state':rng.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all(),
                'initial_validation_loss':report['initial_validation_loss']},dest/'training-state.pt')
            save_json(dest/'run-config.json',config)
            return str(dest)
        for step in range(start_step,total_steps):
            if config.get('empty_cache_each_step'):torch.cuda.empty_cache()
            tick=time.perf_counter();optimizer.zero_grad(set_to_none=True);loss_sum=0.;used=0
            # One epoch means each record exactly once, including final partial batch.
            count=min(accumulation,len(order)-cursor) if config.get('epochs')==1 else accumulation
            for _ in range(count):
                if cursor==len(order):cursor=0;rng.shuffle(order)
                row=train[order[cursor]];cursor+=1
                loss=model(**batch(torch,row),use_cache=False).loss
                if not torch.isfinite(loss):raise RuntimeError('Non-finite train loss')
                (loss/count).backward();loss_sum+=float(loss.detach().cpu());used+=1
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):raise RuntimeError('Missing or non-finite gradient')
            grad_norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.0).detach().cpu())
            if not math.isfinite(grad_norm) or grad_norm==0:raise RuntimeError('No finite nonzero gradient')
            optimizer.step()
            if not all(torch.isfinite(p).all() for p in parameters):raise RuntimeError('Non-finite adapter parameter')
            torch.cuda.synchronize()
            elapsed=time.perf_counter()-tick
            row={'step':step+1,'train_loss':loss_sum/used,'gradient_norm':grad_norm,'step_seconds':elapsed,'memory':memory(torch),'time':stamp(),
                'cpu_layers':config['cpu_layers'],'cpu_workers':torch.get_num_threads(),'gpu_memory_fraction':config.get('gpu_memory_fraction')}
            if (step+1)%config['validation_every']==0 or step+1==total_steps:row['validation_loss']=validation(torch,model,valid)
            if (step+1)%config['checkpoint_every']==0 or step+1==total_steps:report['latest_checkpoint']=checkpoint(step+1)
            report['optimizer_steps']=step+1;report['history'].append(row)
            append_jsonl(output/'metrics.jsonl',row);save_json(output/'summary.json',report)
            print(json.dumps(row,ensure_ascii=False),flush=True)
            if row['memory']['available_ram_gib']<6:raise RuntimeError('Stopped at RAM reserve <6 GiB')
            if elapsed>config['max_step_seconds']:raise RuntimeError(f'Step exceeded useful-speed budget: {elapsed:.1f}s')
        deltas={n:float((p.detach().cpu()-before[n]).norm()) for n,p in model.named_parameters() if p.requires_grad}
        assert sum(deltas.values())>0
        report['adapter_parameter_delta_l2_sum']=sum(deltas.values())
        report['updated_adapter_tensors']=sum(v>0 for v in deltas.values())
        report['final_validation_loss']=validation(torch,model,valid)
        if config.get('overfit_records'):report['final_overfit_loss']=validation(torch,model,train)
        model.save_pretrained(output/'adapter',safe_serialization=True);tokenizer.save_pretrained(output/'adapter')
        # A generated result is required in addition to parameter updates.
        model.eval()
        print('STAGE final saved adapter / inference check',flush=True)
        messages=[{'role':'system','content':train_raw[0]['messages'][0]['content']},{'role':'user','content':'晚上好，今天有点累。'}]
        ids=tokenizer.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,enable_thinking=False,return_tensors='pt',return_dict=False).to('cuda')
        with torch.no_grad():generated=model.generate(input_ids=ids,attention_mask=torch.ones_like(ids),max_new_tokens=96,do_sample=False,use_cache=True,pad_token_id=tokenizer.eos_token_id)
        report['inference_text']=tokenizer.decode(generated[0,ids.shape[1]:],skip_special_tokens=True)
        if not report['inference_text'].strip():raise RuntimeError('Empty inference output')
        report['status']='completed'
    except Exception:
        report['status']='failed';report['error']=traceback.format_exc();print(report['error'],flush=True)
    report.update(finished=stamp(),total_seconds=time.perf_counter()-started)
    save_json(output/'summary.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('history','base_model')},ensure_ascii=False,indent=2),flush=True)
    if report['status']!='completed':raise SystemExit(1)


if __name__=='__main__':main()
