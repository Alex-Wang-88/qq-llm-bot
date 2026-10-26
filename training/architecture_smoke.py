"""Random tiny Qwen3.5 structural test. This is NOT pretrained-model training."""
import copy
import json
import time
from types import SimpleNamespace
from common import LOCAL, configure, save_json


def main():
    torch=configure()
    from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
    from peft import LoraConfig, get_peft_model
    config=Qwen3_5TextConfig(vocab_size=256,hidden_size=128,intermediate_size=256,num_hidden_layers=4,
        num_attention_heads=2,num_key_value_heads=1,head_dim=64,linear_num_key_heads=2,linear_num_value_heads=2,
        linear_key_head_dim=32,linear_value_head_dim=32,layer_types=['linear_attention']*3+['full_attention'],
        max_position_embeddings=512,partial_rotary_factor=.5,tie_word_embeddings=True)
    config._attn_implementation='eager'
    model=Qwen3_5ForCausalLM(config).to(device='cuda',dtype=torch.bfloat16)
    model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,target_modules=['q_proj','k_proj','v_proj','o_proj','in_proj_qkv','in_proj_z','out_proj','gate_proj','up_proj','down_proj'],task_type='CAUSAL_LM'))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    model.enable_input_require_grads()
    model.config.use_cache=False
    reference=copy.deepcopy(model).float().cpu()
    x=torch.randint(0,256,(1,64),device='cuda')
    with torch.no_grad():
        cpu_logits=reference(x.cpu(),use_cache=False).logits
        gpu_logits=model(x,use_cache=False).logits.float().cpu()
        error=float((cpu_logits-gpu_logits).norm()/cpu_logits.norm())
        assert error<.08,error
    before={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.005,foreach=False,fused=False)
    losses=[]
    started=time.perf_counter()
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        loss=model(x,labels=x,use_cache=False).loss
        assert torch.isfinite(loss)
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    deltas={n:float((p.detach()-before[n]).norm().cpu()) for n,p in model.named_parameters() if p.requires_grad}
    assert sum(deltas.values())>0 and losses[-1]<losses[0]
    report={'random_tiny_model_only':True,'pretrained_training':False,'passed':True,'forward_relative_l2':error,'losses':losses,'adapter_delta_l2_sum':sum(deltas.values()),'seconds':time.perf_counter()-started}
    # Exercise precisely the same static-sharding hooks used by the real model.
    from model_utils import place
    raw=model.base_model.model
    model.eval()
    with torch.no_grad():gpu_reference=model(x,use_cache=False).logits.float().cpu()
    wrapper=SimpleNamespace(model=SimpleNamespace(language_model=raw.model,visual=torch.nn.Identity()),lm_head=raw.lm_head)
    place(torch,wrapper,2)
    with torch.no_grad():mixed=model(x,use_cache=False).logits.float().cpu()
    offload_error=float((mixed-gpu_reference).norm()/gpu_reference.norm())
    assert offload_error<.03,offload_error
    model.train()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,foreach=False,fused=False)
    device_deltas={}
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True);loss=model(x,labels=x,use_cache=False).loss;loss.backward()
        snapshot={n:p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        optimizer.step()
        for n,p in model.named_parameters():
            if p.requires_grad:device_deltas[str(p.device)]=device_deltas.get(str(p.device),0)+float((p.detach()-snapshot[n]).norm().cpu())
    assert device_deltas.get('cpu',0)>0 and device_deltas.get('cuda:0',0)>0,device_deltas
    report['static_cpu_offload']={'passed':True,'forward_relative_l2':offload_error,'adapter_update_l2_by_device':device_deltas,'optimizer_steps':3}
    save_json(LOCAL/'architecture-smoke.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
