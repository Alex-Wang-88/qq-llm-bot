"""Plain Transformers/PEFT; explicit static placement that preserves autograd."""
import json
from common import LOCAL

TARGETS=['q_proj','k_proj','v_proj','o_proj','in_proj_qkv','in_proj_z','out_proj','gate_proj','up_proj','down_proj']


def move_tree(torch, value, device):
    if isinstance(value,torch.Tensor):return value.to(device)
    if isinstance(value,tuple):return tuple(move_tree(torch,v,device) for v in value)
    if isinstance(value,list):return [move_tree(torch,v,device) for v in value]
    if isinstance(value,dict):return {k:move_tree(torch,v,device) for k,v in value.items()}
    return value


def place(torch,model,cpu_layers=0):
    # No accelerate device_map='auto' inference/offload hooks: copies remain
    # visible to autograd, including during checkpoint recomputation.
    language=model.model.language_model
    handles=[]
    def assign(module,device):
        module.to(device)
        def pre_hook(mod,args,kwargs):
            return move_tree(torch,args,device),move_tree(torch,kwargs,device)
        handles.append(module.register_forward_pre_hook(pre_hook,with_kwargs=True))
    assign(language.embed_tokens,'cuda')
    language.rotary_emb.to('cuda')
    for i,layer in enumerate(language.layers):assign(layer,'cpu' if i<cpu_layers else 'cuda')
    assign(language.norm,'cuda')
    assign(model.lm_head,'cuda')
    # Visual encoder stays frozen on CPU and is never run for text-only data.
    model.model.visual.to('cpu')
    return handles


def load(torch,size='4B',cpu_layers=0,rank=8,alpha=16,adapter=None):
    from transformers import Qwen3_5ForConditionalGeneration,AutoTokenizer
    from peft import LoraConfig,get_peft_model,PeftModel
    path=LOCAL/'models'/f'Qwen3.5-{size}'
    model,info=Qwen3_5ForConditionalGeneration.from_pretrained(path,dtype=torch.bfloat16,
        attn_implementation='eager',local_files_only=True,output_loading_info=True)
    info={k:sorted(v) if isinstance(v,set) else v for k,v in info.items()}
    if info.get('missing_keys') or info.get('mismatched_keys') or info.get('error_msgs'):
        raise RuntimeError(f'Weight loading failed: {info}')
    # MTP keys, if present, are not used by the Transformers architecture.
    unexpected=[k for k in info.get('unexpected_keys',[]) if not k.startswith('mtp.')]
    if unexpected:raise RuntimeError(f'Unexpected checkpoint keys: {unexpected[:10]}')
    model.requires_grad_(False)
    if adapter:
        model=PeftModel.from_pretrained(model,adapter,is_trainable=False)
        raw=model.base_model.model
    else:
        # Regex excludes all visual encoder linears even if they share names.
        pattern=r'model\.language_model\.layers\.\d+\.(?:self_attn|linear_attn|mlp)\.(?:'+('|'.join(TARGETS))+')'
        model=get_peft_model(model,LoraConfig(r=rank,lora_alpha=alpha,lora_dropout=0.0,
            target_modules=pattern,bias='none',task_type='CAUSAL_LM'))
        raw=model.base_model.model
    place(torch,raw,cpu_layers)
    for name,p in model.named_parameters():
        if p.requires_grad:
            assert 'lora_' in name and 'visual' not in name
            p.data=p.data.float()
    model.config.use_cache=False
    tokenizer=AutoTokenizer.from_pretrained(path,local_files_only=True)
    return model,tokenizer,info


def placement_summary(model):
    result={}
    for p in model.parameters():
        key=f'{p.device}/{p.dtype}/'+('trainable' if p.requires_grad else 'frozen')
        result[key]=result.get(key,0)+p.numel()
    return result
