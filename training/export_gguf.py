"""Pinned converter/quantizer plus actual BF16/Q4 inference, no Ollama replacement."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from common import LOCAL,REPO,configure,save_json,stamp
from build_data import TOOLS


def execute(argv,log,timeout=1800):
    with Path(log).open('w',encoding='utf-8') as output:
        completed=subprocess.run([str(x) for x in argv],stdout=output,stderr=subprocess.STDOUT,
            timeout=timeout,creationflags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS)
    if completed.returncode:raise RuntimeError(f'Command exit {completed.returncode}: {log}')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args()
    torch=configure();run=Path(args.run).resolve();merged=run/'merged-bf16'
    merge_report=json.loads((run/'merge-report.json').read_text(encoding='utf-8'))
    assert merge_report.get('saved'),'Merge not verified'
    if shutil.disk_usage(run).free<30*2**30:raise RuntimeError('Less than 30 GiB disk reserve')
    export=run/'gguf';export.mkdir(exist_ok=True)
    source=LOCAL/'tools/llama.cpp';cpu=LOCAL/'tools/llama-cpu-b11898';vulkan=LOCAL/'tools/llama-vulkan-b11898'
    size=json.loads((run/'summary.json').read_text(encoding='utf-8'))['config']['size'].lower()
    bf16=export/f'qianwen-{size}-exp1-BF16.gguf';quantized=export/f'qianwen-{size}-exp1-Q4_K_M.gguf'
    report={'started':stamp(),'toolchain':json.loads((LOCAL/'tools/llama-toolchain.json').read_text()),'status':'running'}
    save_json(export/'export-report.json',report)
    tick=time.perf_counter()
    if not bf16.exists():execute([sys.executable,'-u',source/'convert_hf_to_gguf.py',merged,'--outfile',bf16,'--outtype','bf16','--no-mtp'],export/'conversion.log')
    report['conversion_seconds']=time.perf_counter()-tick
    # Inspect metadata with the exact same gguf-py revision as the converter.
    sys.path.insert(0,str(source/'gguf-py'))
    import gguf
    reader=gguf.GGUFReader(str(bf16))
    architecture=reader.get_field('general.architecture').contents()
    block_count=int(reader.get_field(architecture+'.block_count').contents())
    assert block_count==32,f'Unexpected block count {block_count}'
    report.update(architecture=architecture,block_count=block_count,bf16_bytes=bf16.stat().st_size)
    del reader
    tick=time.perf_counter()
    if not quantized.exists():execute([cpu/'llama-quantize.exe',bf16,quantized,'Q4_K_M',str(torch.get_num_threads())],export/'quantization.log')
    report.update(quantization_seconds=time.perf_counter()-tick,q4_bytes=quantized.stat().st_size)
    save_json(export/'export-report.json',report)
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(merged,local_files_only=True)
    persona=(REPO/'config/qianwen35_persona.txt').read_text(encoding='utf-8')
    cases=[('greeting','我的工作列表越写越长，有点烦。',None),('fact','只算结果：23 × 7 - 18。',None),
       ('tool','只读查一下 Codex 的剩余额度。',TOOLS)]
    results=[]
    for precision,path,layers in [('BF16',bf16,16),('Q4_K_M',quantized,99)]:
        for name,prompt,tools in cases:
            text=tokenizer.apply_chat_template([{'role':'system','content':persona},{'role':'user','content':prompt}],
                tools=tools,tokenize=False,add_generation_prompt=True,enable_thinking=False)
            prompt_file=export/f'prompt-{name}.txt';prompt_file.write_text(text,encoding='utf-8')
            output=export/f'inference-{precision}-{name}.txt'
            execute([vulkan/'llama-completion.exe','--device','Vulkan0','-m',path,'-f',prompt_file,'-n','160','-c','2048','-ngl',str(layers),
                '-t',str(torch.get_num_threads()),'-tb',str(torch.get_num_threads()),'--temp','0','--seed','3407',
                '--no-display-prompt','--no-conversation','--simple-io'],output,timeout=600)
            results.append({'precision':precision,'case':name,'log':str(output)})
    report.update(status='inference_executed_review_required',inference=results,finished=stamp())
    save_json(export/'export-report.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
