"""Portable, loopback-only Ollama import/inference with a separate model store."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import httpx
from common import LOCAL,REPO,configure,save_json,stamp
from build_data import TOOLS


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args()
    configure();run=Path(args.run).resolve()
    files=list((run/'gguf').glob('*-Q4_K_M.gguf'))
    if len(files)!=1:raise RuntimeError('Expected exactly one quantized experiment')
    executable=LOCAL/'tools/ollama-v0.40.11/ollama.exe'
    if not executable.exists():raise RuntimeError('Extract the verified portable archive first')
    output=run/'ollama';output.mkdir(exist_ok=True)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    endpoint=f'http://127.0.0.1:{port}'
    env=os.environ.copy()
    env.update(OLLAMA_HOST=f'127.0.0.1:{port}',OLLAMA_MODELS=str(LOCAL/'ollama-models'),
        OLLAMA_NO_CLOUD='1',OLLAMA_NOHISTORY='1',OLLAMA_NOPRUNE='1',OLLAMA_NUM_PARALLEL='1',
        OLLAMA_MAX_LOADED_MODELS='1',OLLAMA_KEEP_ALIVE='0',OLLAMA_VULKAN='1',GGML_VK_VISIBLE_DEVICES='0')
    name='qwen3.5:9b-qq-lora-exp1'
    modelfile=output/'Modelfile'
    modelfile.write_text('FROM '+json.dumps(files[0].as_posix())+'\nPARAMETER num_ctx 4096\nPARAMETER num_thread 4\nPARAMETER temperature 0.7\nPARAMETER top_p 0.8\nPARAMETER top_k 20\n',encoding='utf-8')
    report={'started':stamp(),'endpoint':endpoint,'model':name,'model_store':env['OLLAMA_MODELS'],'status':'running','results':[]}
    flags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS
    process=None
    try:
        with (output/'server.log').open('w',encoding='utf-8') as log,httpx.Client(timeout=300,trust_env=False) as client:
            process=subprocess.Popen([str(executable),'serve'],env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
            for _ in range(90):
                if process.poll() is not None:raise RuntimeError('Portable Ollama server exited; see server.log')
                try:
                    version=client.get(endpoint+'/api/version').json()
                    if 'version' in version:break
                except (httpx.HTTPError,ValueError):pass
                time.sleep(1)
            else:raise RuntimeError('Portable Ollama startup timed out')
            report['version']=version
            with (output/'create.log').open('w',encoding='utf-8') as log:
                result=subprocess.run([str(executable),'create',name,'-f',str(modelfile)],env=env,stdout=log,stderr=subprocess.STDOUT,creationflags=flags,timeout=300)
            if result.returncode:raise RuntimeError('Ollama import failed; see create.log')
            persona=(REPO/'config/qianwen35_persona.txt').read_text(encoding='utf-8')
            cases=[('chat','事情忙了一整天，现在只想放空。',None),('fact','只算结果：23 × 7 - 18。',None),('tool','只读查一下 Codex 的剩余额度。',TOOLS)]
            for case,prompt,tools in cases:
                payload={'model':name,'messages':[{'role':'system','content':persona},{'role':'user','content':prompt}],
                    'stream':False,'think':False,'keep_alive':0,'options':{'seed':3407,'temperature':0,'num_predict':192,'num_ctx':4096,'num_thread':4}}
                if tools:payload['tools']=tools
                tick=time.perf_counter();response=client.post(endpoint+'/api/chat',json=payload);response.raise_for_status()
                row={'case':case,'seconds':time.perf_counter()-tick,'response':response.json()}
                report['results'].append(row);save_json(output/'verification.json',report)
                print(json.dumps(row,ensure_ascii=False),flush=True)
            report['loaded_after_tests']=client.get(endpoint+'/api/ps').json()
            report['status']='executed_review_required'
    except Exception as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if process is not None and process.poll() is None:
            import psutil
            try:
                owned=psutil.Process(process.pid)
                for child in owned.children(recursive=True):
                    try:child.terminate()
                    except psutil.NoSuchProcess:pass
                process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
            except (psutil.NoSuchProcess,ProcessLookupError):pass
        report['finished']=stamp();save_json(output/'verification.json',report)


if __name__=='__main__':main()
