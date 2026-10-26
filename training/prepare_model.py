"""Download official immutable Transformers artifacts, never Ollama/GGUF weights."""
import argparse
import json
import os
import shutil
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor
from common import LOCAL, configure, save_json, stamp


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--size', choices=['9B','4B'],default='9B')
    p.add_argument('--metadata-only',action='store_true')
    p.add_argument('--source',choices=['huggingface','modelscope'],default='modelscope')
    # huggingface.co is the integrity authority because it publishes LFS SHA256 per
    # revision. When it is unreachable (observed 502/SSL failure), ModelScope's own
    # per-file SHA256 is the only authority left; that is recorded, never silently swapped.
    p.add_argument('--integrity',choices=['huggingface','modelscope'],default='huggingface')
    args=p.parse_args()
    # CAS/Xet stalls on this host; official HTTPS streaming is resumable too.
    os.environ.setdefault('HF_HUB_DISABLE_XET','1')
    os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT','30')
    configure()
    from huggingface_hub import HfApi, snapshot_download
    repo=f'Qwen/Qwen3.5-{args.size}'
    previous_path=LOCAL/f'model-{args.size}.json'
    previous=json.loads(previous_path.read_text(encoding='utf-8')) if previous_path.exists() else {}
    info=HfApi().model_info(repo,revision=previous.get('revision'),files_metadata=True) if args.integrity=='huggingface' else None
    target=LOCAL/'models'/f'Qwen3.5-{args.size}'
    if info is not None:
        files=[{'name':s.rfilename,'bytes':s.size,'sha256':getattr(s.lfs,'sha256',None) if s.lfs else None} for s in info.siblings if s.rfilename.endswith(('.json','.jinja','.safetensors'))]
        revision=info.sha
    else:
        import httpx
        listing=httpx.get(f'https://modelscope.cn/api/v1/models/{repo}/repo/files',params={'Revision':'master','Recursive':'true'},timeout=40)
        listing.raise_for_status()
        entries=[s for s in listing.json()['Data']['Files'] if s['Name'].endswith(('.json','.jinja','.safetensors'))]
        if not entries:raise RuntimeError('ModelScope listing is empty; refusing')
        if not all(s.get('Sha256') for s in entries):raise RuntimeError('ModelScope listing lacks SHA256; refusing')
        # ModelScope reports the commit that last touched each file, not one snapshot.
        # Shards must still come from one revision or the weight set is inconsistent.
        shard_revisions={s['Revision'] for s in entries if s['Name'].endswith('.safetensors')}
        if len(shard_revisions)!=1:raise RuntimeError(f'Weight shards span revisions: {sorted(shard_revisions)}')
        revision=shard_revisions.pop()
        files=[{'name':s['Name'],'bytes':s['Size'],'sha256':s['Sha256'],'revision':s['Revision']} for s in entries]
    manifest={'model':repo,'revision':revision,'files':files,'total_bytes':sum(s['bytes'] or 0 for s in files),'path':str(target),'download_started':stamp(),'download_complete':False}
    mirror_revision=None
    if args.source=='modelscope':
        import httpx
        if args.integrity=='modelscope':
            mirror_revision=revision
            manifest.update(download_source='Qwen official ModelScope mirror',mirror_revision=mirror_revision,
                integrity_authority=f'ModelScope repository SHA256 only; huggingface.co unreachable at {stamp()}')
        else:
            mirror=httpx.get(f'https://modelscope.cn/api/v1/models/{repo}/repo/files',params={'Revision':previous.get('mirror_revision','master'),'Recursive':'true'},timeout=20).json()['Data']['Files']
            mirror_files={s['Name']:s for s in mirror}
            for file in files:
                if file['name'].endswith('.safetensors'):
                    peer=mirror_files[file['name']]
                    assert peer['Sha256']==file['sha256'] and peer['Size']==file['bytes'],'Official mirrors differ; refusing'
                    mirror_revision=peer['Revision']
            manifest.update(download_source='Qwen official ModelScope mirror',mirror_revision=mirror_revision,
                integrity_authority=f'Hugging Face {info.sha} LFS SHA256')
    save_json(LOCAL/f'model-{args.size}.json',manifest)
    print(json.dumps(manifest,ensure_ascii=False),flush=True)
    if args.metadata_only:return
    if shutil.disk_usage(LOCAL).free < manifest['total_bytes']+30*2**30: raise RuntimeError('Less than weights + 30 GiB disk reserve')
    selected=[s['name'] for s in files if not s['name'].endswith('.safetensors')]
    if info is not None:
        snapshot_download(repo,revision=info.sha,local_dir=target,allow_patterns=selected,max_workers=2)
    else:
        import httpx
        target.mkdir(parents=True,exist_ok=True)
        with httpx.Client(follow_redirects=True,timeout=httpx.Timeout(90,connect=20)) as client:
            for file in files:
                if file['name'].endswith('.safetensors'):continue
                destination=target/file['name']
                if destination.exists() and hashlib.file_digest(destination.open('rb'),'sha256').hexdigest()==file['sha256']:continue
                response=client.get(f'https://modelscope.cn/models/{repo}/resolve/{file["revision"]}/{file["name"]}')
                response.raise_for_status()
                if len(response.content)!=file['bytes']:raise RuntimeError(f'Size mismatch {file["name"]}')
                if hashlib.sha256(response.content).hexdigest()!=file['sha256']:raise RuntimeError(f'SHA256 mismatch {file["name"]}')
                destination.write_bytes(response.content)
                print('VERIFIED',file['name'],flush=True)
    # The host's CDN path stalls on unbounded multi-GB HTTP responses. Bounded
    # official HTTPS ranges validate Content-Range and final official LFS SHA256.
    def download_weight(file):
        import httpx
        destination=target/file['name'];partial=destination.with_suffix(destination.suffix+'.part')
        if destination.exists() and destination.stat().st_size==file['bytes']:
            with destination.open('rb') as stream:existing_digest=hashlib.file_digest(stream,'sha256').hexdigest()
            if existing_digest!=file['sha256']:raise RuntimeError('Existing official weight digest mismatch; not overwriting')
            return
        chunk=32*1024*1024
        with httpx.Client(follow_redirects=True,timeout=httpx.Timeout(40,connect=20)) as client:
            url=(f'https://modelscope.cn/models/{repo}/resolve/{file.get("revision") or mirror_revision}/{file["name"]}'
                 if args.source=='modelscope' else f'https://huggingface.co/{repo}/resolve/{info.sha}/{file["name"]}')
            offset=partial.stat().st_size if partial.exists() else 0
            while offset<file['bytes']:
                end=min(offset+chunk,file['bytes'])-1
                for attempt in range(5):
                    try:
                        response=client.get(url,headers={'Range':f'bytes={offset}-{end}'})
                        response.raise_for_status()
                        expected=f'bytes {offset}-{end}/{file["bytes"]}'
                        if response.status_code!=206 or response.headers.get('content-range')!=expected:
                            raise RuntimeError(f'Unexpected range response: {response.status_code}, {response.headers.get("content-range")}')
                        if len(response.content)!=end-offset+1:raise RuntimeError('Incomplete HTTP range')
                        with partial.open('ab') as out:out.write(response.content)
                        offset=end+1
                        if offset//(256*1024*1024)!=(offset-len(response.content))//(256*1024*1024) or offset==file['bytes']:
                            print(json.dumps({'file':file['name'],'bytes':offset,'total':file['bytes'],'time':stamp()}),flush=True)
                        break
                    except Exception as exc:
                        print(json.dumps({'file':file['name'],'offset':offset,'attempt':attempt+1,'error_type':type(exc).__name__}),flush=True)
                        if attempt==4:raise
                        time.sleep(1+attempt)
        with partial.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        if not file['sha256'] or digest!=file['sha256']:raise RuntimeError('Official weight SHA256 mismatch')
        partial.replace(destination)
        print('VERIFIED',file['name'],digest,flush=True)
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(download_weight,[s for s in files if s['name'].endswith('.safetensors')]))
    manifest.update(download_complete=True,download_finished=stamp())
    save_json(LOCAL/f'model-{args.size}.json',manifest)
    print('DOWNLOADED',target,flush=True)


if __name__=='__main__':main()
