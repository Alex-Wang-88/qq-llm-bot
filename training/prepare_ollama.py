"""Download verified portable Ollama; extraction is a separate post-training step."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import httpx
from common import LOCAL,save_json,stamp


def main():
    p=argparse.ArgumentParser();p.add_argument('--extract',action='store_true');args=p.parse_args()
    manifest_path=LOCAL/'sources/ollama-release.json'
    if manifest_path.exists():metadata=json.loads(manifest_path.read_text(encoding='utf-8-sig'))
    else:
        release=httpx.get('https://api.github.com/repos/ollama/ollama/releases/tags/v0.40.11',timeout=20).json()
        metadata={'tag':release['tag_name'],'assets':[{k:a[k] for k in ('name','size','digest','browser_download_url')} for a in release['assets']]}
        save_json(manifest_path,metadata)
    asset=next(a for a in metadata['assets'] if a['name']=='ollama-windows-amd64.zip')
    target=LOCAL/'tools'/asset['name'];partial=target.with_suffix('.zip.part')
    if not (target.exists() and target.stat().st_size==asset['size']):
        offset=partial.stat().st_size if partial.exists() else 0
        with httpx.Client(follow_redirects=True,timeout=40) as client:
            while offset<asset['size']:
                end=min(offset+8*1024*1024,asset['size'])-1
                for attempt in range(4):
                    try:
                        r=client.get(asset['browser_download_url'],headers={'Range':f'bytes={offset}-{end}'})
                        r.raise_for_status()
                        assert r.status_code==206 and r.headers.get('content-range')==f'bytes {offset}-{end}/{asset["size"]}'
                        assert len(r.content)==end-offset+1
                        with partial.open('ab') as stream:stream.write(r.content)
                        offset=end+1
                        if offset%(128*1024*1024)==0:print(json.dumps({'downloaded':offset,'total':asset['size']}),flush=True)
                        break
                    except Exception:
                        if attempt==3:raise
        partial.replace(target)
    with target.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    assert 'sha256:'+digest==asset['digest'],'Official Ollama digest mismatch'
    result={'tag':metadata['tag'],'verified_sha256':digest,'archive':str(target),'timestamp':stamp(),'extracted':False}
    if args.extract:
        base=(LOCAL/'tools/ollama-v0.40.11').resolve()
        with zipfile.ZipFile(target) as archive:
            for item in archive.infolist():
                destination=(base/item.filename).resolve()
                if not destination.is_relative_to(base):raise RuntimeError('Unsafe archive path')
                if item.is_dir():continue
                destination.parent.mkdir(parents=True,exist_ok=True)
                with archive.open(item) as source,destination.open('wb') as out:
                    import shutil
                    shutil.copyfileobj(source,out)
        result.update(extracted=True,path=str(base))
    save_json(LOCAL/'tools/ollama-toolchain.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
