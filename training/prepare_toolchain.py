"""Fetch pinned official llama.cpp releases with published SHA256 verification."""
import hashlib
import json
from pathlib import Path
import zipfile
import httpx
from common import LOCAL,save_json,stamp


def main():
    metadata_path=LOCAL/'sources/llama-release-b11898.json'
    if metadata_path.exists():metadata=json.loads(metadata_path.read_text(encoding='utf-8-sig'))
    else:
        release=httpx.get('https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b11898',timeout=20).json()
        assert release['target_commitish']=='8b28c2394324465bf7727089e7eaf7aec9442d6a'
        metadata={'tag':'b11898','commit':release['target_commitish'],'assets':[{k:a[k] for k in ('name','size','digest','browser_download_url')} for a in release['assets'] if a['name'] in ('llama-b11898-bin-win-cpu-x64.zip','llama-b11898-bin-win-vulkan-x64.zip')]}
        save_json(metadata_path,metadata)
    source_archive=LOCAL/'tools/llama-source-b11898.zip'
    if not source_archive.exists():
        source_archive.parent.mkdir(parents=True,exist_ok=True)
        url='https://github.com/ggml-org/llama.cpp/archive/8b28c2394324465bf7727089e7eaf7aec9442d6a.zip'
        with httpx.stream('GET',url,follow_redirects=True,timeout=40) as response,source_archive.open('wb') as out:
            response.raise_for_status()
            for chunk in response.iter_bytes():out.write(chunk)
    source=(LOCAL/'tools/llama.cpp').resolve()
    with zipfile.ZipFile(source_archive) as archive:
        for item in archive.infolist():
            relative=item.filename.partition('/')[2]
            if not (relative.startswith(('conversion/','gguf-py/','requirements/')) or relative in ('convert_hf_to_gguf.py','LICENSE','requirements.txt')):continue
            destination=(source/relative).resolve()
            if not destination.is_relative_to(source):raise RuntimeError('Unsafe source path')
            if item.is_dir():continue
            destination.parent.mkdir(parents=True,exist_ok=True)
            destination.write_bytes(archive.read(item))
    for asset in metadata['assets']:
        target=LOCAL/'tools'/asset['name'];part=target.with_suffix('.zip.part')
        if not (target.exists() and target.stat().st_size==asset['size']):
            offset=part.stat().st_size if part.exists() else 0
            with httpx.Client(follow_redirects=True,timeout=30) as client:
                while offset<asset['size']:
                    end=min(offset+4*1024*1024,asset['size'])-1
                    r=client.get(asset['browser_download_url'],headers={'Range':f'bytes={offset}-{end}'})
                    r.raise_for_status()
                    assert r.status_code==206 and r.headers.get('content-range')==f'bytes {offset}-{end}/{asset["size"]}'
                    assert len(r.content)==end-offset+1
                    with part.open('ab') as stream:stream.write(r.content)
                    offset=end+1
            part.replace(target)
        with target.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
        assert 'sha256:'+digest==asset['digest'],'Official binary digest mismatch'
        folder=LOCAL/'tools'/('llama-vulkan-b11898' if 'vulkan' in asset['name'] else 'llama-cpu-b11898')
        folder=folder.resolve()
        with zipfile.ZipFile(target) as archive:
            for item in archive.infolist():
                destination=(folder/item.filename).resolve()
                if not destination.is_relative_to(folder):raise RuntimeError('Unsafe archive path')
                if item.is_dir():continue
                destination.parent.mkdir(parents=True,exist_ok=True)
                destination.write_bytes(archive.read(item))
        print(json.dumps({'verified':asset['name'],'sha256':digest,'path':str(folder)}),flush=True)
    save_json(LOCAL/'tools/llama-toolchain.json',{'tag':'b11898','commit':'8b28c2394324465bf7727089e7eaf7aec9442d6a','verified_at':stamp(),'assets':metadata['assets']})


if __name__=='__main__':main()
