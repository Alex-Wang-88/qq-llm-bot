"""Switch a verified local profile; never deletes weights or conversations."""
import argparse
import json
from pathlib import Path
import sqlite3
import urllib.request

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('profile',choices=['qwen3','qwen35','qwen35_4b']);args=parser.parse_args()
    profile=json.loads((ROOT/'config/model_profiles.json').read_text(encoding='utf-8'))[args.profile]
    request=urllib.request.Request('http://127.0.0.1:11434/api/show',data=json.dumps({'model':profile['model']}).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=10) as response:metadata=json.load(response)
    if metadata['details']['quantization_level']!='Q4_K_M':raise SystemExit('Model quantization does not match Q4_K_M')
    if args.profile=='qwen35' and 'vision' not in metadata.get('capabilities',[]):raise SystemExit('New profile lacks vision capability')
    path=ROOT/'data/cmd_config.json';cfg=json.loads(path.read_text(encoding='utf-8-sig'))
    for provider in cfg['provider']:
        if provider['id']=='ollama_local':
            provider.update(model=profile['model'],modalities=profile['modalities'],max_context_tokens=profile['context_tokens'],custom_extra_body=profile['generation'])
    cfg['agent_runner']['config']['compression']['fallback_max_tokens']=profile['context_tokens']
    if args.profile=='qwen35':
        cfg['provider_settings']['image_compress_options']['max_size']=768
        cfg['provider_settings']['max_quoted_fallback_images']=2
    temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(cfg,ensure_ascii=False,indent=4),encoding='utf-8');temporary.replace(path)
    text=(ROOT/profile['persona_file']).read_text(encoding='utf-8').strip()
    with sqlite3.connect(str(ROOT/'data/data_v4.db')) as db:
        db.execute('UPDATE personas SET system_prompt=?,skills=?,tools=?,updated_at=CURRENT_TIMESTAMP WHERE persona_id=?',(text,'[]',json.dumps(['web_search','fetch_web_page','harness_query']),'千问娘'))
    (ROOT/'data/harness/model-profile.json').write_text(json.dumps({'active':args.profile,'model':profile['model']},indent=2),encoding='utf-8')
    print(f"Selected {args.profile}: {profile['model']}; restart AstrBot to load it. Old weights retained.")


if __name__=='__main__':main()
