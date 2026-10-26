"""Select by validation loss before looking at the reserved test generations."""
import argparse
import json
from pathlib import Path
import shutil
from common import save_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);args=p.parse_args()
    run=Path(args.run).resolve();report=json.loads((run/'summary.json').read_text(encoding='utf-8'))
    assert report['status']=='completed','Training incomplete'
    candidates=[r for r in report['history'] if 'validation_loss' in r and (run/f"checkpoint-{r['step']:04}/adapter_model.safetensors").exists()]
    best=min(candidates,key=lambda r:r['validation_loss'])
    source=run/f"checkpoint-{best['step']:04}";target=run/'selected-adapter'
    if target.exists():raise RuntimeError('Selection already exists; refusing silent replacement')
    target.mkdir()
    for file in source.iterdir():
        if file.is_file() and file.name!='training-state.pt':shutil.copy2(file,target/file.name)
    selection={'criterion':'minimum validation loss among saved checkpoints; test not consulted',
       'selected_step':best['step'],'validation_loss':best['validation_loss'],'source':str(source),'adapter':str(target),
       'candidates':[{'step':r['step'],'validation_loss':r['validation_loss']} for r in candidates]}
    save_json(run/'adapter-selection.json',selection)
    print(json.dumps(selection,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
