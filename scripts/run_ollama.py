"""Start one existing Ollama binary; sanitize its local diagnostic stream."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from assistant_security import redact_text


if __name__=='__main__':
    executable=shutil.which('ollama')
    if not executable:raise SystemExit('Ollama is not installed')
    flags=subprocess.CREATE_NO_WINDOW|subprocess.BELOW_NORMAL_PRIORITY_CLASS if os.name=='nt' else 0
    proc=subprocess.Popen([executable,'serve'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,creationflags=flags)
    path=ROOT/'data/logs/ollama-runtime.log'
    with path.open('a',encoding='utf-8') as log:
        for line in proc.stdout:
            log.write(redact_text(line.decode('utf-8',errors='replace')));log.flush()
    raise SystemExit(proc.wait())
