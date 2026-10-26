"""Local-only experiment utilities. No QQ/AstrBot/SQLite access."""
import ctypes
import datetime as dt
import json
import os
from pathlib import Path
import random
import sys
import time

REPO = Path(__file__).resolve().parents[1]
LOCAL = Path(os.environ.get('TRAINING_ROOT', REPO / '.training-local'))


def json_default(value):
    if isinstance(value,set):return sorted(value)
    if isinstance(value,Path):return str(value)
    raise TypeError(f'Unsupported JSON value: {type(value).__name__}')


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default), encoding='utf-8')
    temporary.replace(path)


def stamp():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat()


def configure(seed=3407):
    for stream in (sys.stdout,sys.stderr):
        if hasattr(stream,'reconfigure'):stream.reconfigure(encoding='utf-8')
    default_workers = min(8,max(1,(os.cpu_count() or 2)-2))
    workers = min(int(os.environ.get('TRAINING_WORKERS', str(default_workers))), 16)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = str(workers)
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ.setdefault('HF_HOME', str(LOCAL / 'hf'))
    os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
    if os.name == 'nt':
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    random.seed(seed)
    import torch
    torch.set_num_threads(workers)
    torch.set_num_interop_threads(min(2, workers))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return torch


def memory(torch=None):
    import psutil
    process_memory = psutil.Process().memory_info()
    result = {'rss_gib': process_memory.rss / 2**30,
              'peak_rss_gib': getattr(process_memory,'peak_wset',process_memory.rss) / 2**30,
              'available_ram_gib': psutil.virtual_memory().available / 2**30,
              'timestamp': stamp()}
    if torch is not None and torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        result.update(gpu_free_gib=free / 2**30, gpu_total_gib=total / 2**30,
                      gpu_allocated_gib=torch.cuda.memory_allocated() / 2**30,
                      gpu_peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                      gpu_reserved_gib=torch.cuda.memory_reserved() / 2**30,
                      gpu_peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
    return result


def append_jsonl(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False) + '\n')
