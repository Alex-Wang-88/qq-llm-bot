"""Isolate whether the arithmetic miss comes from quantization or from the model.

Runs the same prompt under greedy decoding and five sampled seeds, for one GGUF.
Compare an f16 build against a Q4_K_M build of the same weights: if both miss at
a similar rate, the loss is the model's sampling variance, not quantization.

Requires `llama-server.exe` under `.training-local/llama-bin/`.
"""
import argparse
import json
import re
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / '.training-local' / 'llama-bin' / 'llama-server.exe'
PERSONA = (ROOT / 'config' / 'qianwen35_persona.txt').read_text(encoding='utf-8')
PROMPT = '只算结果：23 × 7 - 18。'
EXPECTED = '143'


def wait_ready(port, timeout=240):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2) as r:
                if r.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(0.5)
    return False


def ask(port, temp, seed):
    body = json.dumps({
        'messages': [
            {'role': 'system', 'content': PERSONA},
            {'role': 'user', 'content': PROMPT},
        ],
        'temperature': temp, 'top_p': 0.8, 'top_k': 20, 'seed': seed, 'max_tokens': 120,
        'chat_template_kwargs': {'enable_thinking': False},
    }).encode('utf-8')
    req = urllib.request.Request(f'http://127.0.0.1:{port}/v1/chat/completions',
                                 data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.load(r)
    return (data['choices'][0]['message'].get('content') or '').strip()


def numbers(text):
    return re.findall(r'-?\d+', text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--port', type=int, default=8097)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    log = open(ROOT / '.training-local' / f'server-{args.label}-math.log', 'w', encoding='utf-8')
    t0 = time.time()
    proc = subprocess.Popen(
        [str(BIN), '-m', str(Path(args.model).resolve()), '--port', str(args.port),
         '-ngl', '99', '-c', '2048', '--no-warmup', '-a', 'qwen35'],
        stdout=log, stderr=subprocess.STDOUT)
    runs = []
    try:
        if not wait_ready(args.port):
            print(json.dumps({'label': args.label, 'error': 'server not ready'}))
            return
        load_seconds = round(time.time() - t0, 1)
        for temp, seed in [(0.0, 42)] + [(0.7, s) for s in (1, 2, 3, 4, 5)]:
            try:
                text = ask(args.port, temp, seed)
            except Exception as exc:  # noqa: BLE001
                text = f'<error {type(exc).__name__}>'
            hit = EXPECTED in numbers(text)
            runs.append({'temperature': temp, 'seed': seed, 'text': text, 'correct': hit})
            print(f"  temp={temp} seed={seed}: {'OK ' if hit else 'MISS'} {text[:40]!r}", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()

    payload = {'label': args.label, 'model': args.model, 'expected': EXPECTED,
               'load_seconds': load_seconds, 'runs': runs,
               'correct': sum(r['correct'] for r in runs), 'total': len(runs)}
    Path(args.out).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"correct {payload['correct']}/{payload['total']}  load {load_seconds}s -> {args.out}")


if __name__ == '__main__':
    main()
