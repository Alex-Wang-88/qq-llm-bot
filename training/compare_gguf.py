"""Compare base vs SFT GGUF through llama-server's OpenAI-compatible API.

Both models are served with the same persona, sampling params and seed, so the
only variable is the merged adapter. Results go to a JSON file for review.

Requires `llama-server.exe` under `.training-local/llama-bin/`; see
DEPLOYMENT.md for how that prebuilt package is obtained (no local build needed).
"""
import argparse
import json
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / '.training-local' / 'llama-bin' / 'llama-server.exe'
PERSONA = (ROOT / 'config' / 'qianwen35_persona.txt').read_text(encoding='utf-8')

# Same prompts as evaluation/comparison-4b-v2/fixtures.json, minus the ones
# that need tool definitions or multi-turn history.
CASES = [
    ('fact_math', '只算结果：23 × 7 - 18。'),
    ('detail_streaming', '请详细解释分段发送和流式输出的区别，并说怎么验证。'),
    ('banter', '我的待办清单又繁殖了，离谱'),
    ('no_image', '我发的那张照片里，窗帘是什么颜色的？'),
]


def wait_ready(port, timeout=180):
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


def chat(port, prompt, seed=42, n=400, temp=0.7):
    body = json.dumps({
        'messages': [
            {'role': 'system', 'content': PERSONA},
            {'role': 'user', 'content': prompt},
        ],
        'temperature': temp,
        'top_p': 0.8,
        'top_k': 20,
        'seed': seed,
        'max_tokens': n,
        'chat_template_kwargs': {'enable_thinking': False},
    }).encode('utf-8')
    req = urllib.request.Request(
        f'http://127.0.0.1:{port}/v1/chat/completions',
        data=body, headers={'Content-Type': 'application/json'})
    started = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        data = json.load(r)
    choice = data['choices'][0]
    usage = data.get('usage', {})
    return {
        'text': choice['message'].get('content') or '',
        'finish_reason': choice.get('finish_reason'),
        'prompt_tokens': usage.get('prompt_tokens'),
        'completion_tokens': usage.get('completion_tokens'),
        'seconds': round(time.time() - started, 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--label', required=True)
    ap.add_argument('--port', type=int, default=8099)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    log = open(ROOT / '.training-local' / f'server-{args.label}.log', 'w', encoding='utf-8')
    proc = subprocess.Popen(
        [str(BIN), '-m', str(Path(args.model).resolve()), '--port', str(args.port),
         '-ngl', '99', '-c', '4096', '--no-warmup', '-a', 'qwen35'],
        stdout=log, stderr=subprocess.STDOUT)
    rows = []
    try:
        if not wait_ready(args.port):
            print(json.dumps({'label': args.label, 'error': 'server did not become ready'}))
            return
        for case_id, prompt in CASES:
            try:
                row = chat(args.port, prompt)
            except Exception as exc:  # noqa: BLE001 - report, do not abort the batch
                row = {'error': f'{type(exc).__name__}: {exc}'}
            row.update(case=case_id, label=args.label, prompt=prompt)
            row['chars'] = len(row.get('text', ''))
            rows.append(row)
            print(f"  {case_id:18s} {row.get('chars', 0):4d}字 "
                  f"{row.get('completion_tokens', '?')}tok {row.get('finish_reason', '')}", flush=True)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()

    Path(args.out).write_text(
        json.dumps({'label': args.label, 'model': args.model, 'rows': rows},
                   ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'saved {args.out}')


if __name__ == '__main__':
    main()
