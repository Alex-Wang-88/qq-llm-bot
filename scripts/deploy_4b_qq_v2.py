"""把 4B adapter 导出的 GGUF 部署成本机 Ollama 模型并验收。

用法：
    python scripts/deploy_4b_qq_v2.py --gguf <path-to-Q4_K_M.gguf> [--dry-run]
    python scripts/deploy_4b_qq_v2.py --check --gguf <path>     # 只校验文件，不建模型

设计约束（和 training/verify_ollama.py 的区别）：
- 那个脚本用的是训练机上的便携版 Ollama + 独立模型库，目的是不污染环境；
  这里部署的是本机常驻 Ollama（127.0.0.1:11434），因为 AstrBot 走的就是它。
- 本机显存只有 4 GB，所以默认 num_ctx 4096、num_thread 6，
  宁可让 Ollama 少卸载层到 GPU，也不要为了 8192 上下文把权重挤到 CPU 上。
- 不写 SYSTEM：人设由 AstrBot 注入，避免两个事实来源。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OLLAMA = os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_NAMES = ("qwen3.5:4b-qq-v2",)
EXPECTED_GIB = 2.52          # 训练机实测 5.13 BPW
GIB_TOLERANCE = 0.35

# 只读工具，形状与 AstrBot 下发的 OpenAI function schema 一致。
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "harness_query",
            "description": "只读查询 Codex 的剩余额度与会话列表。",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["quota", "sessions"]},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "只读联网搜索。",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]


def find_gguf(explicit: str | None) -> Path | None:
    if explicit:
        return Path(explicit).expanduser()
    candidates = []
    for pattern in (".training-local/gguf/*-Q4_K_M.gguf", "*.gguf", "models/*.gguf"):
        candidates.extend(sorted(REPO.glob(pattern)))
    for path in candidates:
        if path.is_file() and path.stat().st_size > 200 * 1024 * 1024:
            return path
    return None


def sha256_of(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def inspect_gguf(path: Path) -> dict:
    size = path.stat().st_size
    with path.open("rb") as handle:
        magic = handle.read(4)
    return {
        "path": str(path),
        "size_bytes": size,
        "size_gib": round(size / 1024 ** 3, 3),
        "magic": magic.decode("ascii", "replace"),
        "magic_ok": magic == b"GGUF",
        "sha256": sha256_of(path),
    }


def http_json(url: str, payload: dict | None = None, timeout: float = 300.0) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="GET" if payload is None else "POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def write_modelfile(dest: Path, gguf: Path, num_ctx: int, num_thread: int) -> Path:
    # 与 GGUF 同目录时用相对路径：Ollama 在 Windows 上解析绝对路径（含盘符冒号）
    # 偶发失败，同目录 + ./文件名 是仓库里既有 Modelfile 验证过的写法。
    from_line = f"FROM ./{gguf.name}" if dest.parent == gguf.parent else f"FROM {gguf.as_posix()}"
    body = "\n".join([
        from_line,
        f"PARAMETER num_ctx {num_ctx}",
        f"PARAMETER num_thread {num_thread}",
        "PARAMETER temperature 0.7",
        "PARAMETER top_p 0.8",
        "PARAMETER top_k 20",
        "PARAMETER presence_penalty 1.5",
        "",
    ])
    dest.write_text(body, encoding="utf-8")
    return dest


def run_create(name: str, modelfile: Path) -> tuple[bool, str]:
    started = time.perf_counter()
    proc = subprocess.run(
        ["ollama", "create", name, "-f", str(modelfile)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=900,
    )
    elapsed = round(time.perf_counter() - started, 1)
    output = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, f"[{elapsed}s rc={proc.returncode}] {output.strip()[-400:]}"


def chat(name: str, prompt: str, persona: str, num_ctx: int, num_thread: int,
         tools: list | None = None, think: bool = False, num_predict: int = 192) -> dict:
    payload = {
        "model": name,
        "messages": [
            {"role": "system", "content": persona},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": think,
        "keep_alive": "5m",
        "options": {
            "seed": 3407, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
            "num_predict": num_predict, "num_ctx": num_ctx, "num_thread": num_thread,
        },
    }
    if tools:
        payload["tools"] = tools
    tick = time.perf_counter()
    raw = http_json(OLLAMA + "/api/chat", payload)
    wall = time.perf_counter() - tick
    message = raw.get("message") or {}
    content = message.get("content") or ""
    eval_count = raw.get("eval_count") or 0
    eval_duration = raw.get("eval_duration") or 0
    return {
        "prompt": prompt,
        "content": content,
        "thinking_leak": bool(re.search(r"<think|\[Start thinking\]", content, re.I)),
        "has_reasoning_field": bool(message.get("thinking")),
        "tool_calls": message.get("tool_calls") or [],
        "eval_count": eval_count,
        "tok_per_s": round(eval_count / (eval_duration / 1e9), 1) if eval_duration else None,
        "wall_seconds": round(wall, 2),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf")
    parser.add_argument("--name", default=DEFAULT_NAMES[0])
    parser.add_argument("--num-ctx", type=int, default=4096)
    parser.add_argument("--num-thread", type=int, default=6)
    parser.add_argument("--out", default=str(REPO / "data/logs/deploy-4b-qq-v2.json"))
    parser.add_argument("--dry-run", action="store_true", help="只打印将要做什么")
    parser.add_argument("--check", action="store_true", help="只校验 GGUF，不建模型")
    parser.add_argument("--skip-create", action="store_true", help="模型已存在时跳过 create")
    args = parser.parse_args()

    report: dict = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "steps": []}

    gguf = find_gguf(args.gguf)
    if gguf is None or not gguf.is_file():
        print("BLOCKED: 找不到 GGUF。用 --gguf 指定，或把 qwen35-4b-sft-v2-Q4_K_M.gguf 放到仓库根目录。")
        return 2

    info = inspect_gguf(gguf)
    report["gguf"] = info
    print(f"GGUF: {info['path']}")
    print(f"  {info['size_gib']} GiB / magic={info['magic']} ({'ok' if info['magic_ok'] else 'BAD'})")
    print(f"  sha256={info['sha256']}")

    if not info["magic_ok"]:
        print("FAILED: 文件头不是 GGUF，文件可能没传完或不是 GGUF。")
        return 3
    drift = abs(info["size_gib"] - EXPECTED_GIB)
    if drift > GIB_TOLERANCE:
        print(f"WARN: 大小 {info['size_gib']} GiB 与训练机 2.52 GiB 差 {drift:.2f} GiB，"
              f"可能不是同一个 Q4_K_M 产物。继续前请确认。")

    if args.check or args.dry_run:
        modelfile = gguf.parent / "Modelfile.qwen35-4b-v2.generated"
        write_modelfile(modelfile, gguf, args.num_ctx, args.num_thread)
        print(f"\n[dry-run] Modelfile -> {modelfile}")
        print(modelfile.read_text(encoding="utf-8"))
        print(f"[dry-run] 将执行: ollama create {args.name} -f <上述文件>")
        return 0

    try:
        version = http_json(OLLAMA + "/api/version", timeout=10)
        print(f"Ollama: {version.get('version')} @ {OLLAMA}")
    except (urllib.error.URLError, OSError) as exc:
        print(f"BLOCKED: Ollama 不可达 ({exc})。先启动 Ollama 服务。")
        return 4
    report["ollama"] = version

    modelfile = gguf.parent / "Modelfile.qwen35-4b-v2.generated"
    write_modelfile(modelfile, gguf, args.num_ctx, args.num_thread)

    existing = {m.get("name") for m in http_json(OLLAMA + "/api/tags").get("models", [])}
    if args.skip_create and args.name in existing:
        print(f"跳过 create：{args.name} 已存在")
        report["steps"].append({"step": "create", "skipped": True})
    else:
        ok, detail = run_create(args.name, modelfile)
        report["steps"].append({"step": "create", "ok": ok, "detail": detail})
        print(("OK  " if ok else "FAIL") + " create " + detail)
        if not ok:
            return 5

    persona_path = REPO / "config/qianwen35_persona.txt"
    persona = persona_path.read_text(encoding="utf-8") if persona_path.exists() else ""

    cases = [
        ("chat", "事情忙了一整天，现在只想放空。", None),
        ("fact_math", "只算结果：23 × 7 - 18。", None),
        ("tool_quota", "只读查一下 Codex 的剩余额度。", TOOLS),
    ]
    results = []
    for label, prompt, tools in cases:
        try:
            row = chat(args.name, prompt, persona, args.num_ctx, args.num_thread, tools=tools)
        except Exception as exc:  # noqa: BLE001 - 记录原始错误比中断有用
            row = {"error": f"{type(exc).__name__}: {exc}"}
        row["case"] = label
        results.append(row)
        print(f"\n[{label}] {prompt}")
        if "error" in row:
            print(f"  ERROR {row['error']}")
            continue
        print(f"  {row['content'][:300]!r}")
        print(f"  {row['eval_count']} tok @ {row['tok_per_s']} tok/s, {row['wall_seconds']}s")
        if row["thinking_leak"]:
            print("  WARN thinking 未关闭（输出里出现 <think> / [Start thinking]）")
        if row["tool_calls"]:
            print(f"  tool_calls: {json.dumps(row['tool_calls'], ensure_ascii=False)[:200]}")

    try:
        report["ps_after"] = http_json(OLLAMA + "/api/ps")
        loaded = report["ps_after"].get("models") or []
        if loaded:
            m = loaded[0]
            print(f"\n加载后: size={m.get('size')} vram={m.get('size_vram')} "
                  f"processor={m.get('processor', '?')}")
    except Exception:  # noqa: BLE001
        pass

    guard_note = []
    try:
        sys.path.insert(0, str(REPO))
        from assistant_output_guard import sanitize_text  # noqa: PLC0415

        for row in results:
            if "content" not in row:
                continue
            clean, info2 = sanitize_text(row["content"])
            guard_note.append({"case": row["case"], "changed": clean != row["content"], "info": info2})
    except Exception as exc:  # noqa: BLE001
        guard_note.append({"error": f"{type(exc).__name__}: {exc}"})
    report["guard_pass"] = guard_note

    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n报告: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
