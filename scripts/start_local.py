"""Start the local stack without Start-Process.

PowerShell 5.1's Start-Process raises "已添加了具有相同键的项" when the parent
environment holds duplicate names that differ only in case (e.g. HTTPS_PROXY and
https_proxy). This launcher builds a de-duplicated environment and starts the
proxy and AstrBot directly, keeping the same ports and log files as
scripts/start_local.ps1.
"""
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(os.environ.get("APPDATA", "")) / "uv/tools/astrbot/Scripts/python.exe"
LOGS = ROOT / "data" / "logs"
PROXY_PORT = 8899
BOT_PORT = 6185
OLLAMA_PORT = 11434


def listening(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.3)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def clean_env() -> dict:
    """Collapse case-duplicate env names so child processes start cleanly."""
    merged = {}
    for key, value in os.environ.items():
        merged[key.upper()] = value
    return merged


CREATE_BREAKAWAY_FROM_JOB = 0x01000000


def spawn(script: str, out_name: str, err_name: str) -> subprocess.Popen:
    LOGS.mkdir(parents=True, exist_ok=True)
    out = open(LOGS / out_name, "w", encoding="utf-8", errors="replace")
    err = open(LOGS / err_name, "w", encoding="utf-8", errors="replace")
    base = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
        subprocess, "CREATE_NEW_PROCESS_GROUP", 0
    )
    command = [str(PYTHON), "-X", "utf8", str(ROOT / script)]
    # Breakaway lets the service outlive the shell that launched it; some job
    # objects forbid it, so fall back to a plain detached launch.
    for flags in (base | CREATE_BREAKAWAY_FROM_JOB, base):
        try:
            return subprocess.Popen(
                command, cwd=str(ROOT), env=clean_env(), stdout=out,
                stderr=err, creationflags=flags,
            )
        except OSError:
            continue
    raise SystemExit(f"Could not start {script}")


def wait_port(port: int, timeout: float, label: str) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if listening(port):
            return
        time.sleep(0.5)
    raise SystemExit(f"{label} did not open port {port} in {timeout:.0f}s")


def main() -> None:
    if not PYTHON.exists():
        raise SystemExit(f"AstrBot Python not found: {PYTHON}")
    if not listening(OLLAMA_PORT):
        raise SystemExit("Ollama is not listening on 11434; start it before this script.")

    if not listening(PROXY_PORT):
        proc = spawn("ollama_fix_proxy.py", "proxy.out.log", "proxy.err.log")
        print(f"Proxy started, PID {proc.pid}.")
    else:
        print("Proxy already listening; left alone.")
    wait_port(PROXY_PORT, 20, "Proxy")

    if not listening(BOT_PORT):
        proc = spawn("scripts/run_astrbot.py", "astrbot.out.log", "astrbot.err.log")
        print(f"AstrBot started, PID {proc.pid}.")
    else:
        print("AstrBot already listening; no duplicate process started.")
    wait_port(BOT_PORT, 90, "AstrBot")
    print("Stack ready. Dashboard: http://127.0.0.1:6185")


if __name__ == "__main__":
    main()
