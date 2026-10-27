"""Local-only control panel. It never exposes the control API to the network."""
import json
import os
import re
from pathlib import Path
import secrets
import socket
import statistics
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit, parse_qs
from urllib.request import build_opener, ProxyHandler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from assistant_security import redact_text

PORT = 8765
TOKEN = secrets.token_urlsafe(32)
LOGS = {"assistant": "astrbot.out.log", "errors": "astrbot.err.log", "proxy": "proxy.err.log", "startup": "startup.log"}
lock = threading.Lock()
operation = {"running": False, "message": ""}


def estimate_duration(action, text=None):
    """Median of recent completed comparable runs; ignore already-running starts."""
    if text is None:
        try:
            with (ROOT / "data" / "logs" / "startup.log").open("rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 200_000))
                text = f.read().decode("utf-8-sig", errors="replace")
        except OSError:
            text = ""
    samples, current = [], None
    for line in text.splitlines():
        match = re.match(r"\[(\d\d):(\d\d):(\d\d)\] (.*)", line)
        if not match:
            continue
        hour, minute, second = map(int, match.group(1, 2, 3))
        clock = hour * 3600 + minute * 60 + second
        message = match.group(4)
        if message.startswith("start root="):
            kind = "stop" if "stoponly=True" in message else "start" if "restart=False" in message and "stoponly=False" in message else None
            current = {"kind": kind, "start": clock, "spawned": False, "failed": False}
        elif current:
            if "astrbot pid " in message:
                current["spawned"] = True
            if message.startswith("FAILED:"):
                current["failed"] = True
            done = (current["kind"] == "start" and message == "exit 0") or (current["kind"] == "stop" and message == "Known local bot services stopped.")
            if done:
                duration = (clock - current["start"]) % 86400
                if current["kind"] == action and not current["failed"] and (action == "stop" or current["spawned"]) and 1 <= duration <= 180:
                    samples.append(duration)
                current = None
    return round(statistics.median(samples[-10:])) if samples else {"start": 35, "stop": 20}[action]


def operation_status():
    with lock:
        result = operation.copy()
    started = result.pop("started", None)
    result["elapsed"] = max(0, time.monotonic() - started) if started is not None and result["running"] else 0
    return result


def listening(port):
    with socket.socket() as sock:
        sock.settimeout(.4)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def gpu_status():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
                                capture_output=True, text=True, timeout=3, check=True,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        name, used, total = result.stdout.splitlines()[0].rsplit(",", 2)
        used, total = int(used.strip()), int(total.strip())
        if total <= 0:
            return None
        return {"name": name.strip(), "used": used, "total": total}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def loaded_models():
    """Read Ollama residency without loading or requesting a model."""
    try:
        opener = build_opener(ProxyHandler({}))
        with opener.open("http://127.0.0.1:11434/api/ps", timeout=2) as response:
            data = json.load(response)
        models = data.get("models") if isinstance(data, dict) else None
        if not isinstance(models, list):
            return None
        return [{"name": str(m.get("name") or m.get("model") or "未知模型"),
                 "size_vram": m.get("size_vram") if isinstance(m.get("size_vram"), int) else None}
                for m in models if isinstance(m, dict)]
    except (OSError, ValueError, TypeError):
        return None


def tail(name):
    path = ROOT / "data" / "logs" / LOGS[name]
    try:
        with path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 100_000))
            data = f.read().decode("utf-8", errors="replace")
        return format_log(redact_text(data))[-80000:]
    except FileNotFoundError:
        return "日志尚未生成。"


def format_log(text):
    """Remove terminal styling and visually separate entries without changing files."""
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    lines = []
    for line in text.splitlines():
        if " Result: {" in line:
            prefix, raw = line.split(" Result: ", 1)
            try:
                line = prefix + " Result:\n" + json.dumps(json.loads(raw), ensure_ascii=False, indent=2)
            except (ValueError, TypeError):
                pass
        if re.match(r"^\[\d\d:\d\d:\d\d(?:\.\d+)?\]", line) and lines and lines[-1]:
            lines.append("")
        lines.append(line)
    return "\n".join(lines)


def run_action(action):
    try:
        cmd = ["powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(ROOT / "scripts" / "start_local.ps1")]
        if action == "stop":
            cmd += ["-Restart", "-StopOnly"]
        # Service children can inherit pipes after PowerShell exits. Diagnostics
        # are already captured by startup.log, so do not wait for pipe EOF.
        result = subprocess.run(cmd, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                timeout=180, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        operation["message"] = "操作完成。" if result.returncode == 0 else "操作失败，请查看启动日志。"
    except Exception as exc:
        operation["message"] = "操作失败：" + redact_text(str(exc))
    finally:
        with lock:
            operation["running"] = False


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send(self, status, data, content_type="application/json; charset=utf-8"):
        if isinstance(data, (dict, list)):
            data = json.dumps(data, ensure_ascii=False).encode("utf-8")
        elif isinstance(data, str):
            data = data.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'; base-uri 'none'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(data)

    def valid_host(self):
        return self.headers.get("Host", "") in (f"127.0.0.1:{PORT}", f"localhost:{PORT}")

    def do_GET(self):
        if not self.valid_host():
            return self.send(403, {"error": "Invalid host"})
        path = urlsplit(self.path).path
        files = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "application/javascript; charset=utf-8"), "/style.css": ("style.css", "text/css; charset=utf-8")}
        if path in files:
            name, kind = files[path]
            return self.send(200, (ROOT / "dashboard" / name).read_bytes(), kind)
        if path == "/api/status":
            return self.send(200, {"assistant": listening(6185), "proxy": listening(8899), "ollama": listening(11434), "gpu": gpu_status(), "models": loaded_models(), "operation": operation_status(), "token": TOKEN})
        if path == "/api/log":
            name = parse_qs(urlsplit(self.path).query).get("name", [""])[0]
            if name in LOGS:
                return self.send(200, {"text": tail(name)})
        self.send(404, {"error": "Not found"})

    def do_POST(self):
        host = self.headers.get("Host", "")
        if not self.valid_host() or self.headers.get("Origin") != f"http://{host}" or self.headers.get("X-Control-Token") != TOKEN:
            return self.send(403, {"error": "Forbidden"})
        action = urlsplit(self.path).path.removeprefix("/api/")
        if action not in ("start", "stop"):
            return self.send(404, {"error": "Not found"})
        with lock:
            if operation["running"]:
                return self.send(409, {"error": "已有操作正在进行"})
            operation.update(running=True, message="正在执行，请稍候……", action=action, estimate=estimate_duration(action), started=time.monotonic())
        threading.Thread(target=run_action, args=(action,), daemon=True).start()
        self.send(202, {"ok": True})


class LocalServer(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR permits another process to bind the same port.
    allow_reuse_address = False

    def server_bind(self):
        if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


if __name__ == "__main__":
    LocalServer(("127.0.0.1", PORT), Handler).serve_forever()
