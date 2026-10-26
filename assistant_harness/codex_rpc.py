"""Version-matched Codex App Server JSONL transport; no credential copying."""

import asyncio
from collections import deque
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

# Editor extensions ship the CLI under a versioned directory that is never
# added to PATH, so `shutil.which` alone reports a hosted Codex as missing.
EDITOR_ROOTS = (".vscode", ".vscode-insiders", ".vscode-oss", ".cursor", ".windsurf")


def _executable_name():
    return "codex.exe" if os.name == "nt" else "codex"


def _version_key(value):
    return tuple(int(part) for part in value.split(".") if part.isdigit())


def extension_candidates(home=None):
    """Newest-first CLI binaries bundled inside an installed editor extension."""
    executable = _executable_name()
    root_home = Path(home) if home else Path.home()
    found = []
    for editor in EDITOR_ROOTS:
        root = root_home / editor / "extensions"
        if not root.is_dir():
            continue
        for directory in root.glob("openai.chatgpt-*"):
            if not directory.is_dir():
                continue
            match = re.search(r"openai\.chatgpt-(\d+(?:\.\d+)*)", directory.name)
            version = _version_key(match.group(1)) if match else ()
            for binary in directory.glob(f"bin/*/{executable}"):
                if binary.is_file():
                    found.append((version, binary))
    found.sort(key=lambda item: item[0], reverse=True)
    return [binary for _, binary in found]


def static_candidates(home=None, environ=None):
    """Remaining known install layouts, checked after PATH and extensions."""
    executable = _executable_name()
    environment = os.environ if environ is None else environ
    root_home = Path(home) if home else Path.home()
    bases = [root_home / ".codex" / "bin", root_home / ".local" / "bin",
             root_home / ".npm-global" / "bin", root_home / ".bun" / "bin",
             root_home / ".volta" / "bin"]
    if os.name == "nt":
        for variable in ("LOCALAPPDATA", "APPDATA"):
            base = environment.get(variable)
            if base:
                bases.append(Path(base) / "npm")
        bases.append(root_home / "AppData" / "Local" / "Programs" / "codex")
    else:
        bases.extend([Path("/usr/local/bin"), Path("/opt/homebrew/bin"), Path("/usr/bin")])
    return [base / executable for base in bases]


def resolve_codex_executable(home=None, environ=None):
    """Locate the Codex CLI without copying or reading any credential file."""
    environment = os.environ if environ is None else environ
    override = environment.get("CODEX_CLI_PATH")
    if override:
        path = Path(override).expanduser()
        if path.is_file():
            return path
    located = shutil.which("codex", path=environment.get("PATH", ""))
    if located:
        return Path(located)
    for candidate in (*extension_candidates(home), *static_candidates(home, environment)):
        if candidate.is_file():
            return candidate
    return None


class RpcError(RuntimeError):
    def __init__(self, error):
        self.code = error.get("code")
        super().__init__(error.get("message", "Codex RPC error"))


class CodexRpc:
    def __init__(self, command=None, timeout=30):
        executable = resolve_codex_executable()
        if command is None and not executable:
            raise FileNotFoundError(
                "Codex CLI is not installed: no `codex` on PATH, no bundled binary under "
                "an editor extension (e.g. ~/.vscode/extensions/openai.chatgpt-*/bin/*/codex.exe), "
                "and no known install location. Set CODEX_CLI_PATH to the full path to override."
            )
        self.command = command or [str(executable), "app-server", "--listen", "stdio://"]
        self.timeout = timeout
        self.proc = None
        self.pending = {}
        self.sequence = 0
        self.events = asyncio.Queue(maxsize=256)
        self.server_requests = asyncio.Queue(maxsize=32)
        self.stderr = deque(maxlen=12)
        self.tasks = []
        self._write_lock = asyncio.Lock()
        self._start_lock = asyncio.Lock()
        self.ready = False

    async def start(self):
        async with self._start_lock:
            if self.ready and self.proc and self.proc.returncode is None:
                return
            if self.proc:
                await self.close()
            await self._start()

    async def _start(self):
        flags = 0
        if os.name == "nt":
            flags = subprocess.CREATE_NO_WINDOW | subprocess.BELOW_NORMAL_PRIORITY_CLASS
        env = dict(os.environ)
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "RAYON_NUM_THREADS"):
            env[key] = "4"
        self.proc = await asyncio.create_subprocess_exec(
            *self.command, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, creationflags=flags, env=env, limit=8*1024*1024,
        )
        self.tasks = [asyncio.create_task(self._read()), asyncio.create_task(self._read_stderr())]
        try:
            self.initialized = await self.request("initialize", {
                "clientInfo":{"name":"qq_assistant","title":"Local QQ Assistant","version":"0.1.0"},
                "capabilities":{"experimentalApi":True},
            })
            await self.notify("initialized", {})
            self.ready = True
        except BaseException:
            await self.close()
            raise

    async def _write(self, message):
        if not self.proc or self.proc.returncode is not None:
            raise ConnectionError("Codex App Server is offline")
        async with self._write_lock:
            self.proc.stdin.write((json.dumps(message, ensure_ascii=False)+"\n").encode("utf-8"))
            await self.proc.stdin.drain()

    async def request(self, method, params=None):
        self.sequence += 1
        identity = self.sequence
        future = asyncio.get_running_loop().create_future()
        self.pending[identity] = future
        try:
            await self._write({"id":identity,"method":method,"params":params or {}})
            return await asyncio.wait_for(future, self.timeout)
        finally:
            self.pending.pop(identity, None)

    async def notify(self, method, params):
        await self._write({"method":method,"params":params})

    async def answer_server_request(self, identity, result):
        await self._write({"id":identity,"result":result})

    async def _read(self):
        error = ConnectionError("Codex App Server connection closed")
        try:
            while raw := await self.proc.stdout.readline():
                message = json.loads(raw)
                if "method" in message:
                    queue = self.server_requests if "id" in message else self.events
                    if queue.full():
                        # Never silently drop an approval request.
                        if "id" in message:
                            await self._write({"id":message["id"],"error":{"code":-32000,"message":"Approval queue unavailable; request not approved"}})
                            continue
                        queue.get_nowait()
                    queue.put_nowait(message)
                elif (future := self.pending.get(message.get("id"))) and not future.done():
                    if "error" in message:
                        future.set_exception(RpcError(message["error"]))
                    else:
                        future.set_result(message.get("result"))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            error = exc
        finally:
            self.ready = False
            for future in list(self.pending.values()):
                if not future.done(): future.set_exception(error)

    async def _read_stderr(self):
        # Keep only error type/transport diagnostics in memory, never persist raw
        # stderr: upstream diagnostics may include credential-bearing URLs.
        while await self.proc.stderr.readline():
            self.stderr.append("upstream diagnostic received")

    async def close(self):
        self.ready = False
        if self.proc and self.proc.returncode is None:
            self.proc.terminate()
            try: await asyncio.wait_for(self.proc.wait(), 5)
            except asyncio.TimeoutError:
                self.proc.kill()
                await self.proc.wait()
        for task in self.tasks: task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks = []
