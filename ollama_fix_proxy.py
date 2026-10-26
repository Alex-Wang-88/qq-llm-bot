"""Loopback Ollama compatibility proxy; serial inference and scoped chat budgets."""

import json
import logging
import os
from pathlib import Path
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from assistant_security import redact
from assistant_output_guard import StreamGuard, sanitize_text

TARGET = os.environ.get("QQ_BOT_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
ROOT = Path(__file__).resolve().parent
POLICY_MARKER = "<!-- local_chat_policy:v1 -->"
BACKEND_LOCK = threading.Lock()
HOP_HEADERS = {"transfer-encoding", "content-length", "connection", "keep-alive"}
LOG = logging.getLogger(__name__)


def flatten_content(messages):
    """Flatten text-only parts. Keep multimodal messages intact, including images."""
    for message in messages or []:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        if not all(isinstance(p, str) or isinstance(p, dict) and p.get("type") == "text" for p in content):
            continue
        message["content"] = "\n".join(
            p if isinstance(p, str) else str(p.get("text", "")) for p in content
        )


def prepare_payload(data):
    flatten_content(data.get("messages"))
    system = "\n".join(
        m.get("content", "") for m in data.get("messages", [])
        if m.get("role") == "system" and isinstance(m.get("content"), str)
    )
    if POLICY_MARKER in system:
        policy = system.rsplit(POLICY_MARKER, 1)[1]
        limit = 2048 if "response_budget=expanded" in policy or "thinking=on" in policy else 384
        current = data.get("max_tokens")
        data["max_tokens"] = min(current, limit) if isinstance(current, int) and current > 0 else limit
        # Formal Ollama OpenAI thinking controls. Only a trusted, per-turn
        # policy marker can opt in; no legacy /no_think prompt suffix.
        data["reasoning_effort"]="high" if "thinking=on" in policy else "none"
    return data


def save_json(name, data):
    target = ROOT / name
    temp = target.with_suffix(".tmp")
    temp.write_text(json.dumps(redact(data), ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(target)


def normalize_reasoning(data):
    """Ollama emits reasoning; installed AstrBot expects reasoning_content."""
    for choice in data.get('choices',[]):
        for key in ('message','delta'):
            message=choice.get(key)
            if isinstance(message,dict) and message.get('reasoning') and not message.get('reasoning_content'):
                message['reasoning_content']=message['reasoning']
    return data


def normalize_sse_line(line):
    if line.startswith(b'data:') and line[5:].strip()!=b'[DONE]':
        try:return b'data: '+json.dumps(normalize_reasoning(json.loads(line[5:])),ensure_ascii=False).encode('utf-8')+b'\n'
        except (ValueError,TypeError):pass
    return line


def guard_response(data):
    """Drop fabricated turns and stray think blocks from a non-streamed reply."""
    for choice in data.get('choices',[]):
        for key in ('message','delta'):
            message=choice.get(key)
            if not isinstance(message,dict):continue
            content=message.get('content')
            if not isinstance(content,str) or not content:continue
            clean,info=sanitize_text(content)
            if not info.get('changed'):continue
            message['content']=clean
            LOG.warning('output guard cleaned %s: %s',key,{k:v for k,v in info.items() if k!='changed'})
    return data


def guard_sse_line(line,guard):
    """Filter one streamed line. Returns a list: flushing can add a trailing delta."""
    if not line.startswith(b'data:'):return [line]
    payload=line[5:].strip()
    if payload==b'[DONE]':
        pending=guard.flush()
        if not pending:return [line]
        extra=b'data: '+json.dumps({'choices':[{'index':0,'delta':{'content':pending}}]},ensure_ascii=False).encode('utf-8')+b'\n'
        return [extra,line]
    if not payload:return [line]
    try:data=json.loads(payload)
    except (ValueError,TypeError):return [line]
    for choice in data.get('choices',[]):
        delta=choice.get('delta')
        if not isinstance(delta,dict):continue
        content=delta.get('content')
        if isinstance(content,str) and content:
            kept=guard.feed(content)
            if kept:delta['content']=kept
            else:delta.pop('content',None)
        if guard.blocked:
            # The turn was cut; tell the client to stop instead of waiting for more.
            choice['finish_reason']='stop'
            LOG.warning('output guard truncated a streamed reply at a role marker')
    return [b'data: '+json.dumps(normalize_reasoning(data),ensure_ascii=False).encode('utf-8')+b'\n']


class Handler(BaseHTTPRequestHandler):
    def _reply(self, status, body, content_type="application/json"):
        try:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except ConnectionError:
            self.close_connection=True

    def do_POST(self):
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            with BACKEND_LOCK:
                if self.path == "/v1/chat/completions":
                    data = json.loads(body)
                    save_json("_last_request.json", data)
                    body = json.dumps(prepare_payload(data), ensure_ascii=False).encode("utf-8")
                    save_json("_last_forwarded_request.json", data)
                req = urllib.request.Request(TARGET + self.path, data=body, method="POST")
                for key in ("Content-Type", "Authorization"):
                    if self.headers.get(key):
                        req.add_header(key, self.headers[key])
                with urllib.request.urlopen(req, timeout=120) as resp:
                    streaming = "text/event-stream" in resp.headers.get("Content-Type", "")
                    if streaming:
                        self.send_response(resp.status)
                        for key, value in resp.getheaders():
                            if key.lower() not in HOP_HEADERS:
                                self.send_header(key, value)
                        self.send_header("Connection", "close")
                        self.end_headers()
                        self.close_connection = True
                        guard=StreamGuard()
                        for line in resp:
                            if self.path != '/v1/chat/completions':
                                self.wfile.write(line);self.wfile.flush();continue
                            for chunk in guard_sse_line(line,guard):
                                self.wfile.write(chunk);self.wfile.flush()
                    else:
                        result=resp.read()
                        if self.path == '/v1/chat/completions':
                            data=guard_response(normalize_reasoning(json.loads(result)))
                            result=json.dumps(data,ensure_ascii=False).encode('utf-8')
                        self._reply(resp.status, result, resp.headers.get("Content-Type", "application/json"))
        except urllib.error.HTTPError as exc:
            self._reply(exc.code, exc.read(), exc.headers.get("Content-Type", "application/json"))
        except ConnectionError:
            pass
        except (ValueError, TypeError) as exc:
            self._reply(400, json.dumps({"error": {"message": str(exc)}}).encode())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self._reply(502, json.dumps({"error": {"message": str(exc)}}).encode())

    def do_GET(self):
        if self.path == "/health":
            self._reply(200, b'{"status":"ok","proxy":"chat-tuning-v1","inference_concurrency":1}')
            return
        try:
            with urllib.request.urlopen(TARGET + self.path, timeout=10) as resp:
                self._reply(resp.status, resp.read(), resp.headers.get("Content-Type", "application/json"))
        except urllib.error.HTTPError as exc:
            self._reply(exc.code, exc.read())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            self._reply(502, json.dumps({"error": {"message": str(exc)}}).encode())

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8899), Handler).serve_forever()
