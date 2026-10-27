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

REQUEST_TIMEOUT = 120
# 上下文溢出的补救阶梯，从最温和到最激进。实测（2026-10-27）单靠丢旧轮次治不了：
# 爆窗口的往往是尾部那条大工具结果，丢前面根本不改变尾部，最后还是 500。所以
# 前三级丢轮次、后两级把工具结果截短——这才是能真正救回来的那一步。
REPAIR_STEPS = (
    ("drop", 1),
    ("drop", 2),
    ("drop", 4),
    ("shrink", 6000),
    ("shrink", 3000),
)
MAX_CONTEXT_RETRIES = len(REPAIR_STEPS)
# Ollama 在上下文装不下时不会报"太长"，而是把消息截断到模板认不出来，然后抛
# Jinja 异常——所以只能按错误文本识别，不能按状态码。
OVERFLOW_MARKERS = (
    "No user query found",
    "exceeds the available context size",
    "input length exceeds",
)


def is_context_overflow(status, body):
    """True when upstream rejected the request because the context did not fit."""
    if status != 500:
        return False
    text = body.decode("utf-8", "replace") if isinstance(body, bytes) else str(body)
    return any(marker in text for marker in OVERFLOW_MARKERS)


def drop_oldest_round(messages):
    """丢掉最旧的一轮对话，保留 system 与其余轮次；无可丢时返回 None。

    按"下一个 user 消息"整段切，而不是按条数切：OpenAI 协议要求 assistant 的
    tool_calls 后面紧跟 tool 结果，按条数切会把配对拆开，上游会直接 400。
    """
    if not isinstance(messages, list) or not messages:
        return None
    start = 1 if isinstance(messages[0], dict) and messages[0].get("role") == "system" else 0
    seen_user = 0
    for index in range(start, len(messages)):
        message = messages[index]
        if isinstance(message, dict) and message.get("role") == "user":
            seen_user += 1
            if seen_user == 2:  # 第一条 user 到第二条之前，正好是一整轮
                return messages[:start] + messages[index:]
    return None


def drop_oldest_rounds(messages, count=1):
    """丢掉最旧的 count 轮；一轮都丢不掉时返回 None，避免空转重试。"""
    dropped = False
    for _ in range(max(1, count)):
        trimmed = drop_oldest_round(messages)
        if trimmed is None:
            break
        messages, dropped = trimmed, True
    return messages if dropped else None


def clip_text(text, limit):
    """保留头和尾，中间说明截掉了多少——搜索结果的有用部分通常就在前面。"""
    if len(text) <= limit:
        return text
    head = text[: limit * 2 // 3]
    tail = text[-(limit // 3):]
    return f"{head}\n…（原文 {len(text)} 字，已截断）…\n{tail}"


def shrink_tool_results(messages, limit):
    """把所有超长的 tool 结果截到 limit 字；没有可截的返回 None。"""
    if not isinstance(messages, list):
        return None
    changed = False
    trimmed = []
    for message in messages:
        if isinstance(message, dict) and message.get("role") == "tool":
            content = message.get("content")
            if isinstance(content, str) and len(content) > limit:
                message = dict(message, content=clip_text(content, limit))
                changed = True
        trimmed.append(message)
    return trimmed if changed else None


def apply_repair(messages, kind, amount):
    """执行一级补救；这一步不适用时返回 None，由调用方换下一级。"""
    if kind == "drop":
        return drop_oldest_rounds(messages, amount)
    if kind == "shrink":
        return shrink_tool_results(messages, amount)
    return None


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
        was_blocked=guard.blocked
        content=delta.get('content')
        if isinstance(content,str) and content:
            kept=guard.feed(content)
            if kept:delta['content']=kept
            else:delta.pop('content',None)
        if guard.blocked:
            # The turn was cut; tell the client to stop instead of waiting for more.
            choice['finish_reason']='stop'
            if not was_blocked:LOG.warning('output guard truncated a streamed reply at a role marker')
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

    def _open(self, body):
        request = urllib.request.Request(TARGET + self.path, data=body, method="POST")
        for key in ("Content-Type", "Authorization"):
            if self.headers.get(key):
                request.add_header(key, self.headers[key])
        return urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT)

    def _forward(self, body, payload):
        """转发一次，遇上下文溢出就按 REPAIR_STEPS 逐级补救后重试。

        返回上游响应对象；返回 None 表示已经回过错误，调用方直接结束。
        只有 /v1/chat/completions 带 payload，其它路径一律原样透传。
        """
        step = 0
        while True:
            try:
                return self._open(body)
            except urllib.error.HTTPError as exc:
                detail = exc.read()
                content_type = exc.headers.get("Content-Type", "application/json")
                repaired = None
                if payload is not None and is_context_overflow(exc.code, detail):
                    # 跳过不适用的级别（没有旧轮次可丢、没有工具结果可截），
                    # 而不是就此放弃——最后那条大工具结果才是最常见的元凶。
                    while step < len(REPAIR_STEPS) and repaired is None:
                        kind, amount = REPAIR_STEPS[step]
                        step += 1
                        repaired = apply_repair(payload.get("messages"), kind, amount)
                if repaired is None:
                    self._reply(exc.code, detail, content_type)
                    return None
                payload["messages"] = repaired
                body = json.dumps(prepare_payload(payload), ensure_ascii=False).encode("utf-8")
                save_json("_last_forwarded_request.json", payload)
                LOG.warning(
                    "context overflow: repair %d/%d (%s %s), %d messages left",
                    step, len(REPAIR_STEPS), kind, amount, len(repaired),
                )

    def do_POST(self):
        try:
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            with BACKEND_LOCK:
                payload = None
                if self.path == "/v1/chat/completions":
                    payload = json.loads(body)
                    save_json("_last_request.json", payload)
                    body = json.dumps(prepare_payload(payload), ensure_ascii=False).encode("utf-8")
                    save_json("_last_forwarded_request.json", payload)
                resp = self._forward(body, payload)
                if resp is None:  # 已经回过错误（或重试用尽）
                    return
                with resp:
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
