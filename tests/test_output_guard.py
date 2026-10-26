"""Guards the two failures that only showed up after the 4B adapter got longer output:
fabricated conversation turns and leaked think blocks."""

import importlib.util
import json
import random
import sys
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

from assistant_output_guard import StreamGuard, sanitize_text, cut_at_role, split_think

ROOT = Path(__file__).resolve().parents[1]

# Verbatim tail of the C-variant output for heldout-curated-026, the case that
# looped to the 256-token ceiling during the 4b-bf16-sft-v2 evaluation.
RUNAWAY = (
    '先按第二段改，开头先不动。\n'
    'user\n'
    '第二段开头这句可以换个说法\n'
    'assistant\n'
    '<think>\n\n</think>\n\n'
    '第二段开头这句先换说法。\n'
    'user\n'
    '第二段开头这句可以换个说法\n'
    'assistant\n'
    '<think>\n\n</think>\n\n'
    '第二段开头这句可以换个说法。\n'
)

TOOL_THEN_ROLES = (
    '<tool_call>\n<function=harness_query>\n<parameter=operation>\nquota\n</parameter>\n'
    '</function>\n</tool_call>\n'
    'user\nassistant\n<think>\n\n</think>\n\nCodex 的剩余额度是 10000。\n'
)


class CutTests(unittest.TestCase):
    def test_plain_reply_is_untouched(self):
        text = '晚上好，今天有点累。先歇一会儿，不用急着把今天的事都理完。'
        self.assertEqual(cut_at_role(text), (text, None))

    def test_role_line_truncates_the_runaway_turn(self):
        kept, marker = cut_at_role(RUNAWAY)
        self.assertEqual(kept, '先按第二段改，开头先不动。\n')
        self.assertEqual(marker, 'user')

    def test_role_line_after_tool_call_keeps_the_tool_call(self):
        kept, marker = cut_at_role(TOOL_THEN_ROLES)
        self.assertTrue(kept.startswith('<tool_call>'))
        self.assertTrue(kept.rstrip().endswith('</tool_call>'))
        self.assertEqual(marker, 'user')

    def test_qwen_style_role_marker_is_caught(self):
        kept, marker = cut_at_role('好的。\n<|im_start|>user\n再问一句')
        self.assertEqual(kept.strip(), '好的。')
        self.assertEqual(marker, '<|im_start|>user')

    def test_role_word_inside_a_sentence_is_not_a_marker(self):
        text = '这个字段叫 user，别和 assistant 搞混。'
        self.assertEqual(cut_at_role(text), (text, None))


class ThinkTests(unittest.TestCase):
    def test_empty_think_block_is_removed(self):
        visible, thinking = split_think('<think>\n\n</think>\n\nMossChat 最新版是 8.7。')
        self.assertEqual(visible.strip(), 'MossChat 最新版是 8.7。')
        self.assertEqual(thinking.strip(), '')

    def test_unclosed_think_is_treated_as_thinking(self):
        visible, thinking = split_think('答案在这里。<think>还在想')
        self.assertEqual(visible, '答案在这里。')
        self.assertEqual(thinking, '还在想')


class SanitizeTests(unittest.TestCase):
    def test_runaway_turn_collapses_to_the_real_reply(self):
        clean, info = sanitize_text(RUNAWAY)
        self.assertEqual(clean, '先按第二段改，开头先不动。')
        self.assertEqual(info['role_marker'], 'user')
        self.assertTrue(info['changed'])

    def test_answer_only_inside_think_is_kept_not_dropped(self):
        clean, info = sanitize_text('<think>真正的答案</think>')
        self.assertEqual(clean, '真正的答案')
        self.assertEqual(info['stripped_think_chars'], len('真正的答案'))

    def test_empty_after_cleaning_is_flagged(self):
        clean, info = sanitize_text('<think>\n\n</think>')
        self.assertEqual(clean, '')
        self.assertTrue(info['empty'])

    def test_normal_reply_reports_no_change(self):
        text = '没有执行结果，暂时确认不了。'
        clean, info = sanitize_text(text)
        self.assertEqual(clean, text)
        self.assertNotIn('changed', info)


class StreamMatchesBatchTests(unittest.TestCase):
    """The invariant that matters: streaming must not leak what batching removes."""

    SAMPLES = [
        RUNAWAY,
        TOOL_THEN_ROLES,
        '<think>\n\n</think>\n\nMossChat 最新版是 8.7。',
        '晚上好，今天有点累。',
        '<think>真正的答案</think>',
        '<think>',
        '普通文本，没有任何标记。',
        '第一句。\n\n第二段，也许有\n换行。',
    ]

    def collect(self, chunks):
        guard = StreamGuard()
        parts = []
        for chunk in chunks:
            parts.append(guard.feed(chunk))
        parts.append(guard.flush())
        return ''.join(parts)

    def assert_matches(self, text):
        expected, _ = sanitize_text(text)
        self.assertEqual(self.collect(text), expected, f'whole-chunk split failed for {text!r}')
        self.assertEqual(self.collect(list(text)), expected, f'per-character split failed for {text!r}')

    def test_all_samples_match_whole_and_per_character(self):
        for text in self.SAMPLES:
            with self.subTest(text=text[:24]):
                self.assert_matches(text)

    def test_fixed_and_random_splits_match_batch(self):
        rng = random.Random(3407)
        for text in self.SAMPLES:
            for size in (2, 3, 5, 7, 11, 16, 29):
                chunks = [text[i:i + size] for i in range(0, len(text), size)]
                self.assertEqual(self.collect(chunks), sanitize_text(text)[0],
                                 f'split={size} text={text[:20]!r}')
            for _ in range(40):
                chunks, index = [], 0
                while index < len(text):
                    step = rng.randint(1, 9)
                    chunks.append(text[index:index + step])
                    index += step
                self.assertEqual(self.collect(chunks), sanitize_text(text)[0])

    def test_stream_stops_emitting_after_the_marker(self):
        guard = StreamGuard()
        emitted = ''.join(guard.feed(c) for c in list(RUNAWAY))
        emitted += guard.flush()
        self.assertTrue(guard.blocked)
        self.assertNotIn('user', emitted)
        self.assertNotIn('assistant', emitted)
        self.assertNotIn('<think>', emitted)

    def test_truncated_marker_across_chunks_never_leaks(self):
        # 'user' delivered one character at a time must not appear in output.
        guard = StreamGuard()
        emitted = ''
        for char in '你好。\nuser\n后续内容':
            emitted += guard.feed(char)
        emitted += guard.flush()
        self.assertEqual(emitted, '你好。')
        self.assertTrue(guard.blocked)


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


proxy = load('guard_proxy', 'ollama_fix_proxy.py')

CLEAN = '先按第二段改，开头先不动。'


class ProxyGuardIntegrationTests(unittest.TestCase):
    """The guard has to work through the real proxy, not just as a pure function."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        tempdir = Path(self.temp.name)

        class Upstream(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if payload.get("mode") == "stream":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.end_headers()
                    # Small chunks so role markers and tags straddle boundaries.
                    for index in range(0, len(RUNAWAY), 6):
                        delta = {"choices": [{"index": 0, "delta": {"content": RUNAWAY[index:index + 6]}}]}
                        self.wfile.write(b'data: ' + json.dumps(delta, ensure_ascii=False).encode("utf-8") + b'\n\n')
                        self.wfile.flush()
                    self.wfile.write(b'data: [DONE]\n\n')
                    self.wfile.flush()
                else:
                    text = "晚上好，今天有点累。" if payload.get("mode") == "clean" else RUNAWAY
                    body = {"choices": [{"index": 0, "message": {"role": "assistant", "content": text}}]}
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(body, ensure_ascii=False).encode("utf-8"))

            def log_message(self, *args):
                pass

        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), proxy.Handler)
        self.patches = [
            patch.object(proxy, "TARGET", f"http://127.0.0.1:{self.upstream.server_port}"),
            patch.object(proxy, "ROOT", tempdir),
        ]
        for item in self.patches:
            item.start()
        for server in (self.upstream, self.server):
            threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True).start()

    def tearDown(self):
        for server in (self.server, self.upstream):
            server.shutdown()
            server.server_close()
        for item in self.patches:
            item.stop()
        self.temp.cleanup()

    def post(self, payload):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=5)
        connection.request("POST", "/v1/chat/completions", json.dumps(payload),
                           {"Content-Type": "application/json"})
        return connection, connection.getresponse()

    def test_non_streamed_reply_is_cut_before_delivery(self):
        connection, response = self.post({"messages": [{"role": "user", "content": "开头那段要不要也改"}]})
        try:
            self.assertEqual(response.status, 200)
            data = json.loads(response.read())
            self.assertEqual(data["choices"][0]["message"]["content"], CLEAN)
        finally:
            connection.close()

    def test_streamed_reply_is_cut_before_delivery(self):
        connection, response = self.post({"mode": "stream", "messages": [{"role": "user", "content": "开头那段要不要也改"}]})
        try:
            self.assertEqual(response.status, 200)
            collected = []
            stopped = False
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    continue
                data = json.loads(payload)
                for choice in data.get("choices", []):
                    content = (choice.get("delta") or {}).get("content")
                    if content:
                        collected.append(content)
                    if choice.get("finish_reason") == "stop":
                        stopped = True
            joined = "".join(collected)
            self.assertEqual(joined.strip(), CLEAN)
            self.assertNotIn("user", joined)
            self.assertNotIn("assistant", joined)
            self.assertNotIn("<think>", joined)
            self.assertTrue(stopped, "truncated stream should signal finish_reason=stop")
        finally:
            connection.close()

    def test_clean_reply_passes_through_proxy_untouched(self):
        connection, response = self.post({"mode": "clean", "messages": [{"role": "user", "content": "在吗"}]})
        try:
            data = json.loads(response.read())
            self.assertEqual(data["choices"][0]["message"]["content"], "晚上好，今天有点累。")
        finally:
            connection.close()


if __name__ == '__main__':
    unittest.main()
