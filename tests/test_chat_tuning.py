import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


proxy = load("tuned_proxy", "ollama_fix_proxy.py")
tuning = load("tuned_config", "scripts/apply_chat_tuning.py")
web = load("page_extract", "data/plugins/astrbot_plugin_searxng_search/web_text.py")
natural = load("natural_plugin_test", "data/plugins/astrbot_plugin_chat_natural/main.py")
# 插件的闲置阈值状态默认落在 data/plugin_data/ 下；测试必须指到临时目录，
# 否则 prepare_chat 会把假的 UMO 写进真实的 session_state.json。
natural.SESSION_STATE = Path(tempfile.mkdtemp(prefix="chat-tuning-state-")) / "session_state.json"


class Event:
    unified_msg_origin = "test:FriendMessage:u1"

    def __init__(self, persona="千问娘"):
        self.extra = {"_trace_persona_id": persona}
        self.result = None

    def get_extra(self, key):
        return self.extra.get(key)

    def set_extra(self, key, value):
        self.extra[key] = value

    def get_result(self):
        return self.result


class ChatPolicyTests(unittest.IsolatedAsyncioTestCase):
    async def test_removes_framework_added_tools_but_keeps_search(self):
        plugin = natural.NaturalChatPlugin(SimpleNamespace())
        tools = SimpleNamespace(tools=[SimpleNamespace(name=n) for n in ["web_search", "fetch_web_page", "future_task", "send_message_to_user"]])
        tools.remove_tool = lambda name: tools.tools.__setitem__(slice(None), [x for x in tools.tools if x.name != name])
        req = SimpleNamespace(func_tool=tools, prompt="今天好累", system_prompt="persona")
        await plugin.prepare_chat(Event(), req)
        self.assertEqual([t.name for t in tools.tools], ["web_search", "fetch_web_page"])
        self.assertIn("response_budget=brief", req.system_prompt)

    async def test_explicit_detail_is_not_given_brief_budget(self):
        plugin = natural.NaturalChatPlugin(SimpleNamespace())
        req = SimpleNamespace(func_tool=None, prompt="请详细解释原理，一步一步说", system_prompt="")
        await plugin.prepare_chat(Event(), req)
        self.assertIn("response_budget=expanded", req.system_prompt)

    async def test_other_persona_is_untouched(self):
        plugin = natural.NaturalChatPlugin(SimpleNamespace())
        req = SimpleNamespace(func_tool=None, prompt="你好", system_prompt="other")
        await plugin.prepare_chat(Event("other"), req)
        self.assertEqual(req.system_prompt, "other")

    async def test_meme_cooldown_only_counts_completed_replies(self):
        plugin = natural.NaturalChatPlugin(SimpleNamespace())
        event = Event()
        req = SimpleNamespace(func_tool=None, prompt="哈哈", system_prompt="")
        await plugin.prepare_chat(event, req)
        await plugin.note_meme(event, SimpleNamespace(completion_text="&&happy&&"))
        self.assertFalse(plugin._meme_history)
        await plugin.record_sent(event)
        follow = SimpleNamespace(func_tool=None, prompt="嘿嘿", system_prompt="")
        await plugin.prepare_chat(Event(), follow)
        self.assertIn("刚发过表情", follow.system_prompt)

    async def test_repeated_meme_is_removed_before_delivery(self):
        plugin = natural.NaturalChatPlugin(SimpleNamespace())
        plugin._meme_history[Event.unified_msg_origin] = [True]
        event=Event()
        await plugin.prepare_chat(event, SimpleNamespace(func_tool=None,prompt="嗯嗯",system_prompt=""))
        response=SimpleNamespace(completion_text="嗯嗯 &&happy&&")
        await plugin.note_meme(event,response)
        self.assertEqual(response.completion_text,"嗯嗯")

    async def test_explicit_meme_request_overrides_cooldown(self):
        plugin = natural.NaturalChatPlugin(SimpleNamespace())
        plugin._meme_history[Event.unified_msg_origin] = [True]
        event=Event()
        await plugin.prepare_chat(event, SimpleNamespace(func_tool=None,prompt="再发一张开心表情",system_prompt=""))
        response=SimpleNamespace(completion_text="&&happy&&")
        await plugin.note_meme(event,response)
        self.assertEqual(response.completion_text,"&&happy&&")

    async def test_plain_text_request_is_enforced(self):
        plugin = natural.NaturalChatPlugin(SimpleNamespace())
        event=Event()
        await plugin.prepare_chat(event, SimpleNamespace(func_tool=None,prompt="纯文字回复，别发表情",system_prompt=""))
        response=SimpleNamespace(completion_text="好 &&happy&&")
        await plugin.note_meme(event,response)
        self.assertEqual(response.completion_text,"好")

    async def test_harness_tool_only_appears_for_harness_intent(self):
        plugin=natural.NaturalChatPlugin(SimpleNamespace())
        for prompt,expected in [('看看 Codex 还剩多少额度',True),('今天好累',False)]:
            tools=SimpleNamespace(tools=[SimpleNamespace(name=n) for n in ['web_search','harness_query']])
            tools.remove_tool=lambda name:tools.tools.__setitem__(slice(None),[t for t in tools.tools if t.name!=name])
            req=SimpleNamespace(func_tool=tools,prompt=prompt,system_prompt='')
            await plugin.prepare_chat(Event(),req)
            self.assertEqual(any(t.name=='harness_query' for t in tools.tools),expected)

    async def test_search_preamble_is_dropped_from_history(self):
        """Preamble lines in history were copied verbatim on 2 of 10 replays."""
        plugin=natural.NaturalChatPlugin(SimpleNamespace())
        req=SimpleNamespace(func_tool=None,prompt="搜一下今天的 A 股行情",system_prompt="",
                            contexts=[{"role":"assistant","content":"行，我这就去搜一下今天的行情。"},
                                      {"role":"assistant","content":"好的，我马上搜索。"},
                                      {"role":"assistant","content":"行，再试一次。"},
                                      {"role":"user","content":"搜一下今天的 A 股行情"}])
        await plugin.prepare_chat(Event(),req)
        self.assertEqual([m["role"] for m in req.contexts],["user"])

    async def test_preamble_keeps_its_tool_call_pair(self):
        plugin=natural.NaturalChatPlugin(SimpleNamespace())
        call={"id":"c1","type":"function","function":{"name":"web_search","arguments":"{}"}}
        req=SimpleNamespace(func_tool=None,prompt="搜一下今天的 A 股行情",system_prompt="",
                            contexts=[{"role":"assistant","content":[{"type":"text","text":"好的，我这就去查一下。"}],"tool_calls":[call]},
                                      {"role":"tool","content":"{}","tool_call_id":"c1"},
                                      {"role":"user","content":"搜一下今天的 A 股行情"}])
        await plugin.prepare_chat(Event(),req)
        self.assertEqual(len(req.contexts),3)
        self.assertEqual(req.contexts[0]["content"],"")
        self.assertEqual(req.contexts[0]["tool_calls"],[call])

    def test_real_answers_are_not_mistaken_for_preambles(self):
        self.assertTrue(natural.is_announcement("好的，我这就去查。"))
        self.assertTrue(natural.is_announcement("行，再试一次。"))
        for text in ("行啊，你说。","好，那我先记着。","你可以自己搜一下今天的行情。",
                     "好的，今天上证收在 3697.52 点，深证成指跌了 1.19%，成交额比昨天更低一些。"):
            self.assertFalse(natural.is_announcement(text),text)

    async def test_history_marker_left_by_model_does_not_starve_tools(self):
        """A stray &&search&& in history made tool calls fail 2 out of 3 times.

        Stored history keeps assistant content as content parts, so both the list
        and the plain-string shapes have to be cleaned.
        """
        plugin=natural.NaturalChatPlugin(SimpleNamespace())
        req=SimpleNamespace(func_tool=None,prompt="刚才没搜成功，你再搜一遍",system_prompt="",
                            contexts=[{"role":"assistant","content":[{"type":"text","text":"昨天那个报错还在，我再看看。&&search&&"}]},
                                      {"role":"assistant","content":"等我把结果捞出来。&&search&&"},
                                      {"role":"user","content":"搜一下今天的A股"}])
        await plugin.prepare_chat(Event(),req)
        self.assertEqual(req.contexts[0]["content"],"昨天那个报错还在，我再看看。")
        self.assertEqual(req.contexts[1]["content"],"等我把结果捞出来。")
        self.assertEqual(len(req.contexts),3)
        self.assertIn("intent=query",req.system_prompt)

    async def test_marker_only_turn_is_dropped_from_history(self):
        plugin=natural.NaturalChatPlugin(SimpleNamespace())
        req=SimpleNamespace(func_tool=None,prompt="在吗",system_prompt="",
                            contexts=[{"role":"assistant","content":[{"type":"text","text":"&&search&&"}]},
                                      {"role":"user","content":"在吗"}])
        await plugin.prepare_chat(Event(),req)
        self.assertEqual([m["role"] for m in req.contexts],["user"])

    async def test_fabricated_sticker_marker_is_not_delivered(self):
        plugin=natural.NaturalChatPlugin(SimpleNamespace())
        event=Event()
        await plugin.prepare_chat(event,SimpleNamespace(func_tool=None,prompt="今天忙吗",system_prompt=""))
        response=SimpleNamespace(completion_text="行，我这就去搜。&&search&&")
        await plugin.note_meme(event,response)
        self.assertEqual(response.completion_text,"行，我这就去搜。")

    async def test_real_sticker_marker_survives_cleanup(self):
        plugin=natural.NaturalChatPlugin(SimpleNamespace())
        event=Event()
        await plugin.prepare_chat(event,SimpleNamespace(func_tool=None,prompt="来张开心表情",system_prompt=""))
        response=SimpleNamespace(completion_text="好 &&happy&&")
        await plugin.note_meme(event,response)
        self.assertEqual(response.completion_text,"好 &&happy&&")


class BubbleShapeTests(unittest.IsolatedAsyncioTestCase):
    """One reply is one message unless it is genuinely long."""

    STOCK_REPLY = (
        "好的，我找到了今天 A 股的行情信息：\n\n"
        "上证指数收盘报 3697.52 点，涨 0.19%，主要靠一桶油带动；深证成指收 11534.38 点，跌 1.19%；"
        "创业板指收 2638.23 点，跌 1.71%。AI 硬件板块一天吐掉 160 亿。\n\n"
        "涨停股有 54 家，连板梯队中机器人概念、三季报增长、食品饮料等是热点题材。\n\n"
        "你感兴趣的是哪个具体板块或个股的动向？我可以进一步帮你分析。"
    )

    def plugin(self):
        return natural.NaturalChatPlugin(SimpleNamespace())

    def test_one_sentence_is_one_bubble(self):
        self.assertEqual(natural.plan_bubbles("行，再试一次。"), ["行，再试一次。"])

    def test_stock_answer_is_delivered_as_one_message(self):
        bubbles = natural.plan_bubbles(self.STOCK_REPLY)
        self.assertEqual(len(bubbles), 1)
        self.assertNotIn("\n\n", bubbles[0])
        self.assertIn("3697.52", bubbles[0])

    def test_only_a_long_answer_may_use_two_bubbles(self):
        sentence = "上证指数今天收在三千六百九十七点，涨了不到半个百分点，成交额比昨天更低。"
        text = sentence * 12
        bubbles = natural.plan_bubbles(text)
        self.assertEqual(len(bubbles), 2)
        self.assertEqual("".join(bubbles), text.strip())
        self.assertTrue(all(len(part) >= natural.BUBBLE_MIN_SIDE for part in bubbles))

    def test_long_text_without_a_boundary_stays_whole(self):
        self.assertEqual(len(natural.plan_bubbles("盘" * 400)), 1)

    def test_blank_line_runs_are_collapsed(self):
        self.assertEqual(natural.normalize_reply("第一段\n\n\n  第二段  \n"), "第一段\n第二段")

    async def test_fragmented_chain_is_merged_before_delivery(self):
        event = Event()
        await self.plugin().prepare_chat(event, SimpleNamespace(func_tool=None, prompt="在吗", system_prompt=""))
        event.result = SimpleNamespace(chain=[natural.Plain("第一句。"), natural.Plain("第二句。"), natural.Plain("第三句。")])
        await self.plugin().shape_bubbles(event)
        self.assertEqual([comp.text for comp in event.result.chain], ["第一句。第二句。第三句。"])

    async def test_sticker_keeps_its_own_bubble(self):
        event = Event()
        await self.plugin().prepare_chat(event, SimpleNamespace(func_tool=None, prompt="在吗", system_prompt=""))
        sticker = SimpleNamespace(type="image")
        event.result = SimpleNamespace(chain=[natural.Plain("行啊\n\n"), sticker])
        await self.plugin().shape_bubbles(event)
        self.assertEqual(len(event.result.chain), 2)
        self.assertEqual(event.result.chain[0].text, "行啊")
        self.assertIs(event.result.chain[1], sticker)

    async def test_other_persona_keeps_framework_delivery(self):
        event = Event("other")
        event.result = SimpleNamespace(chain=[natural.Plain("a。"), natural.Plain("b。")])
        await self.plugin().shape_bubbles(event)
        self.assertEqual(len(event.result.chain), 2)

    def test_punctuation_splitter_is_disabled(self):
        cfg = json.loads((ROOT / "data/cmd_config.json").read_text(encoding="utf-8"))
        self.assertEqual(cfg["platform_settings"]["segmented_reply"]["words_count_threshold"], 0)
        copy_cfg = copy.deepcopy(cfg)
        tuning.tune_core(copy_cfg)
        self.assertEqual(copy_cfg["platform_settings"]["segmented_reply"]["words_count_threshold"], 0)

    async def test_query_policy_asks_for_conclusion_first(self):
        plugin = self.plugin()
        req = SimpleNamespace(func_tool=None, prompt="搜一下今天的 A 股行情", system_prompt="")
        await plugin.prepare_chat(Event(), req)
        self.assertIn("intent=query", req.system_prompt)
        self.assertIn("web_search", req.system_prompt)
        self.assertIn("最后一句停在结论上", req.system_prompt)


class PurePolicyTests(unittest.TestCase):
    def test_text_parts_preserve_question_and_time(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "question"}, {"type": "text", "text": "time"}]}]
        proxy.flatten_content(messages)
        self.assertEqual(messages[0]["content"], "question\ntime")

    def test_images_are_preserved(self):
        messages = [{"role": "user", "content": [{"type": "text", "text": "what?"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}]
        original = copy.deepcopy(messages)
        proxy.flatten_content(messages)
        self.assertEqual(messages, original)

    def test_reasoning_alias_preserves_text_tools_and_stream_protocol(self):
        message={'role':'assistant','content':'10','reasoning':'calculation','tool_calls':[]}
        result=proxy.normalize_reasoning({'choices':[{'message':message.copy()}]})
        self.assertEqual(result['choices'][0]['message']['reasoning_content'],'calculation')
        self.assertEqual(result['choices'][0]['message']['content'],'10')
        line=b'data: {"choices":[{"delta":{"reasoning":"calc"}}]}\n'
        parsed=json.loads(proxy.normalize_sse_line(line)[5:])
        self.assertEqual(parsed['choices'][0]['delta']['reasoning_content'],'calc')
        self.assertEqual(proxy.normalize_sse_line(b'data: [DONE]\n'),b'data: [DONE]\n')

    def test_budgets_are_scoped_and_respect_smaller_limits(self):
        self.assertNotIn("max_tokens", proxy.prepare_payload({"messages": [{"role": "system", "content": "extract memories"}]}))
        for mode, expected in [("brief", 384), ("expanded", 2048)]:
            payload = {"messages": [{"role": "system", "content": proxy.POLICY_MARKER + "\nresponse_budget=" + mode}], "max_tokens": 4096}
            self.assertEqual(proxy.prepare_payload(payload)["max_tokens"], expected)
            payload["max_tokens"] = 100
            self.assertEqual(proxy.prepare_payload(payload)["max_tokens"], 100)

    def test_explicit_thinking_uses_formal_control_and_larger_budget(self):
        payload={"messages":[{"role":"system","content":proxy.POLICY_MARKER+"\nresponse_budget=brief\nthinking=on"}]}
        result=proxy.prepare_payload(payload)
        self.assertEqual(result["reasoning_effort"],"high")
        self.assertEqual(result["max_tokens"],2048)
        brief=proxy.prepare_payload({"messages":[{"role":"system","content":proxy.POLICY_MARKER+"\nresponse_budget=brief\nthinking=off"}]})
        self.assertEqual(brief["reasoning_effort"],"none")

    def test_group_and_transport_settings_preserved(self):
        cfg = json.loads((ROOT / "data/cmd_config.json").read_text(encoding="utf-8"))
        before = copy.deepcopy(cfg)
        tuning.tune_core(cfg)
        for key in ["platform", "provider_ltm_settings", "wake_prefix"]:
            self.assertEqual(cfg[key], before[key])
        for key in ["unique_session", "rate_limit", "reply_with_mention", "reply_with_quote"]:
            self.assertEqual(cfg["platform_settings"][key], before["platform_settings"][key])

    def test_provider_keeps_tool_use_modality(self):
        """Omitting tool_use makes AstrBot drop every tool before the request."""
        profiles = json.loads((ROOT / "config/model_profiles.json").read_text(encoding="utf-8"))
        for name, profile in profiles.items():
            self.assertIn("tool_use", profile["modalities"], f"profile {name} would lose all tool calls")
        cfg = json.loads((ROOT / "data/cmd_config.json").read_text(encoding="utf-8"))
        provider = next(p for p in cfg["provider"] if p["id"] == "ollama_local")
        self.assertIn("tool_use", provider["modalities"], "live provider would lose all tool calls")
        import sqlite3
        with sqlite3.connect(str(ROOT / "data/data_v4.db")) as db:
            row = db.execute("select tools from personas where persona_id=?", ("千问娘",)).fetchone()
        self.assertTrue(row and json.loads(row[0]), "persona should still declare its tools")

    def test_config_reapply_accepts_astrbot_utf8_bom(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"settings.json"
            path.write_text('{"value":1}',encoding="utf-8-sig")
            with patch.object(tuning,"ROOT",Path(temp)):
                tuning.update_json("settings.json",lambda cfg:cfg.update(value=2))
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")),{"value":2})

    def test_article_excludes_navigation_and_decodes_entities(self):
        page = web.extract_page('<title>News</title><nav>MENU</nav><main><article><p>A &amp; B</p><p>release 2</p></article><aside>AD</aside></main><footer>END</footer>')
        self.assertEqual(page["title"], "News")
        self.assertEqual(page["text"], "A & B\nrelease 2")


class ProxyIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.release = threading.Event()
        self.started = threading.Event()
        self.attempts = []          # 上游实际收到的请求，用来数重试次数
        self.overflow_failures = 0  # 前几次返回"上下文溢出"
        owner = self

        class Upstream(BaseHTTPRequestHandler):
            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.attempts.append(payload)
                if payload.get("fail"):
                    self.send_response(503); self.end_headers(); self.wfile.write(b'{"error":"busy"}'); return
                if payload.get("broken"):
                    body = b'{"error":{"message":"model crashed"}}'
                    self.send_response(500); self.send_header("Content-Type","application/json")
                    self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
                if payload.get("overflow") and len(owner.attempts) <= owner.overflow_failures:
                    body = b'{"error":{"message":"Jinja Exception: No user query found in messages."}}'
                    self.send_response(500); self.send_header("Content-Type","application/json")
                    self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
                if payload.get("stream"):
                    self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
                    self.wfile.write(b'data: {"delta":"first"}\n\n'); self.wfile.flush(); owner.started.set()
                    owner.release.wait(3)
                    self.wfile.write(b'data: [DONE]\n\n'); self.wfile.flush()
                else:
                    self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
                    self.wfile.write(json.dumps(payload).encode())

            def log_message(self, *args): pass

        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), proxy.Handler)
        self.patches = [patch.object(proxy, "TARGET", f"http://127.0.0.1:{self.upstream.server_port}"), patch.object(proxy, "ROOT", Path(self.temp.name))]
        for p in self.patches: p.start()
        for server in [self.upstream, self.server]:
            threading.Thread(target=server.serve_forever, kwargs={"poll_interval":0.02}, daemon=True).start()

    def tearDown(self):
        self.release.set()
        for server in [self.server, self.upstream]: server.shutdown(); server.server_close()
        for p in self.patches: p.stop()
        self.temp.cleanup()

    def request(self, payload):
        connection = HTTPConnection("127.0.0.1", self.server.server_port, timeout=3)
        connection.request("POST", "/v1/chat/completions", json.dumps(payload), {"Content-Type":"application/json"})
        return connection, connection.getresponse()

    def test_upstream_http_error_is_forwarded(self):
        connection, response = self.request({"fail":True})
        self.assertEqual(response.status, 503)
        self.assertEqual(json.loads(response.read()), {"error":"busy"})
        connection.close()

    def test_stream_chunk_arrives_before_upstream_finishes(self):
        connection, response = self.request({"stream":True})
        self.assertIn(b'first', response.readline())
        self.assertFalse(self.release.is_set())
        self.release.set()
        self.assertIn(b'[DONE]', response.read())
        connection.close()

    def test_effective_request_preserves_text_and_chat_budget(self):
        connection, response = self.request({"messages":[{"role":"system","content":proxy.POLICY_MARKER+"\nresponse_budget=brief"},{"role":"user","content":[{"type":"text","text":"hello"}]}]})
        data = json.loads(response.read())
        self.assertEqual(data["messages"][-1]["content"], "hello")
        self.assertEqual(data["max_tokens"], 384)
        self.assertTrue((Path(self.temp.name)/"_last_forwarded_request.json").exists())
        connection.close()


    def test_context_overflow_drops_the_oldest_round_and_retries(self):
        self.overflow_failures = 1
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "旧问题"},
            {"role": "assistant", "content": "旧回答"},
            {"role": "user", "content": "新问题"},
        ]
        connection, response = self.request({"overflow": True, "messages": messages})
        self.assertEqual(response.status, 200)
        forwarded = json.loads(response.read())["messages"]
        self.assertEqual([m["role"] for m in forwarded], ["system", "user"])
        self.assertEqual(forwarded[1]["content"], "新问题")
        self.assertEqual(len(self.attempts), 2)
        connection.close()

    def test_overflow_retry_is_bounded(self):
        # 丢轮次的三级都适用（有 10 轮），截短的两级不适用（没有 tool 消息），
        # 所以到此为止，不会一直重试下去。
        self.overflow_failures = 99
        messages = [{"role": "system", "content": "sys"}]
        for index in range(10):
            messages.append({"role": "user", "content": f"q{index}"})
            messages.append({"role": "assistant", "content": f"a{index}"})
        connection, response = self.request({"overflow": True, "messages": messages})
        self.assertEqual(response.status, 500)
        self.assertLessEqual(len(self.attempts), proxy.MAX_CONTEXT_RETRIES + 1)
        self.assertGreaterEqual(len(self.attempts), 2)
        connection.close()

    def test_huge_tool_result_is_shrunk_when_no_round_can_be_dropped(self):
        # 实测过的真实形态：只有一条 user，末尾是一条 2 万字的工具结果。
        # 丢轮次一级都走不通，必须靠截短才救得回来。
        self.overflow_failures = 1
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "今天A股高开还是低开"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "搜" * 20000},
        ]
        connection, response = self.request({"overflow": True, "messages": messages})
        self.assertEqual(response.status, 200)
        forwarded = json.loads(response.read())["messages"]
        self.assertEqual([m["role"] for m in forwarded], ["system", "user", "assistant", "tool"])
        self.assertLess(len(forwarded[-1]["content"]), 7000)
        self.assertIn("已截断", forwarded[-1]["content"])
        self.assertEqual(len(self.attempts), 2)
        connection.close()

    def test_dropped_round_never_orphans_a_tool_message(self):
        # 按条数切会把 assistant.tool_calls 和它的 tool 结果拆开，上游会直接 400。
        self.overflow_failures = 1
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "旧问题"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "tool_call_id": "c1", "content": "搜索结果"},
            {"role": "assistant", "content": "旧回答"},
            {"role": "user", "content": "新问题"},
        ]
        connection, response = self.request({"overflow": True, "messages": messages})
        forwarded = json.loads(response.read())["messages"]
        self.assertEqual([m["role"] for m in forwarded], ["system", "user"])
        connection.close()

    def test_unrelated_upstream_500_is_not_retried(self):
        connection, response = self.request({"broken": True, "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(response.status, 500)
        self.assertEqual(json.loads(response.read()), {"error": {"message": "model crashed"}})
        self.assertEqual(len(self.attempts), 1)
        connection.close()

    def test_overflow_that_cannot_be_trimmed_is_forwarded(self):
        self.overflow_failures = 99
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "只有一个问题"}]
        connection, response = self.request({"overflow": True, "messages": messages})
        self.assertEqual(response.status, 500)
        self.assertEqual(len(self.attempts), 1)
        connection.close()


if __name__ == "__main__": unittest.main()
