"""Isolated model smoke test. Does not send QQ messages or write bot memories."""
import asyncio
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from astrbot.core.agent.tool import FunctionTool
from astrbot.core.astr_main_agent import _ensure_persona_and_skills
from astrbot.core.astr_main_agent_resources import LLM_SAFETY_MODE_SYSTEM_PROMPT, TOOL_CALL_PROMPT

spec = importlib.util.spec_from_file_location("natural_chat_verification", ROOT / "data/plugins/astrbot_plugin_chat_natural/main.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
POLICY = None


class Event:
    unified_msg_origin = "local_validation:FriendMessage:fixture"
    def __init__(self): self.extra = {}
    def get_extra(self, key): return self.extra.get(key)
    def set_extra(self, key, value): self.extra[key] = value
    def get_platform_name(self): return "qq_official"


async def build_request(prompt):
    global POLICY
    cfg = json.loads((ROOT / "data/cmd_config.json").read_text(encoding="utf-8-sig"))
    with sqlite3.connect(str(ROOT / "data/data_v4.db")) as db:
        text, tools, skills = db.execute("select system_prompt,tools,skills from personas where persona_id=?", ("千问娘",)).fetchone()
    persona = {"prompt": text, "tools": json.loads(tools), "skills": json.loads(skills), "custom_error_message":None}
    functions = [FunctionTool(name=n, description=d, parameters={"type":"object","properties":{p:{"type":"string"}},"required":[p]})
                 for n,d,p in [("web_search","联网搜索实时信息","query"),("fetch_web_page","读取网页正文","url")]]
    manager = SimpleNamespace(get_func=lambda name: next((f for f in functions if f.name == name),None))
    context = SimpleNamespace(
        persona_manager=SimpleNamespace(resolve_selected_persona=AsyncMock(return_value=("千问娘",persona,None,False))),
        get_llm_tool_manager=lambda:manager, get_config=lambda *a:cfg, subagent_orchestrator=None,
    )
    req = SimpleNamespace(system_prompt=LLM_SAFETY_MODE_SYSTEM_PROMPT+"\n\n", prompt=prompt, contexts=[],
                          conversation=SimpleNamespace(persona_id="千问娘"),func_tool=None)
    event = Event()
    await _ensure_persona_and_skills(req, cfg["provider_settings"], context, event)
    assert "## Skills" not in req.system_prompt
    req.system_prompt += "\n"+TOOL_CALL_PROMPT+"\n"
    meme = json.loads((ROOT / "data/config/astrbot_plugin_meme_manager_config.json").read_text(encoding="utf-8-sig"))["generation"]["prompt"]
    categories = json.loads((ROOT / "data/plugin_data/meme_manager/packs/seio-stickers/memes_data.json").read_text(encoding="utf-8"))
    req.system_prompt += "\n" + meme["head"] + "\n".join(f"{k} - {v}" for k,v in categories.items()) + meme["tail"]
    if POLICY is None:
        POLICY = module.NaturalChatPlugin(context)
    await POLICY.prepare_chat(event, req)
    req.test_event = event
    return req


async def delivered(req, reply):
    response=SimpleNamespace(completion_text=reply.get("content") or "")
    await POLICY.note_meme(req.test_event,response)
    await POLICY.record_sent(req.test_event)
    reply=dict(reply,content=response.completion_text)
    return reply


def completion(system, history, tools=None):
    cfg=json.loads((ROOT/'data/cmd_config.json').read_text(encoding='utf-8-sig'))
    provider=next(p for p in cfg['provider'] if p['id']=='ollama_local')
    payload = {"model":provider['model'],"messages":[{"role":"system","content":system},*history],
               "stream":False,"reasoning_effort":"none","seed":42,**provider.get('custom_extra_body',{})}
    if tools: payload["tools"] = tools
    started = time.monotonic()
    request = urllib.request.Request("http://127.0.0.1:8899/v1/chat/completions",data=json.dumps(payload,ensure_ascii=False).encode(),headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(request,timeout=120) as response: data=json.load(response)
    elapsed=round(time.monotonic()-started,2)
    choice=data["choices"][0]
    if choice.get("finish_reason") == "length": raise RuntimeError("Output was truncated")
    return choice["message"], elapsed


async def main():
    results=[]
    prompt="今天好累，连着开了四个会"
    req=await build_request(prompt)
    history=[{"role":"user","content":prompt}]
    reply,elapsed=completion(req.system_prompt,history)
    reply=await delivered(req,reply)
    results.append({"case":"疲惫与承接","prompt":prompt,"response":reply.get("content"),"seconds":elapsed})
    history += [reply,{"role":"user","content":"嗯，我现在只想躺会儿，不想听建议"}]
    next_req=await build_request(history[-1]["content"])
    reply,elapsed=completion(next_req.system_prompt,history)
    reply=await delivered(next_req,reply)
    results.append({"case":"接续并尊重结束","response":reply.get("content"),"seconds":elapsed})

    prompt="记住，我说的是无糖，不是少糖"
    req=await build_request(prompt)
    reply,elapsed=completion(req.system_prompt,[{"role":"user","content":"我喜欢喝咖啡"},{"role":"assistant","content":"少糖那种？"},{"role":"user","content":prompt}])
    reply=await delivered(req,reply)
    results.append({"case":"接受更正","response":reply.get("content"),"seconds":elapsed})

    prompt="帮我查一下测试软件 DemoChat 的最新版本，主要改了啥"
    req=await build_request(prompt)
    history=[{"role":"user","content":prompt}]
    tools=[{"type":"function","function":{"name":"web_search","description":"联网搜索最新信息","parameters":{"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}}}]
    tool_rounds=0;elapsed_total=0
    for _ in range(3):
        reply,elapsed=completion(req.system_prompt,history,tools);elapsed_total+=elapsed
        history.append(reply)
        if not reply.get("tool_calls"):break
        for call in reply["tool_calls"]:
            tool_rounds+=1
            # Deliberately synthetic, fixed evidence; no live news or website access.
            history.append({"role":"tool","tool_call_id":call["id"],"content":json.dumps({"source_type":"web_search","results":[{"title":"DemoChat 2.0 测试发布说明","url":"https://example.invalid/releases/2.0","snippet":"仅用于测试的虚构材料：DemoChat 2.0 修复搜索超时，改善历史消息加载。"}]},ensure_ascii=False)})
    if reply.get("tool_calls"):
        reply,elapsed=completion(req.system_prompt,history);elapsed_total+=elapsed
    reply=await delivered(req,reply)
    results.append({"case":"工具后回答原问题（固定虚构资料）","response":reply.get("content"),"tool_calls":tool_rounds,"seconds":round(elapsed_total,2)})

    prompt="请详细解释一下为什么分段回复不等于流式输出，分步骤说清楚"
    req=await build_request(prompt)
    reply,elapsed=completion(req.system_prompt,[{"role":"user","content":prompt}])
    reply=await delivered(req,reply)
    results.append({"case":"明确要求详细","response":reply.get("content"),"seconds":elapsed})
    cfg=json.loads((ROOT/'data/cmd_config.json').read_text(encoding='utf-8-sig'))
    model=next(p['model'] for p in cfg['provider'] if p['id']=='ollama_local')
    output={"model":model,"system_chars":len(req.system_prompt),"skills_injected":False,"tools":[f.name for f in req.func_tool.tools],"results":results}
    path=ROOT/"data/logs/chat-tuning-smoke.json"
    path.write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(output,ensure_ascii=False,indent=2))


if __name__=="__main__": asyncio.run(main())
