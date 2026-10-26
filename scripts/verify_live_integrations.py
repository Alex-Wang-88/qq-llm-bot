"""Real read-only integrations. No QQ sends, Harness turns or memory writes."""
import asyncio
import argparse
from datetime import datetime,timezone
import importlib.util
import json
from pathlib import Path
import sys
import urllib.request
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from assistant_security import install_log_redaction,redact
install_log_redaction()
from astrbot.core.provider.sources.openai_source import ProviderOpenAIOfficial
from astrbot.core.message.components import Image
from verify_chat_tuning import build_request,completion


def load_package(name,path):
    spec=importlib.util.spec_from_file_location(name,path/'main.py',submodule_search_locations=[str(path)])
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module


async def main():
    parser=argparse.ArgumentParser();parser.add_argument('--media-only',action='store_true');parser.add_argument('--raw-thinking-probe',action='store_true');args=parser.parse_args()
    cfg=json.loads((ROOT/'data/cmd_config.json').read_text(encoding='utf-8-sig'))
    selected=next(p for p in cfg['provider'] if p['id']=='ollama_local')
    source=next(p for p in cfg['provider_sources'] if p['id']==selected['provider_source_id'])
    provider=ProviderOpenAIOfficial({**source,**selected},cfg['provider_settings'])
    report={'collected_at':datetime.now(timezone.utc).isoformat(),'model':selected['model'],'scope':'isolated real providers; no phone receive/send','results':[]}
    search_module=load_package('live_search_verification',ROOT/'data/plugins/astrbot_plugin_searxng_search')
    harness_module=load_package('live_harness_verification',ROOT/'data/plugins/astrbot_plugin_harness_manager')
    search=search_module.SearxngSearchPlugin(None)
    harness_cfg=json.loads((ROOT/'data/config/astrbot_plugin_harness_manager_config.json').read_text(encoding='utf-8-sig'))
    harness=harness_module.HarnessManager(None,harness_cfg)
    event=SimpleNamespace(unified_msg_origin=(harness_cfg.get('allowed_private_sessions') or [''])[0])

    async def exercise(prompt,tool_name,properties,required,callback):
        result={'case':tool_name,'prompt':prompt}
        try:
            req=await build_request(prompt)
            tools=[{'type':'function','function':{'name':tool_name,'description':'必须使用此只读工具读取真实资料后回答。','parameters':{'type':'object','properties':properties,'required':required}}}]
            history=[{'role':'user','content':prompt}]
            reply,elapsed=completion(req.system_prompt,history,tools)
            history.append(reply);calls=reply.get('tool_calls') or []
            result.update(initial_seconds=elapsed,tool_calls=len(calls),actual_evidence=[])
            if not calls:raise RuntimeError('Model did not invoke required read-only provider')
            for call in calls:
                if call['function']['name']!=tool_name:raise RuntimeError('Unexpected tool name')
                args=json.loads(call['function']['arguments'])
                evidence=await callback(**args)
                result['actual_evidence'].append(evidence)
                history.append({'role':'tool','tool_call_id':call['id'],'content':evidence})
            answer,elapsed=completion(req.system_prompt,history)
            result.update(response=answer.get('content'),answer_seconds=elapsed,state='Completed')
        except Exception as exc:result.update(state='Error',error_type=type(exc).__name__,error=str(exc)[:200])
        report['results'].append(result)
        print(json.dumps(redact(result),ensure_ascii=False),flush=True)

    try:
        # Actual image component -> framework media resolver -> installed provider -> proxy -> vision model.
        if 'image' in selected.get('modalities',[]):
            result={'case':'astrbot_image_component'}
            try:
                fixture=ROOT/'data/harness/vision-fixture.png'
                image=Image.fromFileSystem(str(fixture))
                resolved=await image.convert_to_file_path()
                assembled=await provider.assemble_context('统计图形和数字',image_urls=[resolved])
                images=[p for p in assembled['content'] if p.get('type')=='image_url']
                result['image_parts']=len(images)
                result['encoded_image_reaches_provider']=len(images)==1 and images[0]['image_url']['url'].startswith('data:image/')
                original_body=provider.provider_config.get('custom_extra_body',{})
                provider.provider_config['custom_extra_body']={**original_body,'response_format':{'type':'json_object'},'max_tokens':384}
                reply=await provider.text_chat(prompt='只返回 JSON：blue_circles 蓝色圆形数量，red_squares 红色方块数量，code 图片中的数字。',image_urls=[resolved],contexts=[],system_prompt='按实际图片回答，看不清就说看不清。',request_max_retries=1)
                provider.provider_config['custom_extra_body']=original_body
                result['response']=reply.completion_text
                parsed=json.loads(reply.completion_text)
                result.update(parsed_json=parsed,passed=parsed.get('blue_circles')==3 and parsed.get('red_squares')==1 and str(parsed.get('code'))=='42')
            except Exception as exc:result.update(error_type=type(exc).__name__,error=str(exc)[:200])
            report['results'].append(result);print(json.dumps(result,ensure_ascii=False),flush=True)
        else:report['results'].append({'case':'astrbot_image_component','state':'Unsupported','reason':'Text-only profile; no image sent'})
        if not args.media_only:
            await exercise('帮我联网查一下 Ollama 的 Qwen3.5 9B 官方模型页面，只说查到的内容。','web_search',{'query':{'type':'string'}},['query'],lambda **args:search.web_search(event,**args))
            await exercise('读取这个链接，概括它实际介绍的模型和输入能力：https://ollama.com/library/qwen3.5:9b-q4_K_M','fetch_web_page',{'url':{'type':'string'}},['url'],lambda **args:search.fetch_web_page(event,**args))
            await exercise('查询 Codex 当前真实剩余额度。','harness_query',{'operation':{'type':'string','enum':['quota']},'harness':{'type':'string','enum':['codex']}},['operation','harness'],lambda **args:harness.query_tool(event,**args))
        # Formal opt-in thinking through the same installed provider and proxy.
        req=await build_request('请深度思考：一共 3 个盒子，每盒 4 个球，拿走 2 个，还剩几个？')
        reply=await provider.text_chat(prompt=req.prompt,contexts=[],system_prompt=req.system_prompt,request_max_retries=1)
        forwarded=json.loads((ROOT/'_last_forwarded_request.json').read_text(encoding='utf-8'))
        report['results'].append({'case':'formal_thinking_opt_in','control':forwarded.get('reasoning_effort'),'reasoning_chars':len(reply.reasoning_content or ''),'response':reply.completion_text})
        if args.raw_thinking_probe:
            result={'case':'formal_thinking_raw_api'}
            try:
                payload={'model':selected['model'],'messages':[{'role':'user','content':'算一下：3 盒球，每盒 4 个，拿走 2 个，还剩几个？'}],'stream':False,'reasoning_effort':'high','max_tokens':2048}
                request=urllib.request.Request('http://127.0.0.1:8899/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
                with urllib.request.urlopen(request,timeout=120) as response:raw=json.load(response)
                message=raw['choices'][0]['message']
                result.update(message_keys=list(message),reasoning_chars=len(message.get('reasoning_content') or message.get('reasoning') or ''),response=message.get('content'),finish_reason=raw['choices'][0].get('finish_reason'))
            except Exception as exc:result.update(state='Error',error_type=type(exc).__name__)
            report['results'].append(result)
    finally:
        await harness.terminate()
        await provider.client.close()
        path=ROOT/f"data/logs/live-integrations-{'qwen35' if '3.5' in selected['model'] else 'qwen3'}{'-media' if args.media_only else ''}.json"
        path.write_text(json.dumps(redact(report),ensure_ascii=False,indent=2),encoding='utf-8')
        print('SAVED',path,flush=True)


if __name__=='__main__':asyncio.run(main())
