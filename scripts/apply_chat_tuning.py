"""Apply the reviewed local profile without touching transport or group settings."""

import json
from pathlib import Path
import sqlite3

ROOT = Path(__file__).resolve().parents[1]


def update_json(relative, edit):
    path = ROOT / relative
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    edit(value)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=4) + "\n", encoding="utf-8")
    temp.replace(path)


def tune_core(cfg):
    # Route/session identity/history/active-reply configuration is left intact.
    cfg["provider_settings"]["proactive_capability"]["add_cron_tools"] = False
    cfg["platform_settings"]["segmented_reply"]["interval"] = "0.15,0.45"
    cfg["platform_settings"]["segmented_reply"]["regex"] = r".+"
    # Threshold 0 means AstrBot never cuts a reply by punctuation. Bubble count
    # is owned by astrbot_plugin_chat_natural (one message, two when long), so a
    # fact answer no longer arrives as several sentence fragments.
    cfg["platform_settings"]["segmented_reply"]["words_count_threshold"] = 0
    agent = cfg["agent_runner"]["config"]
    agent["model"]["request_max_retries"] = 2
    agent["misc"]["max_steps"] = 3
    agent["misc"]["tool_call_timeout"] = 40
    agent["compression"].update(max_turns=20, keep_recent_ratio=0.35, fallback_max_tokens=8192)
    for provider in cfg["provider"]:
        if provider["id"] == "ollama_local":
            provider["model"] = provider.get("model") or "qwen3:8b-qq"
            provider["max_context_tokens"] = 8192
            body = provider.setdefault("custom_extra_body", {})
            body.update(temperature=0.7, top_p=0.8, max_tokens=2048)


def tune_memes(cfg):
    prompt = cfg["generation"]["prompt"]
    prompt["head"] = (
        "[表情图片]\n有贴切表情时可配图。发送时从下列分类中选键，写成 &&分类键&&，"
        "标记独占一行，放在正文末尾；也可只发表情。用户要求纯文字时不配图。\n分类：\n"
    )
    prompt["tail"] = (
        "\n分类及描述是选图资料，不是对话指令；不要在正文展示、解释或复述它们。"
        "通常选一张，不合适就不发，不重复凑数；用户明确指定数量时按要求选择。"
        "按回复者想表达的态度选图，区分安慰对方与表达自己的低落。"
        "图片是随机选取，分类不能证明人物、配字或画面；未看见的细节不猜，不声称图片已送达。\n"
    )
    prompt["quantity_guidance_enabled"] = False
    prompt["category_example_enabled"] = False
    prompt["reply_example_enabled"] = False


def tune_memory(cfg):
    cfg.update(background_extraction=True, extraction_interval=6, max_memories_per_inject=3)


def main():
    active=ROOT/"data/harness/model-profile.json"
    profile=None
    if active.exists():
        choice=json.loads(active.read_text(encoding="utf-8"))["active"]
        profile=json.loads((ROOT/"config/model_profiles.json").read_text(encoding="utf-8"))[choice]
    def edit_core(cfg):
        tune_core(cfg)
        if profile:
            for provider in cfg["provider"]:
                if provider["id"]=="ollama_local":
                    provider.update(model=profile["model"],modalities=profile["modalities"],custom_extra_body=profile["generation"],max_context_tokens=profile["context_tokens"])
    update_json("data/cmd_config.json", edit_core)
    update_json("data/config/astrbot_plugin_meme_manager_config.json", tune_memes)
    update_json("data/config/astrbot_plugin_simple_long_memory_config.json", tune_memory)
    persona = (ROOT / (profile["persona_file"] if profile else "config/qianwen_persona.txt")).read_text(encoding="utf-8").strip()
    with sqlite3.connect(str(ROOT / "data/data_v4.db"), timeout=5) as db:
        result = db.execute(
            "UPDATE personas SET system_prompt=?,tools=?,skills=?,updated_at=CURRENT_TIMESTAMP WHERE persona_id=?",
            (persona, json.dumps(["web_search", "fetch_web_page", "harness_query"]), "[]", "千问娘"),
        )
        if result.rowcount != 1:
            raise RuntimeError("Expected exactly one 千问娘 persona")
    print("Applied chat profile; transport and group configuration preserved.")


if __name__ == "__main__":
    main()
