# qq-llm-bot

把本机 Ollama 上的 `qwen3.5:9b` 接成 QQ 官方机器人。这个仓库不是"跑通一个 LLM 机器人"的模板，
它记录的是把一个小模型调到**像个真人在 QQ 上说话**的过程：压住输出预算、管住气泡数量、
管住它别把"我这就去查"当成回答发出去，以及顺手接了个只读查本机 Codex 会话的入口。

两台机器，职责分得很清楚：

| 机器 | 配置 | 干什么 |
| --- | --- | --- |
| **训练机**（桌面） | AMD Ryzen 7 7800X3D（8 核 16 线程，Windows 只暴露 8 个逻辑核）/ 47 GB / Radeon RX 9070 XT **16 GB**（gfx1201，ROCm 7.2.1） | 数据构建、LoRA 训练、合并、GGUF 转换与量化、A/B/C 评测 |
| **运行机**（笔记本） | AMD Ryzen AI 7 H 350（8 核 16 线程）/ 31 GB / RTX 5060 Laptop **8 GB** + 核显 Radeon 860M | 常驻 Ollama、AstrBot、QQ 接入，单路推理 |

训练机的完整实测记录在 `training/EXPERIMENT_REPORT.md`。运行机显存只有 4 GB，
所以只有量化到 Q4_K_M 的 4B（2.52 GiB）才装得舒服；9B 的 6.6 GB 必然大量卸载到 CPU。

## 结构

| 路径 | 作用 |
| --- | --- |
| `ollama_fix_proxy.py` | 8899 端口上的兼容代理。拍平 text parts、`reasoning`→`reasoning_content`、按 `response_budget` 限 token（brief 384 / expanded 2048）、串行转发、出口净化、上下文溢出阶梯补救 |
| `assistant_output_guard.py` | 出口净化：剥掉模型自己续写的 `user`/`assistant` 轮与 `<think>` 块，流式与非流式结果逐字一致 |
| `assistant_dialogue.py` | 廉价意图分类（chat / query / math / computer / harness），只影响策略，不改权限 |
| `assistant_calc.py` | 算术求值：`ast` 节点白名单 + `Fraction`，不用 `eval`。纯算式由代码直接回，句中算式只注入核验结果 |
| `assistant_security.py` | 出站文本脱敏 |
| `assistant_harness/` | Codex App Server 的 JSONL 传输层与能力适配器，只读查询会话与额度 |
| `data/plugins/astrbot_plugin_chat_natural/` | 聊天策略：工具白名单、历史清理、气泡形状、表情节奏 |
| `data/plugins/astrbot_plugin_harness_manager/` | `/助手` 命令：查会话、读结果、查额度 |
| `data/plugins/astrbot_plugin_searxng_search/` | 搜索插件（正文提取部分被改过） |
| `config/` | 人设、Ollama Modelfile、模型档位、会话闲置阈值（`session_policy.json`） |
| `scripts/` | 启动、预检、离线探针 |
| `training/` | 离线微调工作区：数据构建、训练、评测、adapter 合并、GGUF 导出，入口 `training/README.md` |
| `tests/` | 回归测试，149 项 |

## 修过的几个坑

这个仓库真正值钱的是这几条，都是实测出来的，不是猜的：

1. **`provider.modalities` 少写 `tool_use`，工具会静默全丢。** AstrBot 在
   `_func_tool_for_provider()` 里发现能力声明不含 `tool_use` 就直接返回 `None`，请求里连 `tools`
   字段都没有，日志只有 debug 级别一行。表现是模型嘴上说"我去搜"，但从来没有工具调用。
2. **历史里的 `&&search&&` 会把工具调用饿死。** 模型把表情语法错当工具语法，写进历史后每轮复读。
   实测带标记时工具调用 1/3 次，剥离后 3/3。
3. **工具调用那一轮的文本会单独变成一条消息。** `tool_loop_agent_runner.py` 每轮都 yield 一个
   `llm_result`，带不带工具调用都一样，所以"好的，我这就去查一下。"会先于工具结果发出去。
4. **`Start-Process` 遇上大小写重复的环境变量会整脚崩掉。** `HTTPS_PROXY` 与 `https_proxy`
   同时存在时，PowerShell 5.1 复制环境块抛 `ArgumentException`，脚本在写任何日志之前就退出。
5. **默认的"分段回复"按句号切。** 判定是 `len(text) > 阈值` 时**不分段**，很容易被误读成相反行为，
   实际效果是短回复被切碎、长回复整条发。本仓库把阈值设 0，气泡数改由插件按内容结构决定。
6. **`/v1/chat/completions` 不认 `think: false`。** Ollama 的原生 `/api/chat` 关得掉，
   OpenAI 兼容层关不掉——`think`、`chat_template_kwargs.enable_thinking`、`options.think`
   实测全部无效，一句"今天天气不错"烧掉 1238–2399 token 的思考。`/v1` 上唯一有效的是
   `reasoning_effort: "none"`（7 token）。AstrBot 的 provider 走 `/v1`，所以这项必须写进
   `config/model_profiles.json` 的 generation，否则响应预算全被思考吃掉。
7. **`num_ctx` 给小了，工具一调就 500。** 上下文装不下 system（3176 字）+ 工具 schema +
   多轮工具结果时，Ollama 会截断，Qwen 的 chat template 找不到 user 消息，直接抛
   `Jinja Exception: No user query found in messages`——报错看起来像工具坏了，实际是窗口太小。
   4B 在 4 GB 显存上 8192 仍能全量驻留（3.19 GB），别为了省显存压到 4096。
8. **丢旧轮次治不了这个 500，得截短工具结果。** Ollama 截断是从前面丢、保住尾部，而爆窗口的
   往往是最后那条大工具结果——丢前面的会话轮次不改变尾部，怎么丢都还是 500。代理里因此是
   阶梯补救：`drop 1/2/4` → `shrink 6000/3000`，某一级不适用就跳到下一级。实测
   「1 条 user + 4 万字工具结果」这种形态，只能靠截短救回来。
9. **让回复变长，会带出角色边界失控。** 微调后的 4B 不再说完一句就停，但也开始自己续写
   `user`/`assistant` 行和空 `<think>` 块，一路循环到 token 上限——有 26 个评测用例里 1 条这样。
   训练集里这两个东西一个都没有，所以这是基座的对话先验，不是重训能收敛掉的。
   `assistant_output_guard.py` 在代理出口做兜底：整行只有角色名就截断、剥掉 think 块、
   流式输出与整段输出结果逐字一致（否则流式会漏）。78 条真实评测输出回放，只改动该改的 2 条，
   其余 76 条一字未动。
10. **`on_llm_request` 里光 `set_result` 是发不出去的。** `internal.py` 拿到 hook 结果后直接
   `if await call_event_hook(...): return`，而它的第一个 `yield` 在这之后——所以事件被
   `stop_event()` 之后 ProcessStage 一个 yield 都不产出，respond 阶段压根不执行，消息静默消失。
   想短路（例如纯算式不经模型）必须自己 `await event.send(MessageChain([Plain(text)]))`，
   然后再 `stop_event()`。
11. **别指望模型算数。** 4B 对 `23 × 7 - 18` 连错三次（64 / 81 / 169），9B 给 141，训练报告里
    记着同一个失败模式（"跳过步骤直接猜"）。这是能力问题，改提示词无效——算术交给
    `assistant_calc.py`，模型只负责把结果说成人话。

细节和证据都在 `开发进度.md`。

## 启动

前置：AstrBot 4.29（`uv tool install astrbot`）、Ollama、searxng（docker，用
`searxng/settings.example.yml` 生成 `settings.yml`）、Codex CLI（可选，只影响 `/助手`）。

```bat
start_all.bat
```

它做四件事：资源预检 → 确保 Ollama 在跑 → 起 8899 代理 → 起 AstrBot 并等 6185 就绪。
每一步都写进 `data/logs/startup.log`，失败会打印退出码和日志末几行。

模型要自己建一次：

```bash
ollama create qwen3.5:9b-qq -f config/Modelfile.qianwen35
```

## 测试

```bash
python -m unittest tests.test_harness tests.test_chat_tuning tests.test_dialogue tests.test_security tests.test_output_guard
```

`test_chat_tuning` 需要装了 AstrBot 的环境（`training/.venv-bot-tests/` 里有一份 4.29.3）。
其中 1 项失败、3 项错误是环境问题而非代码问题：表情包分类数据没入库、真实运行配置没上传。

离线探针（不发 QQ，可反复跑）：

| 脚本 | 用途 |
| --- | --- |
| `scripts/probe_toolcall.py` | 重放真实请求，看工具调用率 |
| `scripts/probe_announcement.py` | 统计"工具调用那一轮还写了字"的比例 |
| `scripts/probe_codex_harness.py` | Codex App Server 握手 + 会话 + 额度 |
| `scripts/probe_harness_command.py` | 把 `/助手` 三个命令的回复原样渲染出来 |

## 微调的 4B 模型在哪

`training/` 训出来的 4B（Qwen3.5-4B + 人格 SFT v2，Q4_K_M，2.52 GiB）发布在
[`qwen35-4b-sft-v2`](https://github.com/Alex-Wang-88/qq-llm-bot/releases/tag/qwen35-4b-sft-v2)。
GitHub 单个文件上限 2 GiB，所以切成两片，下载、拼接、校验一步做完：

```powershell
curl.exe -L -o fetch_model_release.ps1 https://github.com/Alex-Wang-88/qq-llm-bot/releases/download/qwen35-4b-sft-v2/fetch_model_release.ps1
powershell -ExecutionPolicy Bypass -File .\fetch_model_release.ps1
```

Release 资产会跳到 `objects.githubusercontent.com`，而 `curl` 不读 Windows 系统代理，所以「梯子开着 curl 仍握手失败」很常见。脚本按**系统代理 → 直连 → `gh-proxy.com` 镜像**自动挑路：走系统代理实测 10.17 MB/s（2.7 GB 约 4 分半），镜像 0.1–4.3 MB/s 波动很大。手工指定用 `-Proxy`，强制镜像用 `-MirrorFirst`。最后仍靠 SHA256 校验拼接结果。

相对基座最明确的变化是**诚实性**：问"我发的照片里窗帘什么颜色"，基座编"深灰色的，带点暗纹，像百叶帘"，
微调后回"我这边没有这张图，颜色看不出来"。代价是闲聊变短变生硬，长度主要靠上层人设撑。
这个产物**不能读图**——导出时只保留了语言塔。

## 仓库里没有的东西

`data/` 只保留三个插件的源码，其余（机器人 appid/secret、会话库、日志、表情包、
面板密码哈希）都不入库。`searxng/settings.yml` 和角色形象图也不入库。
`开发进度.md` 里会出现本机绝对路径，介意的话自己裁掉。

个人自用项目，不提供支持。
