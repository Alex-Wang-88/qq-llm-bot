# Ollama 部署准备（4B / 9B LoRA 产物）

原始 Transformers 基座、LoRA adapter、合并权重和 GGUF 推理模型是不同产物。原 `qwen3.5:9b-qq` 保持为回退，不覆盖、删除或替换。

先确认本机实验报告中训练与 adapter 重载推理通过，再考虑部署。4B 的 adapter 只适配同一 revision 的 Qwen3.5-4B，不能挂到 9B 上；反之亦然。保留基础模型 revision、adapter 配置、数据哈希、评测输出和依赖版本。

## 已执行的步骤与实测结果（4B / experimental-v2 / step 30）

以下四步已在桌面机完成，产物在 `.training-local/`：

1. **合并**：`merge_adapter.py` 输出 `runs/4b-bf16-sft-v2/merged-bf16/`，723 张量、8.46 GiB，相对 logits L2 误差 **0.0105**（阈值 0.03），耗时 22.9 秒。脚本不修改原始权重或 adapter。
2. **转 f16 GGUF**：用 `ggml-org/llama.cpp` **b11900** 的 `convert_hf_to_gguf.py`，426 张量、7.85 GiB。
3. **量化**：`llama-quantize` Q4_K_M → **2.52 GiB / 5.13 BPW**，36.3 秒。
4. **加载推理**：Vulkan 后端实测 **生成 146 tok/s**，能正常加载并给出结构化回答。

### 转换必须加 `--no-mtp`，否则 GGUF 加载失败

Qwen3.5 的 `config.json` 写着 `text_config.mtp_num_hidden_layers: 1`，原始权重含 15 个 `mtp.*` 张量，但 transformers 5.4.0 的 `Qwen3_5PreTrainedModel` 有 `_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]`，**加载时会丢弃它们**。合并后的权重因此没有 MTP，而 config 仍声明 1 层，转换器会按 33 层生成而实际只有 32 层。

判据：转换后 GGUF 元数据 `qwen35.block_count` 应为 **32**。若为 33，模型会加载失败。

### 视觉塔被排除

该权重是 `Qwen3_5ForConditionalGeneration`，738 张量里 297 个属视觉塔。转换只输出文本部分，GGUF 元数据 `modalities: text`。**导出的模型不能读图**；要支持图片输入需另外转 mmproj，本轮未做。

### 推理必须关闭 thinking

实测（llama.cpp b11900）在 `enable_thinking` 未关闭时，48 个 token 全部花在思考过程上，没有正文。llama.cpp 下的写法是 `--chat-template-kwargs '{"enable_thinking": false}'`。

**Ollama 下关闭 thinking 的写法随版本变化**（`PARAMETER think false`、`PARAMETER chat_template_kwargs {...}`、请求体传 `think: false` 均有说法，社区实测结果互相矛盾），本项目**未在真实 Ollama 上验证**哪一种生效。部署后请用判据确认：发一句普通问候，若输出里出现 `[Start thinking]` 或 `<think>` 即未生效。

## 部署到目标机（含 8 GB 笔记本）

量化后的模型是 **2.52 GiB**，8 GB 显存足够驻留（Q8_0 约 4.5 GiB 也可行）。注意区分三件事的硬件需求：

| 环节 | 依赖 | 8 GB 笔记本 |
| --- | --- | --- |
| 合并 / 转换 / 量化 | 纯 CPU，内存 ≥24 GiB、磁盘 ≥30 GiB | 可做但不需要 GPU；bf16 全量 8.68 GiB 显存装不下 |
| 训练 | 4B 全 GPU 峰值 10.30 GiB | 需把层卸载到 CPU，慢很多 |
| 推理 | 权重 2.5 GiB + KV | 宽裕，这才是目标机该干的活 |

推荐做法：在已有环境完成导出，只把 2.52 GiB 的 GGUF 拷到推理机，避免重建 CUDA 环境与重下 8.7 GiB 原始权重。

### 产物已发布到 Release，目标机不用再传文件

Q4_K_M 已发布在 [`qwen35-4b-sft-v2`](https://github.com/Alex-Wang-88/qq-llm-bot/releases/tag/qwen35-4b-sft-v2)。GitHub 单个 asset 上限 2 GiB，而模型是 2.52 GiB（2 708 804 096 字节），因此切成 `part-00`（1400 MiB）+ `part-01`，按序拼回来与原始文件逐字节相同。

目标机上一条命令完成下载、拼接、校验：

```powershell
curl.exe -L -o fetch_model_release.ps1 https://github.com/Alex-Wang-88/qq-llm-bot/releases/download/qwen35-4b-sft-v2/fetch_model_release.ps1
powershell -ExecutionPolicy Bypass -File .\fetch_model_release.ps1 -OutDir D:\models
```

**网络这一环容易卡住**：Release 资产从 `github.com` 302 跳到 `objects.githubusercontent.com`，而 `curl` 不读 Windows 系统代理 —— VPN 客户端通常只改系统代理，于是「梯子开着 curl 仍报 `schannel: failed to receive handshake`」。本机实测：环境里的另一个代理（58176）直连 40 秒 0 字节，镜像 0.1–4.3 MB/s，而系统代理（15732）**10.17 MB/s**，2.7 GB 约 4 分半。

脚本因此按「系统代理 → 直连 → `gh-proxy.com` 镜像」的顺序自动挑路，VPN 开着时不用加任何参数。手工指定用 `-Proxy 127.0.0.1:15732`，禁用代理用 `-NoProxy`，GitHub 完全不通时 `-MirrorFirst` 可省掉每轮 12 秒连接超时。

脚本先核对每个分片的 SHA256，按序拼接后再核对拼接结果的 SHA256（`679b0f0a…1c0327b2`）与字节数，任何一项对不上就报错并提示删掉重下；`-Import` 可在校验通过后顺手 `ollama create`。分片哈希、合并态哈希和手动拼接命令见 Release 说明页。

**注意 1**：仓库里其他 `.ps1` 是无 BOM 的 UTF-8，PowerShell 5.1 在非 UTF-8 系统区域设置下会按 ANSI 解码、把中文串搞坏。`fetch_model_release.ps1` 因此带 BOM，改动它时别把 BOM 去掉。

**注意 2**：脚本里调 curl 时临时把 `$ErrorActionPreference` 降到 `Continue`。curl 的进度条写 stderr，而 PowerShell 5.1 在 `Stop` 下会把任何 stderr 输出升级成终止性 `NativeCommandError` —— 这个坑和 `run.ps1` 被管道打断是同一类。成败一律看退出码，不看 stderr。

5. 创建新名称，例如 `qwen3.5:4b-qq-v2`。仓库内已提供 `config/Modelfile.qianwen35-4b`，把 `FROM` 改成实际 GGUF 路径后：

```powershell
ollama create qwen3.5:4b-qq-v2 -f config/Modelfile.qianwen35-4b
ollama run qwen3.5:4b-qq-v2
```

创建后先使用本机隔离评测，再在 AstrBot 临时选择新名称与相同千问娘人设。仅导出语言 GGUF 时，provider 声明 `text,tool_use`；只有另行导出兼容视觉 projector、并实际通过图片输入回归后，才添加 `image`。检查实际请求中工具存在、工具轮无播报、工具结果后正确回答。不能照抄原多模态 profile 后就宣称图片仍有效；本轮语言微调没有做视觉回归。

有回归时选回原 `qwen3.5:9b-qq`，不用恢复旧 SQLite。切换前保留当前 provider 配置，不能用原电脑的整库覆盖新聊天记录。

## 未验证项

- **Ollama 导入与运行**：本机未安装 Ollama，`ollama create` 与 thinking 关闭方式均未实测。已确认的只是同一份 GGUF 能被 llama.cpp b11900 加载并推理。
- **笔记本 5060 实测**：未在目标机上跑过，速度与显存占用是推断。
- **真实 QQ 会话**：全部验证停留在离线，端到端流式渲染、气泡切分与输出守卫的交互未测。
- **9B 在 v2 数据上的重训**：按决定搁置。

资料：[Ollama 导入](https://docs.ollama.com/import)、[llama.cpp 转换脚本](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py)、[Qwen3.5 官方模型](https://huggingface.co/Qwen/Qwen3.5-9B)。
