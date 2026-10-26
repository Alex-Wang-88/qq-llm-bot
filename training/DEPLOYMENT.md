# Ollama 部署准备（4B / 9B LoRA 产物）

原始 Transformers 基座、LoRA adapter、合并权重和 GGUF 推理模型是不同产物。原 `qwen3.5:9b-qq` 保持为回退，不覆盖、删除或替换。

先确认本机实验报告中训练与 adapter 重载推理通过，再考虑部署。4B 的 adapter 只适配同一 revision 的 Qwen3.5-4B，不能挂到 9B 上；反之亦然。保留基础模型 revision、adapter 配置、数据哈希、评测输出和依赖版本。

1. 使用 `merge_adapter.py` 保存独立的合并 BF16 权重，并核对 logits 误差。脚本不会修改原始权重或 adapter。
2. 使用官方 `ggml-org/llama.cpp` 固定提交的 `convert_hf_to_gguf.py` 转 BF16/F16。确认识别 `Qwen3_5ForConditionalGeneration`，层数、Gated DeltaNet 参数和 tokenizer 正确。不能把转换命令返回 0 当作已能推理；还要检查生成结果，留意 MTP/层数转换兼容性。
3. 使用同一版本的 `llama-quantize` 生成 `Q4_K_M` 或 `Q5_K_M`，先在桌面机串行比较固定聊天、数学、工具格式和长解释。训练本轮未用 4-bit；这里是部署阶段的推理量化，仍要检查量化误差。
4. 将已验证的 GGUF 复制到推理机。量化后的显存余量取决于桌面占用、上下文长度与版本；从 4096 上下文和单路推理开始测，不能仅凭文件大小保证全 GPU 驻留。本机实测 4B 的 BF16 全 GPU 训练张量峰值 10.30 GiB（16 GiB 卡，上限 13.69 GiB），量化后余量更宽；9B 的量化体积与 KV/状态缓存需另测。4B 先验证，9B 后补。
5. 创建新名称，例如 `qwen3.5:4b-qq-lora-exp1`，Modelfile 示例：

```text
FROM C:/Models/qianwen-4b-exp1-Q4_K_M.gguf
PARAMETER num_ctx 4096
PARAMETER num_thread 4
PARAMETER temperature 0.7
PARAMETER top_p 0.8
PARAMETER top_k 20
```

```powershell
ollama create qwen3.5:4b-qq-lora-exp1 -f Modelfile.qianwen-exp1
ollama run qwen3.5:4b-qq-lora-exp1
```

创建后先使用本机隔离评测，再在 AstrBot 临时选择新名称与相同千问娘人设。仅导出语言 GGUF 时，provider 声明 `text,tool_use`；只有另行导出兼容视觉 projector、并实际通过图片输入回归后，才添加 `image`。检查实际请求中工具存在、工具轮无播报、工具结果后正确回答。不能照抄原多模态 profile 后就宣称图片仍有效；本轮语言微调没有做视觉回归。

有回归时选回原 `qwen3.5:9b-qq`，不用恢复旧 SQLite。切换前保留当前 provider 配置，不能用原电脑的整库覆盖新聊天记录。

若转换、量化、Ollama 或笔记本实测未完成，报告必须分别标注“未验证”，保留 Transformers adapter 作为可继续训练的正式产物。

资料：[Ollama 导入](https://docs.ollama.com/import)、[llama.cpp 转换脚本](https://github.com/ggml-org/llama.cpp/blob/master/convert_hf_to_gguf.py)、[Qwen3.5 官方模型](https://huggingface.co/Qwen/Qwen3.5-9B)。
