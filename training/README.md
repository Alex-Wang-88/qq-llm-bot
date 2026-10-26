# 千问娘离线训练工作区

机器人源码保持原样。本目录管理训练代码；`../.training-local/` 管理本机环境证据、模型、数据、运行日志、检查点与生成结果，全部被 Git 忽略。没有 QQ 登录、凭据、原数据库的依赖。

## 本机基线

Windows 10 企业版 22H2，build 19045；Ryzen 7 7800X3D；RX 9070 XT (gfx1201) 16 GiB；47.11 GiB 可见系统内存。Windows 实际只暴露 8 个逻辑处理器，因此后续默认上限降为 6，保留前台调度余量；没有修改 BIOS 或系统启动配置。AMD WMI 驱动版本 `32.0.32015.7029`。WMI `AdapterRAM` 是 32 位字段，本机读到约 4 GB，不能作为真实显存容量；以 PyTorch 设备信息和分配结果为准。

隔离 Python 3.12.0、AMD 官方 PyTorch `2.9.1+rocm7.2.1`、HIP `7.2.53211-158bd99533`、Transformers 5.4.0、PEFT 0.18.1、Accelerate 1.12.0。精确依赖冻结在本机 `environment-freeze.txt`。安装脚本仅修改此虚拟环境，不改驱动和 Windows。

[AMD 7.2.1 Windows 矩阵](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/compatibility/compatibilityrad/windows/windows_compatibility.html)只列 Windows 11。本机 Windows 10 是否可用，以本项目实际数值验收为准，不能推导为 AMD 官方承诺。[官方安装来源](https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/install/installrad/windows/install-pytorch.html)。

研究过 [Unsloth AMD 支持](https://unsloth.ai/docs/get-started/install/amd)、[Qwen3.5 微调说明](https://unsloth.ai/docs/models/qwen3.5/fine-tune)及 [ROCm/TheRock](https://github.com/ROCm/TheRock)。本轮先用 Transformers/PEFT 的可审计原生路径，未安装或声称验证 Unsloth/Triton 加速。Gated DeltaNet 使用 Transformers 的 PyTorch 回退实现，实际速度由运行日志衡量。

## 运行与资源

从仓库根目录在 PowerShell 中执行。重型入口统一走 `run.ps1`：先采样约 4 秒、检查 RAM/前台/现有重任务、选线程数，再用文件锁拒绝第二个计算任务或重复下载；低 CPU 的下载与计算可并行。Python 保持 BelowNormal，BALANCED 原上限 8，本机检测后降为 6。GAME 上限 6，WORK 最多 12，也受实际逻辑处理器数量与前台余量限制；FULL 仅在用户明确要求全速时使用。不会终止用户程序。

```powershell
.\training\install.ps1
.\training\run.ps1 -Script training/gpu_smoke.py
.\training\run.ps1 -Script training/architecture_smoke.py
.\training\run.ps1 -Script training/prepare_model.py -ScriptArgs @('--size','9B')
py -3.12 -X utf8 training/build_data.py
.\training\run.ps1 -Script training/validate_data.py -ScriptArgs @('--size','9B')
.\training\run.ps1 -Script training/train.py -ScriptArgs @('--config','training/configs/pilot-9b.json')
```

下载先固定 Hugging Face 官方模型的 revision，并验证官方 LFS SHA-256。Xet TLS 和不分段大文件 HTTPS 在本机有停滞，因此采用 32 MiB Range 请求，严格校验 Content-Range、总大小和最终哈希，支持 `.part` 续传。默认从 Qwen 官方 ModelScope 发布源下载权重：先核对每个文件的大小与 SHA-256 和固定 HF 版本相同，再固定 ModelScope revision，完成后再验哈希。Qwen 的[官方仓库](https://github.com/QwenLM/Qwen3.5)也列出 ModelScope 发布渠道。可用 `--source huggingface` 切回 HF。不会从不明训练包替换权重。

9B 的 16 位权重超出独立显存，配置中将前 12 个语言层显式放在 CPU，其余语言层、嵌入与输出头在 GPU。视觉编码器冻结且留在 CPU；仅向语言模块添加 LoRA。显式 `.to()` 跨设备复制保留 autograd，并在重算检查点时生效。这里的 CPU 层实际在 CPU 上计算，与自动显存换页不同；也不把推理用 `device_map="auto"` 当作训练卸载的证据。

4B 已完成实际训练，配置 `configs/sft-4b-v2.json`：`cpu_layers=0`，32 层全部驻留显存，不需要像 9B 那样做 CPU 卸载。前提是必须开梯度检查点（`use_reentrant=False`）——不开的话最长样本（680 token）在 16 GiB 卡上会 OOM，只差 12 MiB。81 步跑完约 6 分 40 秒，中位步时 4.36 秒。显存探测数据见 `EXPERIMENT_REPORT.md`。

```powershell
py -3.12 training/build_data_v2.py
.\training\run.ps1 -Script training/prepare_model.py -ScriptArgs @('--size','4B','--integrity','modelscope')
.\training\run.ps1 -Script training/train.py -ScriptArgs @('--config','training/configs/sft-4b-v2.json','--data','experimental-v2')
```

`--integrity modelscope` 是显式的降级声明：HF 在本机不可达时，权威来源退为魔搭仓库的逐文件 SHA-256，manifest 里会写明这一点。能连 HF 时不要用它。

注意 `run.ps1` 内部的 `$ErrorActionPreference='Stop'` 会把 Python 写到 stderr 的 warning 当成终止错误，用管道接 `Tee-Object` 时尤其容易误杀。需要看实时输出时直接调 `.venv` 的 python 并做文件重定向。

`build_data_v2.py` 在 `experimental-v1` 之上做两件事：追加 89 条原创长回答记录，并重写 v1 里 6 条"被要求详细讲却只回一句"的 `detail` 记录（`OVERRIDES`，不改原始文件）。验证集与测试集逐字节保留，manifest 用 `held_out_identical_to_v1` 断言，保证与 v1 的验证 loss 可比。

正式训练使用 BF16 冻结基座、FP32 LoRA 与 AdamW 状态，不使用 4-bit QLoRA。`gpu_smoke.py` 中优化器数值对照使用 FP32 主参数，避免原生 FP16 优化器状态下溢。512 tokens 先验证，完整数据最长 656 tokens，完整试验设 1024，micro batch 1、累积 4、rank 8、alpha 16。

训练日志包含真实更新次数、训练/验证 loss、梯度范数、参数差异、步耗时、显存和进程 RAM。检查点含 adapter、tokenizer、optimizer、迭代位置、顺序和随机状态。恢复：

```powershell
.\training\run.ps1 -Script training/train.py -ScriptArgs @('--config','training/configs/sft-4b.json','--resume','.training-local/runs/4b-bf16-sft-v1/checkpoint-0015')
```

恢复只使用本机可信检查点；优化器状态是 PyTorch 序列化文件。已有运行默认拒绝覆盖。`summary.json` 的 `completed` 还要求非零参数更新、最终验证和非空生成；启动成功或训练 loss 下降都不单独算成功。

本机 9B 短程数值与卸载验收已通过，继续 9B 时运行 `configs/overfit-9b.json` 和 `configs/sft-9b.json`。完整一轮从原始基座重新初始化，不接着使用两样本过拟合 adapter。训练结束后先运行 `select_adapter.py --run .training-local/runs/9b-bf16-sft-v1`，只按验证 loss 选择保存的检查点，再做测试集生成。选中副本在 `selected-adapter/`，最终步 adapter 仍独立保留。

## 评估、合并与导出

```powershell
.\training\run.ps1 -Script training/select_adapter.py -ScriptArgs @('--run','.training-local/runs/4b-bf16-sft-v2')
.\training\run.ps1 -Script training/verify_adapter.py -ScriptArgs @('--run','.training-local/runs/4b-bf16-sft-v2','--adapter','.training-local/runs/4b-bf16-sft-v2/selected-adapter')
.\training\run.ps1 -Script training/evaluate.py -ScriptArgs @('--size','4B','--cpu-layers','0','--adapter','.training-local/runs/4b-bf16-sft-v2/selected-adapter','--output','comparison-4b-v2')
.\training\run.ps1 -Script training/merge_adapter.py -ScriptArgs @('--run','.training-local/runs/4b-bf16-sft-v2')
```

`select_adapter.py` 只按验证 loss 选检查点，不碰测试集；4B-v2 那轮选出的是第 30 步（2.41769），不是末步（2.45232）。合并与导出脚本本身尚未在本轮执行过，不能算已验证。

评测 A=无原人设的基座、B=基座加仓库现有人设、C=adapter 加完全相同的人设；每题固定同一随机种子及采样。复用原 benchmark 的 fixture，另含独立合成测试、事实、详细解释与工具题。结果写入 `evaluation/`，工具仅回放固定合成证据，不实际联网或发送消息。套话、问号和字符串命中只是启发式指标，需要结合原始输出人工判断，不能直接冒称自然度评分或工具能力全面保留。

评测之后必须做退化审计，只看字数和套话会漏掉更要命的问题。`audit_degeneration.py` 检查四项：行首角色标记、10 字块重复、`<think>` 标签、6 字片段循环。4B-v2 的 C 组就是靠它发现在 `heldout-curated-026` 上自续写伪 `user`/`assistant` 轮并循环到 token 上限的。

这类失控由仓库根目录的 `assistant_output_guard.py` 在代理出口兜底，接线在 `ollama_fix_proxy.py`。它不负责重生成，命中 token 上限只会留下日志。相关测试：

```powershell
python -m unittest tests.test_output_guard
```

合并脚本会核对合并前后的 logits，再保存新的 BF16 权重。之后需另用固定版本 llama.cpp 转 GGUF、量化、推理并对照；这些步骤未执行前只能标为部署准备。[Ollama 导入文档](https://docs.ollama.com/import)不能证明任意 Qwen3.5 adapter 可直接导入；优先验证合并后的模型。参见 `DEPLOYMENT.md`。

本机可复用导出入口为 `prepare_toolchain.py`（官方 b11898、发布方摘要校验）、`merge_adapter.py --adapter .../selected-adapter` 与 `export_gguf.py`（显式禁用 MTP、检查 32 层、BF16 转换、Q4_K_M 量化及真实 llama.cpp 推理）。额外 Python 依赖在 `requirements-export.txt`，用 `uv pip install --python training/.venv/Scripts/python.exe --no-deps -r training/requirements-export.txt` 安装，保留当前 ROCm torch。工具链的通用 requirements 会指定其他 torch/transformers，不能直接覆盖此训练环境。

数据来源、分组方法和局限见 `DATA_CARD.md`。实际运行结果见 `EXPERIMENT_REPORT.md`；它只记录已执行结果。
