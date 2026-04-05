# Agentic RL GSM8K 框架

Agentic RL GSM8K 是一个基于最新深度强化学习架构构建的轻量级与模块化模型训练框架。此框架专为强化学习（特别是基于 GRPO 的 PPO 变体）和大型语言模型（LLM）的算术推理任务（如 **GSM8K**）设计，采用了纯粹的面向对象编程 (OOP) 范式，使其具有高扩展性和易用性。

---

## 🏗️ 框架架构与实现原理

本项目被拆解为五个核心组件，均封装为松耦合的 Python Package (`gsm`)：

1. **`gsm.datasets`**: 数据集模块。利用抽象基类 `BaseDataset` 规范了针对 SFT 学习和 RL 强化学习的数据自动格式化（Chat Template，长文本补全等）。
2. **`gsm.reward`**: 强化学习打分模块。结合了硬性数学匹配规则 (`AccuracyReward`)、语言格式与结构化链式推理要求 (`StepReward` 鼓励 `<think>` 标签) 和防作弊长度惩罚 (`LengthPenaltyReward`)。
3. **`gsm.train`**: 训练执行核心。提供了内置 LoRA 与 4-bit BitsAndBytes 量化的 `BaseTrainerWrapper`。其派生的 `SFTTrainerWrapper`与 `GRPOTrainerWrapper` 大幅简化了原生的 `trl` 库调用流程，并挂载了 Tensorboard 指标监控。
4. **`gsm.evaluate`**: 独立的评价基准平台。可以针对任何模型直接预测答案并计算**准确率、平均生成长度、平均生成步数**，以及**思考标签规范率**。
5. **`gsm.pipeline`**: 顶层应用级流水线。提供一键聚合对象、调度任务与持久化模型的管道类。

### 🤖 模型实现
本框架默认兼容 HuggingFace 规范的 Causal LM（如 `Qwen3.5`、`Llama 3` 等）。为了使其在消费级端侧设备或单卡环境（如 24GB VRAM）下顺利训练，该引擎使用：
* **模型精度**：使用 `bitsandbytes` fp4 / int4 量化底座加载模型。
* **参数高效微调 (PEFT)**：自动应用 LoRA (Low-Rank Adaptation) 对 Q/K/V 等注意力块进行外挂式调优。
* **强化优化层**：由 TRL (Transformer Reinforcement Learning) 的 `GRPOTrainer` 接管微调与判分反向传播过程。

---

## ⚙️ 使用条件与环境配置 (Prerequisites)

1. **系统环境**：推荐 Python 3.9+，单张 Nvidia 显卡（支持 CUDA 以加速训练）。
2. **包依赖安装**：
由于本框架使用了完整的包级工程化结构，**在首次运行任何脚本前，必须在项目根目录（带有 `pyproject.toml`）运行本地环境挂载：**
```bash
# 进入项目根目录并挂载 package
pip install -e .
```
*(环境需要的核心第三方库为：`torch`, `transformers`, `datasets`, `peft`, `trl`, `tensorboard`)*

---

## 📥 框架输入与输出

* **输入**：
  * **原数据**：位于 `data/GSM8K_zh/` 目录下（如 `train.json`, `test.json` 等）。
  * **基础模型**：本地的大容量因果语言模型文件夹路径（如 `D:\Code\models\Qwen3.5-0.8B`）。
* **输出**：
  * **模型参数**：训练完成后，默认只输出最轻量化的 **LoRA 适配器权重** (`adapter_model.bin/safetensors` 与 `tokenizer.json` 等) 保存至您的指定输出目录中。
  * **运行指标**：在对应输出目录自动生成 `runs/` 文件夹。用户可通过在命令行输入 `tensorboard --logdir outputs/(对应目录)/runs` 来查看 Loss、准确率和奖励攀升曲线。

---

## 🚀 脚本使用指南

我们提供了一系列封装在 `scripts/` 目录下的即插即用脚本，以便您可以专注跑模型而不是关注繁琐的接口实现。

### 1. 监督微调 (SFT) 训练
用标注好的完整解题过程数据对模型进行第一轮前置“知识植入”。
```bash
python scripts/run_sft.py \
    --model "D:\Code\models\Qwen3.5-0.8B" \
    --output "outputs/Qwen3.5-0.8B-sft-adapter" \
    --max_samples 100
```

### 2. PPO/GRPO 强化学习训练
用奖励函数对上述监督或基座模型加以反馈奖惩、迭代出更好的推导模式（需带有明确 `<think>` 推理层级标签的启发）。
```bash
python scripts/run_grpo.py \
    --model "D:\Code\models\Qwen3.5-0.8B" \
    --output "outputs/Qwen3.5-0.8B-grpo-adapter" \
    --max_samples 500
```

### 3. 模型评测计算与对比
测试基础大模型，或者将训练出的 LoRA 临时缝合在模型上共同接受 GSM8K 验证集对 准确率 等多维度的评估测试。
```bash
# 测试纯净的基础模型
python scripts/run_eval.py \
    --model "D:\Code\models\Qwen3.5-0.8B"

# 测试挂载了刚才训练出的 LoRA 的混合模型
python scripts/run_eval.py \
    --model "D:\Code\models\Qwen3.5-0.8B" \
    --adapter "outputs/Qwen3.5-0.8B-sft-adapter"
```

### 4. LoRA 模型导出与合并
如果想要部署最终微调成果（使用 vLLM 等高性能推理引擎），必须将 LoRA weight 并入到原始浮点数主模型参数上，此脚本就是为了“打补丁合并”提供的。
```bash
python scripts/merge_lora.py \
    --base_model "D:\Code\models\Qwen3.5-0.8B" \
    --lora_adapter "outputs\Qwen3.5-0.8B-sft-adapter" \
    --output "outputs\Qwen3.5-0.8B-sft-merged-final"
```
（合并完成后，导出的 `output` 文件夹就是全量模型资源文件，不再依赖外挂模块）。
