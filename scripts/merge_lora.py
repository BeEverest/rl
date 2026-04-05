#!/usr/bin/env python
import argparse
from gsm.train.merge_lora_model import ModelMerger

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base model")
    parser.add_argument("--base_model", type=str, required=True, help="Path to base model (e.g., D:\\Code\\models\\Qwen3.5-0.8B)")
    parser.add_argument("--lora_adapter", type=str, required=True, help="Path to LoRA adapter (e.g., outputs/Qwen3.5-0.8B-sft-adapter)")
    parser.add_argument("--output", type=str, required=True, help="Output directory for merged model (e.g., outputs/Qwen3.5-0.8B-sft-merged)")
    args = parser.parse_args()

    ModelMerger.merge_and_save(
        base_model_path=args.base_model,
        lora_adapter_path=args.lora_adapter,
        output_path=args.output
    )

# 使用示例:
# 1 sft合并
# python scripts/merge_lora.py --base_model /autodl-fs/data/model/Qwen/Qwen3.5-2B --lora_adapter /autodl-fs/data/model/rl/Qwen3.5-2B-sft-adapter --output /autodl-fs/data/model/rl/Qwen3.5-0.8B-sft-merged

# 2 grpo合并·
# python scripts/merge_lora.py --base_model /autodl-fs/data/model/rl/Qwen3.5-0.8B-sft-merged --lora_adapter /autodl-fs/data/model/rl/Qwen3.5-0.8B-grpo-adapter --output /autodl-fs/data/model/rl/Qwen3.5-0.8B-grpo-merged

# 2 grpo合并v1 
# python scripts/merge_lora.py --base_model /autodl-fs/data/model/rl/Qwen3.5-0.8B-sft-merged --lora_adapter /autodl-fs/data/model/rl/Qwen3.5-0.8B-grpo-adapter-v1 --output /autodl-fs/data/model/rl/Qwen3.5-0.8B-grpo-merged-v1

if __name__ == "__main__":
    main()
