#!/usr/bin/env python
import argparse
from gsm.pipeline.eval_pipeline import EvalPipeline

def main():
    parser = argparse.ArgumentParser(description="Evaluate GSM8K Base and SFT Models")
    parser.add_argument("--model", type=str, default=r"D:\Code\models\Qwen3.5-0.8B", help="Path to base model")
    parser.add_argument("--adapter", type=str, default=None, help="Path to LoRA adapter (if empty, runs base model eval)")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum samples to evaluate")
    args = parser.parse_args()

    pipeline = EvalPipeline(
        model_name_or_path=args.model,
        adapter_path=args.adapter,
        max_samples=args.max_samples
    )
    
    pipeline.run()

# python scripts/run_eval.py --model "D:\Code\models\Qwen3.5-0.8B" --adapter "outputs/Qwen3.5-0.8B-sft-adapter"
# 1 sft
# python scripts/run_eval.py --model /autodl-fs/data/model/rl/Qwen3.5-0.8B-sft-merged --max_samples 100

# 2 grpo
# python scripts/run_eval.py --model /autodl-fs/data/model/rl/Qwen3.5-0.8B-grpo-merged-v1 --max_samples 100

if __name__ == "__main__":
    main()

