#!/usr/bin/env python
import argparse
from gsm.pipeline.grpo_pipeline import GRPOPipeline

def main():
    parser = argparse.ArgumentParser(description="Run GSM8K GRPO RL Pipeline")
    parser.add_argument("--model", type=str, default=r"D:\Code\models\Qwen3.5-0.8B", help="Path to base model or SFT model")
    parser.add_argument("--output", type=str, default="outputs/Qwen3.5-0.8B-grpo-adapter", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum samples to use for training")
    args = parser.parse_args()

    pipeline = GRPOPipeline(
        model_name=args.model,
        output_dir=args.output,
        max_samples=args.max_samples
    )
    
    pipeline.run()

# python scripts/run_grpo.py --model /autodl-fs/data/model/rl/Qwen3.5-0.8B-sft-merged --output /autodl-fs/data/model/rl/Qwen3.5-0.8B-grpo-adapter-v1 --max_samples 100
# tensorboard --logdir=/autodl-fs/data/model/rl/Qwen3.5-0.8B-grpo-adapter/runs --port=6006
if __name__ == "__main__":
    main()
