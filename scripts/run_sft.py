#!/usr/bin/env python
import argparse
from gsm.pipeline.sft_pipeline import SFTPipeline

def main():
    parser = argparse.ArgumentParser(description="Run GSM8K SFT Pipeline")
    parser.add_argument("--model", type=str, default=r"D:\Code\models\Qwen3.5-0.8B", help="Path to base model")
    parser.add_argument("--output", type=str, default="outputs/Qwen3.5-0.8B-sft-adapter", help="Output directory")
    parser.add_argument("--max_samples", type=int, default=100, help="Maximum samples to use for training")
    args = parser.parse_args()

    pipeline = SFTPipeline(
        model_name=args.model,
        output_dir=args.output,
        max_samples=args.max_samples
    )
    
    pipeline.run()
# python scripts/run_sft.py --model /autodl-fs/data/model/Qwen/Qwen3.5-2B --output /autodl-fs/data/model/rl/Qwen3.5-2B-sft-adapter --max_samples 10000
# tensorboard --logdir=/autodl-fs/data/model/rl/Qwen3.5-2B-sft-adapter --port=6006
if __name__ == "__main__":
    main()
