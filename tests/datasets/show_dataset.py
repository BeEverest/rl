
import sys
from pathlib import Path

from gsm.datasets.datasets import GSM8KDataset

from transformers import AutoTokenizer


if __name__ == '__main__':
    print("sft数据集" + "="*50 + "\n")
    model_name = "/autodl-fs/data/model/Qwen/Qwen3.5-2B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)  # 这里我们不使用实际的tokenizer，因为我们只是想展示数据格式
    dataset_sft = GSM8KDataset(tokenizer=tokenizer, split="train", max_samples=2, format_type="sft")

    print(dataset_sft[0])
    print(f"数据集大小: {len(dataset_sft)}")
    print(f"数据格式: {dataset_sft.format_type}")
    
    print("\n rl数据集" + "="*50 + "\n")
    dataset_rl = GSM8KDataset(tokenizer=tokenizer, split="train", max_samples=2, format_type="rl")
    # print(dataset_rl[0])
    print(f"数据集大小: {len(dataset_rl)}")
    print(f"数据格式: {dataset_rl.format_type}")
    

