"""RL训练数据集"""

from typing import Dict, Any, Optional
from datasets import load_dataset, Dataset
from pathlib import Path

from .base import BaseDataset

class GSM8KDataset(BaseDataset):
    """GSM8K数学推理数据集

    GSM8K (Grade School Math 8K) 是一个包含8500个高质量小学数学问题的数据集。
    每个问题都需要2-8步的推理过程来解决。
    """

    def __init__(
        self,
        tokenizer=None,  # 用于RL格式应用chat template
        split: str = "train",
        max_samples: Optional[int] = None,
        format_type: str = "sft",  # "sft" or "rl"
    ):
        """
        初始化GSM8K数据集

        Args:
            tokenizer: Tokenizer对象,用于RL格式应用chat template
            split: 数据集分割 ("train" 或 "test")
            max_samples: 最大样本数（用于快速测试）
            format_type: 数据格式类型 ("sft" 用于监督学习, "rl" 用于强化学习)
        """
        super().__init__(tokenizer=tokenizer, split=split, max_samples=max_samples, format_type=format_type)
        
        self.dataset_path = Path(__file__).resolve().parent.parent.parent / "data" / "GSM8K_zh" 
        
        print(f"📥 加载 GSM8K 数据集 (split={split})...")
        self.dataset = load_dataset("json", data_dir=str(self.dataset_path), split=split)

        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
            print(f"   使用 {len(self.dataset)} 个样本（限制：{max_samples}）")
        else:
            print(f"   加载了 {len(self.dataset)} 个样本")
    
    def format_for_sft(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化为SFT训练格式
        """
        question = example["question_zh"]
        answer = example["answer_zh"]
        ground_truth = example['answer_only']
        
        # 提取最终答案
        # if "####" in answer:
        #     reasoning, final_answer = answer.split("####")
        #     reasoning = reasoning.strip()
        #     final_answer = final_answer.strip()
        # else:
        #     reasoning = answer
        #     final_answer = ""
        
        prompt = f"Question: {question}\n\nLet's solve this step by step:\n"
        completion = f"{answer}\n\n <answer>{ground_truth}</answer>"
        messages = [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}]
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        ) if self.tokenizer else f"{prompt}{completion}"

        return {
            "prompt": prompt,
            "text": text,
            "ground_truth": ground_truth,
            "question": prompt
        }
       
    def format_for_rl(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化为RL训练格式
        """
        question = example["question_zh"]
        answer = example["answer_zh"]
        final_answer = example['answer_only']

        # 提取最终答案
        # if "####" in answer:
        #     _, final_answer = answer.split("####")
        #     final_answer = final_answer.strip()
        # else:
        #     final_answer = answer.strip()

        prompt_content = f"Question: {question}\n\nLet's solve this step by step:"

        if self.tokenizer:
            messages = [{"role": "user", "content": prompt_content}]
            prompt_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt_text = prompt_content

        return {
            "prompt": prompt_text,
            "ground_truth": final_answer,
            "question": prompt_content,
            "full_answer": answer
        }
    def format_raw_data(self, example: Dict[str, Any]) -> Dict[str, Any]:
        question = example["question_zh"]
        ground_truth = example['answer_only']
        
        prompt_content = f"Question: {question}\n\nLet's solve this step by step:"
        
        return {
            "prompt": prompt_content,
            "question": prompt_content,
            "ground_truth": ground_truth
        }

def preview_dataset(dataset: Dataset, num_samples: int = 3) -> None:
    """
    预览数据集样本
    """
    print(f"\n📋 数据集预览（前 {num_samples} 个样本）:")
    print("="*80)
    
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        print(f"\n样本 {i+1}:")
        print("-"*80)
        for key, value in sample.items():
            value_str = str(value)
            if len(value_str) > 200:
                value_str = value_str[:200] + "..."
            print(f"{key}: {value_str}")
    
    print("="*80 + "\n")
