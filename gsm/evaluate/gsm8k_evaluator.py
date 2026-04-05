import torch
from .base import BaseEvaluator
from gsm.reward.math_reward import AccuracyReward
from typing import Dict, Any
from tqdm import tqdm

class GSM8KEvaluator(BaseEvaluator):
    """
    GSM8K特定数据集的评测器
    """
    def evaluate(self, dataset, max_samples: int = None, **kwargs) -> Dict[str, Any]:
        self.model.eval()
        correct = 0
        total = 0
        
        total_length = 0
        total_steps = 0
        format_correct = 0
        
        limit = max_samples if max_samples else len(dataset)
        eval_data = dataset.select(range(min(limit, len(dataset))))
        
        reward_func = AccuracyReward()
        
        print(f"开始评测 {len(eval_data)} 个样本...")
        for item in tqdm(eval_data):
            prompt = item['question']
            gt = item['ground_truth']
            
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,  # greedy decoding for eval
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            completion = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            # 1. 计算准确率 (Accuracy)
            score = reward_func([prompt], [completion], ground_truth=[gt])[0]
            if score > 0:
                correct += 1
                
            # 2. 计算长度 (Length)
            total_length += len(completion)
            
            # 3. 计算步骤数 (Steps - 简单以换行数估算)
            total_steps += completion.count('\n')
            
            # 4. 计算格式正确率 (Format Correctness - 考核推理规范标签)
            has_think = "<think>" in completion and "</think>" in completion
            if has_think:
                format_correct += 1
                
            total += 1
            
        accuracy = correct / total if total > 0 else 0.0
        average_length = total_length / total if total > 0 else 0.0
        average_steps = total_steps / total if total > 0 else 0.0
        format_correctness = format_correct / total if total > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "average_length": average_length,
            "average_steps": average_steps,
            "format_correctness": format_correctness,
            "correct": correct,
            "total": total
        }
