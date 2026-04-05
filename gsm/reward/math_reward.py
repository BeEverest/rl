import re
from typing import List, Dict, Any, Union
from gsm.reward.base import BaseRewardFunction

class AccuracyReward(BaseRewardFunction):
    """准确率奖励函数：提取答案并与 ground_truth 进行匹配"""
    
    def extract_answer(self, completion: str) -> str:
        """从补全内容中提取答案"""
        match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
        if match:
            return match.group(1).strip()
            
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", completion)
        if numbers:
            return numbers[-1]
            
        return ""

    def __call__(
        self, 
        prompts: List[Union[str, List[Dict[str, str]]]], 
        completions: List[str], 
        **kwargs
    ) -> List[float]:
        ground_truths = kwargs.get('ground_truth', [])
        
        if not ground_truths or len(ground_truths) != len(completions):
            return [0.0] * len(completions)
            
        rewards = []
        for completion, gt in zip(completions, ground_truths):
            pred = self.extract_answer(completion)
            if str(pred) == str(gt):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
                
        return rewards

class LengthPenaltyReward(BaseRewardFunction):
    """长度惩罚函数：防止模型生成无意义的过长文本，只有答案正确时进行惩罚"""
    
    def __init__(self, target_length=200, penalty_factor=0.02):
        self.target_length = target_length
        self.penalty_factor = penalty_factor
        
    def extract_answer(self, completion: str) -> str:
        """从补全内容中提取答案"""
        match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
        if match:
            return match.group(1).strip()
            
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", completion)
        if numbers:
            return numbers[-1]
            
        return ""
        
    def __call__(
        self, 
        prompts: List[Union[str, List[Dict[str, str]]]], 
        completions: List[str], 
        **kwargs
    ) -> List[float]:
        ground_truths = kwargs.get('ground_truth', [])
        
        if not ground_truths or len(ground_truths) != len(completions):
            return [0.0] * len(completions)
            
        rewards = []
        for completion, gt in zip(completions, ground_truths):
            pred = self.extract_answer(completion)
            is_correct = (str(pred) == str(gt))
            
            if not is_correct:
                rewards.append(0.0)
            else:
                length = len(completion)
                if length <= self.target_length:
                    rewards.append(1.0)
                else:
                    rewards.append(1.0 - self.penalty_factor * (length - self.target_length))
        return rewards

class StepReward(BaseRewardFunction):
    """步骤奖励函数：鼓励多步推理"""
    
    def __call__(
        self, 
        prompts: List[Union[str, List[Dict[str, str]]]], 
        completions: List[str], 
        **kwargs
    ) -> List[float]:
        rewards = []
        for completion in completions:
            score = 0.0
            # 基础格式奖励
            if "<think>" in completion and "</think>" in completion:
                score += 0.1
                
            # 基于步骤数量（如\n）增加奖励，鼓励详细思考步骤，赋予一定的软性上限
            steps = completion.count("\n")
            score += min(steps * 0.01, 0.5)
            
            rewards.append(score)
            
        return rewards
