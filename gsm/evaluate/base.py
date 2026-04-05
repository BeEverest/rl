from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseEvaluator(ABC):
    """
    模型评测基类
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @abstractmethod
    def evaluate(self, dataset, **kwargs) -> Dict[str, Any]:
        """
        在指定数据集上进行评测
        
        Returns:
            Dict[str, Any]: 包含评测指标的字典
        """
        pass
