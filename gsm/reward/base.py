from abc import ABC, abstractmethod
from typing import List, Dict, Any, Union

class BaseRewardFunction(ABC):
    """
    奖励函数基类，定义了奖励函数的调用接口
    """
    @property
    def __name__(self):
        return self.__class__.__name__
        
    @abstractmethod
    def __call__(
        self, 
        prompts: List[Union[str, List[Dict[str, str]]]], 
        completions: List[str], 
        **kwargs
    ) -> List[float]:
        """
        计算每个完成补全的奖励分数
        
        Args:
            prompts: 输入提示列表 (支持纯文本或消息列表)
            completions: 模型生成的补全列表
            **kwargs: 额外参数，如ground_truth等
            
        Returns:
            List[float]: 对应每个样本的奖励分数列表
        """
        pass
