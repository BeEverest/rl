from abc import ABC, abstractmethod

class BaseTrainingPipeline(ABC):
    """
    训练流水线的基类，负责将整个流程组装起来
    """
    
    @abstractmethod
    def run(self):
        """
        运行完整的训练（可能包含评测）流程
        """
        pass
