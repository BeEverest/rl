from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datasets import Dataset

class BaseDataset(ABC):
    """
    基础数据集抽象类，定义数据集的通用接口
    """

    def __init__(self, tokenizer=None, split: str = "train", max_samples: Optional[int] = None, format_type: str = "sft"):
        self.tokenizer = tokenizer
        self.split = split
        self.max_samples = max_samples
        self.format_type = format_type
        self.dataset = None

    @abstractmethod
    def format_for_sft(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化为SFT训练格式
        """
        pass

    @abstractmethod
    def format_for_rl(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化为RL训练格式
        """
        pass
    
    @abstractmethod
    def format_raw_data(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """
        原始数据格式化（如果需要）
        """
        pass
    
    def get_dataset(self) -> Dataset:
        """
        获取格式化后的数据集
        """
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Please initialize the dataset properly.")

        if self.format_type == "sft":
            formatted_dataset = self.dataset.map(
                self.format_for_sft,
                remove_columns=self.dataset.column_names,
                load_from_cache_file=False
            )
        elif self.format_type == "rl":
            formatted_dataset = self.dataset.map(
                self.format_for_rl,
                remove_columns=self.dataset.column_names,
                load_from_cache_file=False
            )
        else:
            raise ValueError(f"不支持的格式类型: {self.format_type}")

        return formatted_dataset

    def __len__(self) -> int:
        return len(self.dataset) if self.dataset is not None else 0

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.dataset is None:
            raise ValueError("Dataset not loaded.")
        example = self.dataset[idx]
        if self.format_type == "sft":
            return self.format_for_sft(example)
        else:
            return self.format_for_rl(example)
