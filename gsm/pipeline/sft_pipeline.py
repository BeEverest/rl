from .base import BaseTrainingPipeline
from gsm.datasets.datasets import GSM8KDataset
from gsm.train.train_sft import SFTTrainerWrapper
import os

class SFTPipeline(BaseTrainingPipeline):
    """
    SFT训练流水线
    """
    def __init__(self, model_name: str, output_dir: str, max_samples: int = None):
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_samples = max_samples
        
    def run(self):
        print("======== [SFTPipeline] 初始化 ========")
        # 1. 训练器准备 (同时负责加载模型和Tokenizer)
        trainer = SFTTrainerWrapper(
            model_name=self.model_name,
            output_dir=self.output_dir,
            use_4bit=True
        )
        
        # 2. 数据集准备
        print("======== [SFTPipeline] 加载数据集 ========")
        dataset_obj = GSM8KDataset(
            tokenizer=trainer.tokenizer,
            split="train",
            max_samples=self.max_samples,
            format_type="sft"
        )
        train_dataset = dataset_obj.get_dataset()
        
        # 3. 运行训练
        print("======== [SFTPipeline] 开始训练 ========")
        trainer.train(train_dataset)
        
        # 4. 保存模型
        print("======== [SFTPipeline] 保存模型 ========")
        trainer.save()
        print("SFT Pipeline 执行完毕！")
