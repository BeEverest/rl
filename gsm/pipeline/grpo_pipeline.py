from .base import BaseTrainingPipeline
from gsm.datasets.datasets import GSM8KDataset
from gsm.train.train_grpo import GRPOTrainerWrapper
from gsm.reward.math_reward import AccuracyReward, LengthPenaltyReward, StepReward
import os

class GRPOPipeline(BaseTrainingPipeline):
    """
    GRPO RL训练流水线
    """
    def __init__(self, model_name: str, output_dir: str, max_samples: int = None):
        self.model_name = model_name
        self.output_dir = output_dir
        self.max_samples = max_samples
        
    def run(self):
        print("======== [GRPOPipeline] 初始化 ========")
        # 1. 训练器准备
        trainer = GRPOTrainerWrapper(
            model_name=self.model_name,
            output_dir=self.output_dir,
            use_4bit=True
        )
        
        # 2. 数据集准备
        print("======== [GRPOPipeline] 加载数据集 ========")
        dataset_obj = GSM8KDataset(
            tokenizer=trainer.tokenizer,
            split="train",
            max_samples=self.max_samples,
            format_type="rl"
        )
        train_dataset = dataset_obj.get_dataset()
        
        # 3. 奖励函数设定
        reward_funcs = [
            AccuracyReward(),
            # LengthPenaltyReward(target_length= 200, penalty_factor=0.001),
            StepReward()
        ]
        
        reward_weights = [1.0, 0.5]  # 奖励函数权重
        
        # 4. 运行训练
        print("======== [GRPOPipeline] 开始训练 ========")
        trainer.train(train_dataset, reward_funcs=reward_funcs, training_args_dict={"reward_weights": reward_weights})
        
        # 5. 保存模型
        print("======== [GRPOPipeline] 保存模型 ========")
        trainer.save()
        print("GRPO Pipeline 执行完毕！")
