from trl import GRPOTrainer, GRPOConfig
from gsm.train.base import BaseTrainerWrapper
from typing import List, Callable

class GRPOTrainerWrapper(BaseTrainerWrapper):
    """
    GRPO RL训练器封装类
    """
    def train(self, dataset, reward_funcs: List[Callable], training_args_dict=None):
        """
        执行GRPO训练
        """
        print("===== 开始准备GRPO训练 =====")
        default_args = {
            # 基础配置
            "output_dir": self.output_dir,
            "learning_rate": 1e-7,
            "min_lr_ratio": 0.1,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 8,
            
            # GRPO 特定参数
            "num_generations": 8,                  # 每个 prompt 生成的样本数
            "max_completion_length": 256,          # 最大生成长度
            "temperature": 1.0,                    # 生成温度
            "beta": 0.01,                          # KL 惩罚系数
            
            # 训练配置
            "num_train_epochs": 1,
            "logging_steps": 10,
            "save_steps": 100,
            "save_total_limit": 2,
            
            # 日志配置
            "report_to": "tensorboard",
            "logging_dir": f"{self.output_dir}/runs",
            
            # 优化器配置
            "optim": "adamw_torch",
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.1,
            
            # 其他
            "bf16": False,                         # 使用 bfloat16
            "fp16": False,
            "dataloader_num_workers": 4,
            "remove_unused_columns": False,        # 保留 ground_truth 列
        }
        
        if training_args_dict:
            default_args.update(training_args_dict)
            
        grpo_config = GRPOConfig(**default_args)

        self.trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            peft_config=self.peft_config,
            reward_funcs=reward_funcs,
            train_dataset=dataset,
            args=grpo_config,
        )

        print("===== 开始训练 =====")
        trainer_stats = self.trainer.train()
        print(f"训练完成. 耗时: {trainer_stats.metrics.get('train_runtime', 0)} 秒.")
        
        return trainer_stats
