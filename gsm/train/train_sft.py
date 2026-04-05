import torch
from trl import SFTTrainer, SFTConfig
from .base import BaseTrainerWrapper

class SFTTrainerWrapper(BaseTrainerWrapper):
    """
    SFT训练器封装类
    """
    def train(self, dataset, training_args_dict=None):
        """
        执行SFT训练
        """
        print("===== 开始准备SFT训练 =====")
        default_args = {
            "dataset_text_field": "text",
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "warmup_steps": 5,
            "num_train_epochs": 1,
            "learning_rate": 2e-4,
            "logging_steps": 1,
            "optim": "adamw_8bit",
            "weight_decay": 0.01,
            "lr_scheduler_type": "linear",
            "seed": 3407,
            "report_to": "tensorboard",
            "logging_dir": f"{self.output_dir}/runs",
            "output_dir": self.output_dir,
        }
        
        if training_args_dict:
            default_args.update(training_args_dict)
            
        sft_config = SFTConfig(**default_args)

        self.trainer = SFTTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            peft_config=self.peft_config,
            train_dataset=dataset,
            eval_dataset=None,
            args=sft_config,
        )

        gpu_stats = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
        if gpu_stats:
            start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
            print(f"GPU = {gpu_stats.name}. {start_gpu_memory} GB reserved.")

        print("===== 开始训练 =====")
        trainer_stats = self.trainer.train()
        print(f"训练完成. 耗时: {trainer_stats.metrics.get('train_runtime', 0)} 秒.")
        
        return trainer_stats
