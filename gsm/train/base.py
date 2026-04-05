import torch
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig

class BaseTrainerWrapper(ABC):
    """
    训练器封装抽象类，处理模型加载、分词器、LoRA配置等基础操作
    """
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        lora_config_dict: Optional[Dict[str, Any]] = None,
        use_4bit: bool = True
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.use_4bit = use_4bit
        
        # 默认LoRA配置
        default_lora = {
            "r": 8,
            "lora_alpha": 8,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": "CAUSAL_LM"
        }
        self.lora_config_dict = lora_config_dict or default_lora
        
        self.model = None
        self.tokenizer = None
        self._load_model_and_tokenizer()

    def _load_model_and_tokenizer(self):
        """
        加载预训练模型和分词器，配置量化策略
        """
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

        # 4-bit 量化配置
        bnb_config = None
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, 
            device_map="auto" if torch.cuda.is_available() else "cpu", 
            quantization_config=bnb_config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        self.peft_config = LoraConfig(**self.lora_config_dict)

    @abstractmethod
    def train(self, dataset, **kwargs):
        """
        启动训练
        """
        pass
    
    def save(self):
        """
        保存LoRA适配器和分词器
        """
        if hasattr(self, 'trainer') and self.trainer:
            self.trainer.model.save_pretrained(self.output_dir)
            self.tokenizer.save_pretrained(self.output_dir)
            print(f"已保存模型和分词器到 {self.output_dir}")
