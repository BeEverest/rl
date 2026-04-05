from gsm.datasets.datasets import GSM8KDataset
from gsm.evaluate.gsm8k_evaluator import GSM8KEvaluator
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch
import json

class EvalPipeline:
    """
    用于评估模型表现的流水线类。可以支持对 基础模型 和 微调模型 (带有 LoRA Adapter) 的指标评测。
    """
    def __init__(self, model_name_or_path: str, adapter_path: str = None, max_samples: int = None, use_4bit: bool = True):
        self.model_name_or_path = model_name_or_path
        self.adapter_path = adapter_path
        self.max_samples = max_samples
        self.use_4bit = use_4bit
        self.model = None
        self.tokenizer = None
        
    def _load_model(self):
        print(f"\n======== [EvalPipeline] 加载基础模型 {self.model_name_or_path} ========")
        bnb_config = None
        if self.use_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="fp4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True
            )
            
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            quantization_config=bnb_config
        )
        
        if self.adapter_path:
            print(f"======== [EvalPipeline] 加载微调 LoRA 适配器 {self.adapter_path} ========")
            self.model = PeftModel.from_pretrained(self.model, self.adapter_path)
            
        # 设置模型到eval模式
        self.model.eval()

    def run(self):
        # 1. 挂载加载模型（包含或者不包含 SFT adapter）
        self._load_model()
        
        # 2. 准备测试数据 (测试集)
        print("\n======== [EvalPipeline] 准备测试集 ========")
        dataset_obj = GSM8KDataset(
            tokenizer=self.tokenizer,
            split="test",
            max_samples=self.max_samples,
            format_type="sft" # 推理阶段通常使用 sft 格式的 prompt
        )
        test_dataset = dataset_obj.get_dataset()
        
        # 3. 开始执行评估器跑分
        evaluator = GSM8KEvaluator(self.model, self.tokenizer)
        metrics = evaluator.evaluate(test_dataset, max_samples=self.max_samples)
        
        # 4. 输出评估结果
        print("\n======== [EvalPipeline] 最终评估指标 ========")
        model_type = "Base Model" if not self.adapter_path else "SFT/RL Model (with Adapter)"
        print(f"评估目标类型: {model_type}")
        print(json.dumps(metrics, indent=4, ensure_ascii=False))
        print("======== 评测已完成 ========\n")
        return metrics
