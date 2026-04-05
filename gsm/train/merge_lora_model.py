import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

class ModelMerger:
    """
    用于合并基础模型与LoRA权重的工具类
    """
    @staticmethod
    def merge_and_save(base_model_path: str, lora_adapter_path: str, output_path: str):
        """
        合并模型并保存
        """
        print(f"加载基础模型 {base_model_path} ...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        
        print(f"加载并且合并LoRA适配器 {lora_adapter_path} ...")
        model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        model.eval()  # 切换到评估模式，确保权重正确合并    
        model = model.merge_and_unload() # 合并权重
        
        os.makedirs(output_path, exist_ok=True)
        print(f"保存合并后的模型到 {output_path} ...")
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        print("合并完成！")
