import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 加载模型配置
base_model_name = "/autodl-fs/data/model/Qwen/Qwen3.5-2B"  # 基础模型路径
sft_model_name = "/autodl-fs/data/model/rl/Qwen3.5-0.8B-sft-merged"
grpo_model_name = "/autodl-fs/data/model/rl/Qwen3.5-0.8B-grpo-merged"
# 加载分词器
tokenizer = AutoTokenizer.from_pretrained(grpo_model_name)
model = AutoModelForCausalLM.from_pretrained(grpo_model_name, trust_remote_code=True)  # 预加载模型以加速后



def generate_response(
    prompt,
    tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    num_return_sequences=1
):
    """
    生成模型响应
    
    Args:
        prompt: 输入文本
        max_new_tokens: 最大生成token数
        temperature: 温度参数（控制随机性）
        top_p: nucleus sampling参数
        do_sample: 是否使用采样
        num_return_sequences: 返回的序列数量
    """
    # 编码输入
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    ).to(model.device)
    
    # 生成响应
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            num_return_sequences=num_return_sequences,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1  # 减少重复
        )
    
    # 解码响应
    responses = []
    for output in outputs:
        response = tokenizer.decode(
            output[inputs['input_ids'].shape[1]:],  # 只解码新生成的部分
            skip_special_tokens=True
        )
        responses.append(response)
    
    return responses if num_return_sequences > 1 else responses[0]



# 示例1：单个问题推理
if __name__ == "__main__":
    model_list = [base_model_name, sft_model_name, grpo_model_name]
    
    test_question = """ 篮子里有一些橙子。Ana削一个橙子要花3分钟，Jane削同样的橙子要花4分钟。如果Ana和Jane同时开始从这个篮子里拿橙子来削，那么一个小时后Ana比Jane多削了多少个橙子？"""
    
    for model_path in model_list:
        print("=" * 100)     
        print(f"正在评测模型: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to("cuda")
        response = generate_response(
            test_question,
            tokenizer,
            max_new_tokens=256,
            temperature=0.7,
        )
        print(f"Generated response:\n{response}")

    

    

    

    

    




