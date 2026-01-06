
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import uvicorn
import json
import datetime
import torch
import os
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer 
from transformers.generation import GenerationConfig
from tqdm import tqdm
def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# 设置环境变量
def get_result(model,tokenizer,prompt):
    messages = [
        {"role": "user", "content": prompt},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    outputs = model.generate(
        input_ids,
        max_new_tokens=1024,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
    )
    response = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(response, skip_special_tokens=True)
    print(response)
    return  response


# 主函数入口
if __name__ == '__main__':
    model_id = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/yanyi17/huggingface.co/meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    adapter_name_or_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/yanyi17/model/llama3_8b_v1"
    model = PeftModel.from_pretrained(model, adapter_name_or_path)
    print("Successfully loaded the fine-tuned model")
    model = model.merge_and_unload()
    dev_data = load_json("/home/hadoop-hmart-waimaiad/dolphinfs_hdd_hadoop-hmart-waimaiad/yanyi17/llm/eval/result/dev_pair_prompt.json")
    result = []
    for item in tqdm(dev_data):
        item["output"] = get_result(model,tokenizer,item["prompt"])
        result.append(item)
    save_json(result,"/home/hadoop-hmart-waimaiad/dolphinfs_hdd_hadoop-hmart-waimaiad/yanyi17/llm/eval/result/dev_result_500.json")
    
