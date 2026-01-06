#  stop_token_ids = [151329, 151336, 151338]
import datetime
import requests
import time
import json
import re
import random
from tqdm import tqdm
import os
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import os
import json
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

def get_completion(llm, prompts, max_tokens=4096, temperature=0.1):
    stop_token_ids = [151329, 151336, 151338]
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids,
    )
    outputs = llm.generate(prompts, sampling_params)
    return outputs

def read_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def batch_iter(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i+batch_size], i

if __name__ == "__main__":
    # 1. 读取数据
    data = read_json("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/lingshou/yanyi17/llm/result/ds_llama3_8b_rel_prompts.json")
    # model_id = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/lingshou/yanyi17/huggingface.co/meta-llama/Llama-3.1-8B-Instruct"
    # tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_id,
    #     torch_dtype=torch.bfloat16,
    #     device_map="auto",
    # )
    # adapter_name_or_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/lingshou/yanyi17/model/llama3_8b_rel_2"
    # model = PeftModel.from_pretrained(model, adapter_name_or_path)
    # print("Successfully loaded the fine-tuned model")
    # model = model.merge_and_unload()
    model = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/lingshou/yanyi17/model/llama3_rel-lora"
    tokenizer ="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/lingshou/yanyi17/model/llama3_rel-lora"
    # 随机打乱
    # random.shuffle(data)
    # data1 = data[:1000]
    # data2 = data[1000:2000]
    # data3 = data[2000:]
    # save_json(data1, "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/lingshou/yanyi17/llm/process/redoc_train_prompts_1.json")
    # save_json(data2, "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/lingshou/yanyi17/llm/process/redoc_train_prompts_2.json")
    # save_json(data3, "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/lingshou/yanyi17/llm/process/redoc_train_prompts_3.json")
    # 2. 收集所有 prompt 及其索引（便于回填）
    prompt_list = []
    index_list = []  # [(data_idx, prompt_idx, key)]

    for data_idx, item in enumerate(data):
        for prompt_idx, prompt_dict in enumerate(item['prompts']):
            key = list(prompt_dict.keys())[0]
            prompt_list.append(prompt_dict[key])
            index_list.append((data_idx, prompt_idx, key))
    print(len(prompt_list))
    # 3. 只初始化一次 LLM
    llm = LLM(model=model, tokenizer=tokenizer, max_model_len=10000, trust_remote_code=True, tensor_parallel_size=8)

    # 4. 分批推理
    batch_size = 8
    all_outputs = []
    for batch_prompts, start_idx in tqdm(batch_iter(prompt_list, batch_size), total=(len(prompt_list)+batch_size-1)//batch_size):
        outputs = get_completion(llm, batch_prompts, max_tokens=2048)
        print(outputs)
        all_outputs.extend(outputs)

    # 5. 回填结果
    for i, output in enumerate(all_outputs):
        generated_text = output.outputs[0].text
        data_idx, prompt_idx, key = index_list[i]
        # 在原始结构的对应 prompt 字典内增加 'llm_output'
        data[data_idx]['prompts'][prompt_idx]['llm_output'] = generated_text

    # 6. 保存结果
    save_json(data, "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/lingshou/yanyi17/llm/process/redoc_train_prompts_llm_output_50.json")
    print(f"推理完成！共生成{len(all_outputs)}条结果。")
