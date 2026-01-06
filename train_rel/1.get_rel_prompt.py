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
        max_new_tokens=2048,
        eos_token_id=terminators,
        do_sample=False,
        temperature=0.1,
    )
    response = outputs[0][input_ids.shape[-1]:]
    response = tokenizer.decode(response, skip_special_tokens=True)
    print(response)
    return  response

def get_completion(prompts, model, tokenizer=None, max_tokens=2048, temperature=0.1,max_model_len=2048):
    # 创建采样参数。temperature 控制生成文本的多样性，top_p 控制核心采样的概率
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    # 初始化 vLLM 推理引擎
    llm = LLM(model=model, tokenizer=tokenizer, max_model_len=max_model_len,trust_remote_code=True,tensor_parallel_size=4)
    outputs = llm.generate(prompts, sampling_params)
    return outputs

def get_api(model, messages, temperature, max_tokens, retries=3, delay=5):
    # 拼接数据
    APP_ID = "1850856272451657817"
    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    # 发起请求
    for attempt in range(retries):
        starttime = datetime.datetime.now()

        # 创建并发送API请求
        rtn = requests.post(
            'https://aigc.sankuai.com/v1/openai/native/chat/completions',
            headers={
                'Authorization': APP_ID,
            },
            json=data
        )

        # 检查请求结果
        if rtn.status_code == 200:
            # 解析返回的结果
            completion = rtn.json()
            endtime = datetime.datetime.now()

            # 打印完成状态
            finish_reason = completion["choices"][0]["finish_reason"]
            if finish_reason != "stop":
                print(f'Finish reason = {finish_reason}')

            # 获取模型的回复
            response = completion["choices"][0]["message"]["content"]
            return response
        else:
            print(f"Error: {rtn.status_code}, {rtn.text}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)  # 等待指定秒数后重试

    # 如果所有重试都失败
    response = "抱歉，请重新编辑后再发送"
    return response

def load_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def save_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print("总共保存了",len(data),"条")

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # 每行是一个独立的JSON对象
            data.append(json.loads(line))
    return data

def get_api(model, messages, temperature, max_tokens, retries=3, delay=5):
    # 拼接数据
    # APP_ID = "1850856272451657817"
    APP_ID = "21901847454677753881"
    data = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens
    }

    # 发起请求
    for attempt in range(retries):
        starttime = datetime.datetime.now()

        # 创建并发送API请求
        rtn = requests.post(
            'https://aigc.sankuai.com/v1/openai/native/chat/completions',
            headers={
                'Authorization': APP_ID,
            },
            json=data
        )

        # 检查请求结果
        if rtn.status_code == 200:
            # 解析返回的结果
            completion = rtn.json()
            endtime = datetime.datetime.now()

            # 打印完成状态
            finish_reason = completion["choices"][0]["finish_reason"]
            if finish_reason != "stop":
                print(f'Finish reason = {finish_reason}')

            # 获取模型的回复
            response = completion["choices"][0]["message"]["content"]
            return response
        else:
            print(f"Error: {rtn.status_code}, {rtn.text}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)  # 等待指定秒数后重试

    # 如果所有重试都失败
    response = "抱歉，请重新编辑后再发送"
    return response

def get_prompt(doc,h_entity,t_entity,relation_list):
    prompt = f'''# Task Background:
You are a professional document-level relation extraction expert who excels at extracting relationships between entities given head entity and tail entity.
# Task Description:
From a document, given the head entity, tail entity, and list of relations, please select one or more relatios existing between the entities from the relations list.For each entity, there is a corresponding entity mention list.
# Thought Process:
1. Summarize specific information about the head entity by combining the information in the entire text with the given head entity mention list.
2. Summarize specific information about the tail entity by combining the information in the entire text with the given tail entity mention list.
3. Determine which specific relations might exist between the head entity and the tail entity from the provided relation_list.
4. Double-check the relations that were identified to confirm whether they truly exist.
# Input Format:
<Document>
{doc}
<Head Entity mention List>
{h_entity}
<Tail Entity menttion List>
{t_entity}
<Relations List>
{relation_list}
# Output Format:
{{"think":"xxxxx","relation":["relation1","relation2",...]]}}
'''
    return prompt
def get_prompt_v2(doc,h_entity,t_entity,relation_list,truth):
    prompt = f'''# Task Background:
You are an expert in document-level relation extraction, specializing in outputting the reasoning steps and answers for the relationships between head and tail entities based on the correct answers corresponding to the head and tail entities.
# Task Description:
Based on the head and tail entity information provided below and the correct relationships between the head and tail entities, output the reasoning steps and answers for the relationships.
# Thought Process:
1. Summarize specific information about the head entity by combining the information in the entire text with the given head entity mention list.
2. Summarize specific information about the tail entity by combining the information in the entire text with the given tail entity mention list.
3. Determine which specific relations might exist between the head entity and the tail entity from the provided relation_list.
4. Double-check the relations that were identified to confirm whether they truly exist.
# Input Format:
<Document>
{doc}
<Head Entity>
{h_entity}
<Tail Entity>
{t_entity}
<Relations List>
{relation_list}
<Truth>
{truth}
# Output Format:
{{"think":"xxxxx","relation":["relation1","relation2",...]]}}
'''
    return prompt
def get_entity_list(doc,id):
    entity_list = []
    vertexSet = doc["vertexSet"]
    entity = vertexSet[id]
    for mention in entity:
        entity_list.append(mention["name"])
    return entity_list
if __name__ == "__main__":
    model_id = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/lingshou/yanyi17/huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    # model_id = "/home/hadoop-hmart-waimaiad/dolphinfs_hdd_hadoop-hmart-waimaiad/lingshou/yanyi17/huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
    # # model_id = "/home/hadoop-hmart-waimaiad/dolphinfs_hdd_hadoop-hmart-waimaiad/lingshou/yanyi17/huggingface.co/meta-llama/Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
    )
    train_revise_refined_50 = load_json("/home/hadoop-hmart-waimaiad/dolphinfs_hdd_hadoop-hmart-waimaiad/lingshou/yanyi17/llm/data/docred/refine/train_revised_refined.json")
    redoc_train_50 = load_json("/home/hadoop-hmart-waimaiad/dolphinfs_hdd_hadoop-hmart-waimaiad/lingshou/yanyi17/llm/data/docred/refine/redocred_train.json")
    rel_desc = load_json("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/lingshou/yanyi17/llm/data/meta/rel_desc.json")
    rel_info = load_json("/home/hadoop-hmart-waimaiad/dolphinfs_hdd_hadoop-hmart-waimaiad/lingshou/yanyi17/llm/data/meta/rel_info.json")
    # relation_list = []
    temperature = 0  # 生成模型的温度参数
    max_tokens = 2048  # 模型生成的最大tokens数
    messages = []
    save_list = []
    # model = "deepseek-v3-friday"
    # for key,value in rel_info.items():
    #     relation_list.append(value)
    save_data = []
    for i, item in enumerate(tqdm(train_revise_refined_50, desc="Processing")):
        save_dict = {}
        doc = redoc_train_50[i]["passage"]
        labels = item["labels"]
        label_dict = {}
        for label in labels:
            h_id = label["h"]
            t_id = label["t"]
            r = rel_info[label["r"]]
            key = str([h_id,t_id])
            if key not in label_dict:
                label_dict[key] = [r]
            else:
                label_dict[key].append(r)
        save_dict["id"] = i
        save_dict["passage"] = doc
        # inf_result = {}
        print(label_dict)
        print(rel_desc)
        prompts = []
        for label,rel_list in label_dict.items():
            prompt_dict = {}
            print(f'{label}:{rel_list}')
            label = eval(label)
            h_id = label[0]
            t_id = label[1] 
            h_entity = get_entity_list(item,h_id)
            t_entity = get_entity_list(item,t_id)
            # prompt = get_prompt(doc,h_entity,t_entity,rel_desc)
            truth = rel_list
            prompt = get_prompt_v2(doc,h_entity,t_entity,rel_desc,truth)
            key = str([h_id,t_id])
            prompt_dict[key] = prompt
            prompt_dict["truth"] = truth
            prompts.append(prompt_dict)
            # save_list.append({"instruction":prompt,"output":""})
            # messages.append({"role": "user", "content": prompt})
            # result = get_result(model,tokenizer,prompt)
            # print(result)
        #     key = str([h_id,t_id])
        #     inf_result[key] = result
        # redoc_train_50[i]["inf_result"]  = inf_result
        save_dict["labels"] = label_dict
        save_dict["prompts"] = prompts
        save_data.append(save_dict)

    save_json(save_data,"/home/hadoop-hmart-waimaiad/dolphinfs_hdd_hadoop-hmart-waimaiad/lingshou/yanyi17/llm/process/redoc_train_prompts.json")

        
