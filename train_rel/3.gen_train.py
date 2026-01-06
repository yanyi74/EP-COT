

import os
import json
from tqdm import tqdm
import re
import random
import copy

def read_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

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
def get_entity_list(doc,id):
    entity_list = []
    vertexSet = doc["vertexSet"]
    entity = vertexSet[id]
    for mention in entity:
        entity_list.append(mention["name"])
    return entity_list
# if __name__ == "__main__":
#     data1 = read_json("/home/hadoop-hmart-waimaiad/dolphinfs_hdd_hadoop-hmart-waimaiad/lingshou/yanyi17/llm/process/redoc_train_prompts_llm_output_1.json")
#     data2 = read_json("/home/hadoop-hmart-waimaiad/dolphinfs_hdd_hadoop-hmart-waimaiad/lingshou/yanyi17/llm/process/redoc_train_prompts_llm_output_2.json")
#     data3 = read_json("/home/hadoop-hmart-waimaiad/dolphinfs_hdd_hadoop-hmart-waimaiad/lingshou/yanyi17/llm/process/redoc_train_prompts_llm_output_3.json")
#     data = data1 + data2 + data3
#     train_revise_refined_50 = load_json("/home/hadoop-hmart-waimaiad/dolphinfs_hdd_hadoop-hmart-waimaiad/lingshou/yanyi17/llm/data/docred/refine/train_revised_refined.json")
#     redoc_train_50 = load_json("/home/hadoop-hmart-waimaiad/dolphinfs_hdd_hadoop-hmart-waimaiad/lingshou/yanyi17/llm/data/docred/refine/redocred_train.json")
#     rel_desc = load_json("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/lingshou/yanyi17/llm/data/meta/rel_desc.json")
#     rel_info = load_json("/home/hadoop-hmart-waimaiad/dolphinfs_hdd_hadoop-hmart-waimaiad/lingshou/yanyi17/llm/data/meta/rel_info.json")

#     max_per_label = 100000  # 每个类别最多采样200条
#     # 统计label出现次数
#     label_count = {}

#     selected_prompts = []
#     for doc in data:
#         id = doc["id"]
#         redoc = train_revise_refined_50[id]
#         prompts = doc["prompts"]
#         for prompt in prompts:
#             key,value = list(prompt.items())[0]
#             h_t = eval(key)
#             print(h_t)
#             h,t = int(h_t[0]),int(h_t[1])
#             print(h,t)
#             h_entity = get_entity_list(redoc,h)
#             t_entity = get_entity_list(redoc,t)
#             labels = prompt["truth"]
#             #检查该prompt的所有label是否都未超采样上限
#             over_limit = False
#             for label in labels:
#                 if label_count.get(label, 0) >= max_per_label:
#                     over_limit = True
#                     break
#             if over_limit:
#                 continue
#             #采样该prompt，并更新所有label计数
#             passage = redoc_train_50[id]["passage"]
#             prompt["prompt"] = get_prompt(passage,h_entity,t_entity,rel_desc)
#             selected_prompts.append(prompt)
#             for label in labels:
#                 label_count[label] = label_count.get(label, 0) + 1

#     print(f"采样后训练集样本数: {len(selected_prompts)}")
#     sorted_label = sorted(label_count.items(), key=lambda x: -x[1])
#     print(sorted_label)
#     train_data = []
#     for item in selected_prompts:
#         save_dict = {}
#         save_dict["instruction"] = ""
#         # key, value = list(item.items())[0]
#         save_dict["input"] = item["prompt"]
#         output = item["llm_output"] 
#         # 使用正则表达式去掉第一个</think>
#         output = re.sub(r'</think>', '', output, count=1)
#         save_dict["output"] = output
#         save_dict["history"] = []
#         train_data.append(save_dict)
#     # # 如需保存
#     # with open("train2.json", "w", encoding="utf-8") as f:
#     #     json.dump(train_data, f, ensure_ascii=False, indent=2)
    

def stratified_sampling(selected_prompts):
    """
    分层采样并缓解长尾，头部关系保留较多样本，尾部关系自动上采样。
    """
    # 可根据实际情况调整
    HEAD_REL_LIMIT = {
        'located in the administrative territorial entity': 2500,
        'country': 2000,
    }
    MID_LIMIT = 500
    TAIL_TARGET = 100
    MID_MIN = 200
    HEAD_MIN = 1000

    # 统计每个关系对应的样本
    rel2samples = {}
    for prompt in selected_prompts:
        for label in prompt["truth"]:
            rel2samples.setdefault(label, []).append(prompt)

    sampled_prompts = []
    rel_sampled_count = {}

    for rel, samples in rel2samples.items():
        count = len(samples)
        # 头部关系特殊采样
        if rel in HEAD_REL_LIMIT:
            limit = min(HEAD_REL_LIMIT[rel], count)
            sampled = random.sample(samples, limit)
        # 中部关系
        elif count > MID_MIN:
            sampled = random.sample(samples, min(MID_LIMIT, count))
        # 尾部关系
        else:
            sampled = samples.copy()
            if len(sampled) < TAIL_TARGET:
                while len(sampled) < TAIL_TARGET:
                    sampled.append(copy.deepcopy(random.choice(samples)))
        sampled_prompts.extend(sampled)
        rel_sampled_count[rel] = len(sampled)

    print("最终每个关系采样数：")
    for rel, num in sorted(rel_sampled_count.items(), key=lambda x: -x[1]):
        print(f"{rel}: {num}")

    print(f"最终训练集样本总数: {len(sampled_prompts)}")
    return sampled_prompts

if __name__ == "__main__":
    data1 = read_json("/home/hadoop-hmart-waimaiad/dolphinfs_hdd_hadoop-hmart-waimaiad/lingshou/yanyi17/llm/process/redoc_train_prompts_llm_output_1.json")
    data2 = read_json("/home/hadoop-hmart-waimaiad/dolphinfs_hdd_hadoop-hmart-waimaiad/lingshou/yanyi17/llm/process/redoc_train_prompts_llm_output_2.json")
    data3 = read_json("/home/hadoop-hmart-waimaiad/dolphinfs_hdd_hadoop-hmart-waimaiad/lingshou/yanyi17/llm/process/redoc_train_prompts_llm_output_3.json")
    data = data1 + data2 + data3
    train_revise_refined_50 = load_json("/home/hadoop-hmart-waimaiad/dolphinfs_hdd_hadoop-hmart-waimaiad/lingshou/yanyi17/llm/data/docred/refine/train_revised_refined.json")
    redoc_train_50 = load_json("/home/hadoop-hmart-waimaiad/dolphinfs_hdd_hadoop-hmart-waimaiad/lingshou/yanyi17/llm/data/docred/refine/redocred_train.json")
    rel_desc = load_json("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/lingshou/yanyi17/llm/data/meta/rel_desc.json")
    rel_info = load_json("/home/hadoop-hmart-waimaiad/dolphinfs_hdd_hadoop-hmart-waimaiad/lingshou/yanyi17/llm/data/meta/rel_info.json")

    # -----------原始采样部分替换为如下-----------
    selected_prompts = []
    for doc in data:
        id = doc["id"]
        redoc = train_revise_refined_50[id]
        prompts = doc["prompts"]
        for prompt in prompts:
            key, value = list(prompt.items())[0]
            h_t = eval(key)
            h, t = int(h_t[0]), int(h_t[1])
            h_entity = get_entity_list(redoc, h)
            t_entity = get_entity_list(redoc, t)
            labels = prompt["truth"]
            passage = redoc_train_50[id]["passage"]
            prompt["prompt"] = get_prompt(passage, h_entity, t_entity, rel_desc)
            selected_prompts.append(prompt)

    # 分层采样，缓解长尾
    final_sampled_prompts = stratified_sampling(selected_prompts)

    # 生成最终训练集
    train_data = []
    for item in final_sampled_prompts:
        save_dict = {}
        save_dict["instruction"] = ""
        save_dict["input"] = item["prompt"]
        output = item["llm_output"]
        output = re.sub(r'</think>', '', output, count=1)
        save_dict["output"] = output
        save_dict["history"] = []
        train_data.append(save_dict)

    # 保存训练集
    save_json(train_data, "train2.json")
    # 或
    # with open("train2.json", "w", encoding="utf-8") as f:
    #     json.dump(train_data, f, ensure_ascii=False, indent=2)
       
    