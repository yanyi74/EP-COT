import json

import ast
import re
def read_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data
def save_json(data,path):
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
def calculate_recall(result, label):
    # 转换为集合类型以计算交集和差集
    result_set = set(tuple(pair) for pair in result)
    label_set = set(tuple(pair) for pair in label)
    
    # True Positives: 计算 result 和 label 的交集数量
    true_positives = len(result_set & label_set)

    # False Negatives: label 中存在但 result 中不存在的数量
    false_negatives = len(label_set - result_set)
    
    # 召回率计算
    recall = true_positives / len(label_set)
    
    return recall

not_find = 0

def find_id(entity_list,name):
        global not_find  # 声明使用全局变量
        is_find = False
        for id,entity in enumerate(entity_list):
            for mention in entity:
                if mention["name"] == name:
                   is_find = True
                   break
            if is_find:
                break
        if is_find:
            return id
        else:
            not_find+=1
            print("-----------------")
            

if  __name__ == "__main__":
    dev_output = read_json("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/yanyi17/llm/eval/result/dev_output_50.json")
    dev = read_json("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/yanyi17/llm/data/docred/refine/dev_revised_refined.json")
    output = []
    for key,value in dev_output.items():
        if not isinstance(value, str):
            output.append(value)
        else:
            output.append({})
    all_true = 0
    all_nums = 0
    all_pred = 0
    for i in range(len(output)):
    # for i in range(50):
        result = set()
        doc = dev[i]
        entity_list = doc["vertexSet"]
        for head,tail_list in output[i].items():
            for id,entity in enumerate(entity_list):
                is_true = 0
                for mention in entity:
                    if mention["name"] == head:
                        head_id = id
                        is_true = 1
                        break
                if is_true == 1:
                    break
            list1 = []
            for tail in tail_list:
                id = find_id(entity_list,tail)
                if id:
                    result.add((head_id,id))
                    
                        
        print(result)
        labels = doc["labels"]
        ground_truth = set()
        for label in labels:
            ground_truth.add((label["h"],label["t"]))
        print(ground_truth)
        true_positives = len(ground_truth & result)
        all_true += true_positives
        all_nums += len(ground_truth)
        print(true_positives)
        print(len(ground_truth))
        all_pred += len(result)

        # 计算精确率和召回率
    precision = all_true / all_pred if all_pred > 0 else 0
    recall = all_true / all_nums if all_nums > 0 else 0

    # 计算 F1 得分
    if precision + recall > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0

    print("未找着的实体个数为",not_find)
    print("对的总数是",all_true,"总label是",all_nums)
    print("召回率为",all_true/all_nums)
    print("精确率为",all_true/all_pred)
    print("F1为",f1_score)

            

            
                            









    # label = []
    # for item in dev_label:
    #     for key,value in item.items():
    #         for tail in value:
    #             label.append([key,tail])
    # result = []
    # for item in output:
    #     for key,value in item.items():
    #         for tail in value:
    #             result.append([key,tail])
    # print(label)
    # recall = calculate_recall(result, label)
    # print(f"召回率: {recall:.2%}")
