import json
import re
def read_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data
def save_json(data,path):
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    rel_info = read_json("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/lingshou/yanyi17/llm/data/meta/rel_info.json")
    new_desc = read_json("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/lingshou/yanyi17/llm/data/meta/new_rel_desc.json")
    save_dict = {}
    for key,value in rel_info.items():
        if key in new_desc.keys():
            save_dict[value] = new_desc[key]
        else:
            print("key=",key)
    save_json(save_dict,"/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/lingshou/yanyi17/llm/data/meta/rel_desc.json")
    
        
   