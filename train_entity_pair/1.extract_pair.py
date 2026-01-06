import json

def read_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data
def save_json(data,path):
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

# train_revise = read_json("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/yanyi17/test/DocRED_llm/data/docred/train_revised.json")
redoc_train = read_json("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/yanyi17/test/DocRED_llm/data/docred/refine/redocred_train.json")
train_revise_refine = read_json("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/yanyi17/test/DocRED_llm/data/docred/refine/train_revised_refined.json")
# print(len(train_revise),len(redoc_train),len(train_revise_refine))

def get_prompt(doc,entity_list):
    prompt = f'''Given a text and an entity list as input, list the entity pairs that can be identified as possibly containing a relation.
<Text>:
{doc}
<Entity_list>:
{entity_list}
'''
    return prompt
if __name__ == "__main__":
    train_entity_data = []
    id = 0
    for i,doc in enumerate(redoc_train):
        entity_dict = {}
        text = doc["passage"]
        fact_list = doc["fact_list"]
        entity_list = set()
        for fact in fact_list:
            h_name = fact[0]
            t_name = fact[2]
            if h_name not in entity_dict:
                entity_dict[h_name] = []
            if t_name not in entity_dict[h_name]:
                entity_dict[h_name].append(t_name)
        vertexset = train_revise_refine[i]["vertexSet"]
        for entity in vertexset:
            for mention in entity:
                print(mention)
                entity_list.add(mention["name"])
        prompt = get_prompt(text,list(entity_list))
        doc["output"] = entity_dict
        doc["prompt"] = prompt
        doc["input"] = ""
        doc["id"] = i+1
        train_entity_data.append(doc)
    train_data = []
    for item in train_entity_data:
        save_dict = {}
        save_dict["instruction"] = item["prompt"]
        save_dict["input"] = ""
        save_dict["output"] = str(item["output"])
        save_dict["history"] = []
        print(save_dict)
        train_data.append(save_dict)

    save_json(train_data,"/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/yanyi17/LLaMA-Factory/data/train_entity_data.json")
    save_json(train_entity_data,"/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/yanyi17/test/DocRED_llm/data/docred/train/train_entity_data.json")
    

        
    



    