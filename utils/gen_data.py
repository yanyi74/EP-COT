import os
from collections import defaultdict, OrderedDict

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from template import *
from chatgpt_query import *
import json
def read_json_file(file_path):
    # 打开并读取 JSON 文件
    with open(file_path, 'r', encoding='utf-8') as file:
        # 解析 JSON 数据
        data = json.load(file)
    return data


def refine_redocred_data():
    """
        对redocred data进行矫正，使得labels中的head/tail的name名称能够和句子中的匹配
    :return:
    """
    for type in ['train', 'dev', 'test']:
        data = json.load(open(f"../data/docred/{type}_revised.json"))
        new = []
        for index, sample in enumerate(data):
            if sample['labels']:
                sentence = " ".join([" ".join(sent) for sent in [s_ for index, s_ in enumerate(sample['sents'])]])
                for fact in sample['labels']:
                    #遍历labels对应的实体提及检查名字是否在句子中对应，如果没有则用他标签的名字
                    for h in sample['vertexSet'][fact['h']]:
                        if h['name'] not in sentence:
                            h['name'] = " ".join(sample['sents'][h['sent_id']][h['pos'][0]:h['pos'][1]])
                    for t in sample['vertexSet'][fact['t']]:
                        if t['name'] not in sentence:
                            t['name'] = " ".join(sample['sents'][t['sent_id']][t['pos'][0]:t['pos'][1]])
                new.append(sample)
            else:
                new.append(sample)
        json.dump(new, open(f"../data/docred/refine/{type}_revised_refined.json", "w"), indent=4)


def relation_count(source_file, save_file):
    """
        统计数据的relation分布情况
    :return:
    """
    data = json.load(open(source_file))
    relations_dict = defaultdict(int)
    for sample in data:
        for relation in sample['relations']:
            relations_dict[relation] += 1 #记录数据集包含该关系的数量
    relations_dict = OrderedDict(sorted(relations_dict.items(), key=lambda x: x[1], reverse=True))
    json.dump(relations_dict, open(save_file, "w"), indent=4) 


def fact_count(source_file, save_file):
    """
        统计数据的fact分布情况
    :return:
    """
    data = json.load(open(source_file))
    relations_dict = defaultdict(int)
    for sample in data:
        relations = [fact[0][1] for fact in sample['same_fact_list']] 
        for relation in relations:
            relations_dict[relation] += 1 #记录fact中包含关系的数量但这个是没有去重的
    relations_dict = OrderedDict(sorted(relations_dict.items(), key=lambda x: x[1], reverse=True))
    json.dump(relations_dict, open(save_file, "w"), indent=4) 


def make_redocred_data(data_types, source_path, save_path):
    """
        将redocred数据处理成指定的格式
    :param data_types: 处理的数据类型
    :param source_path: redocred 数据所在路径文件夹
    :param save_path: 保存的文件夹
    :return:
    """
    refine_redocred_data()
    for index, data_type in enumerate(data_types):
        final_save = []
        data = json.load(open(os.path.join(source_path, f"{data_type}_revised_refined.json")))
        for page_id, sample in enumerate(data):
            fact_list = []
            same_fact_list = []
            relations = set()
            sentence = " ".join([" ".join(sent) for sent in [s_ for index, s_ in enumerate(sample['sents'])]])
            for fact in sample['labels']:
                head_name = list(set([h['name'] for h in sample['vertexSet'][fact['h']]]))
                tail_name = list(set([t['name'] for t in sample['vertexSet'][fact['t']]]))
                relation = pid_name[fact['r']] #p:name
                same_fact = [] #记录每一个三元组对应所有提及构成的[head, relation, tail]
                for head in head_name:
                    for tail in tail_name:
                        relations.add(relation)
                        if (head, relation, tail) not in same_fact:
                            same_fact.append([head, relation, tail])
                        if (head, relation, tail) not in fact_list:
                            fact_list.append(
                                [head, relation, tail],
                            )  #记录对应所有fact
                same_fact_list.append(same_fact) #[[[],[]],[]] 一个rel对应的所有提及名字组成的[[[name1,name2],[name1,name2]]]
            save = {
                "index": index,
                "page_id": page_id,
                "passage": sentence,
                "relations": list(relations), #记录文档包含的关系
                "fact_list": fact_list, #所有的三元组信息
                "same_fact_list": same_fact_list, #每个三元组对应的提及列表的列表
                "data_from": f"redocred_{data_type}"
            }
            final_save.append(save) #记录每一个文档
        with open(os.path.join(save_path, f"redocred_{data_type}.json"), "w") as f:
            json.dump(final_save, f, indent=4)


#D_RS_F 先问关系列表 然后根据关系列表问三元组
def relations_fact(source_file, save_file):
    """
        D_RS_F
    :param source_file: 文件所在路径
    :param save_path: 保存文件路径
    :return:
    """
    train_data = []
    data = json.load(open(source_file))
    global_id = 0
    for sample in data:
        sentence = sample['passage']
        block_dict = {
            "id": f"identity_{global_id}",
            "instruction": templates[version]["relation_list_template"].format(sentences=sentence),
            "input": "",
            "output": str(str("\n".join(sample['relations']))),
            "history": []
        }
        train_data.append(block_dict)
        global_id += 1

        block_dict = {
            "id": f"identity_{global_id}",
            "instruction": templates[version]["fact_list_template"].format(sentences=sentence, relations=sample['relations']),
            "input": "",
            "output": str("\n".join([str(fact) for fact in sample['fact_list']])),
            "history": []
        }
        train_data.append(block_dict)
        global_id += 1
    os.makedirs(os.path.dirname(save_file), exist_ok=True) if not os.path.exists(save_file) else None
    json.dump(train_data, open(save_file, "w"), indent=4)

def D_S_F(source_file,save_file):
    rel_desc = read_json_file("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/yanyi17/test/DocRED_llm/data/meta/rel_info.json")
    relation_list = [name for id,name in rel_desc.items()]
    train_data = []
    data = json.load(open(source_file))
    global_id = 0
    for sample in data:
        sentence = sample['passage']
        subjects = list(set(fact[0] for fact in sample['fact_list']))
        block_dict = {
            "id": f"identity_{global_id}",
            "instruction": templates[version]["entity_list_template"].format(sentences=sentence,relation_list = relation_list),
            "input": "",
            "output": str(subjects),
            "history": []
        }
        train_data.append(block_dict)
        global_id += 1
        for subject in subjects:
            sub_facts = [fact for fact in sample['fact_list'] if fact[0] == subject]
            reltobj = {}
            for fact in sub_facts:
                if fact[1] not in reltobj:
                    reltobj[fact[1]] = [fact[2]]
                else:
                    reltobj[fact[1]].append(fact[2])
                
            block_dict = {
                "id": f"identity_{global_id}",
                "instruction": templates[version]["fact_list_template"].format(sentences=sentence,relation_list = relation_list,head = subject),
                "input": "",
                "output": str(reltobj),
                "history": []
            }
            train_data.append(block_dict)
            global_id += 1
    os.makedirs(os.path.dirname(save_file), exist_ok=True) if not os.path.exists(save_file) else None
    json.dump(train_data, open(save_file, "w"), indent=4)


if __name__ == '__main__':
    # preprocess for redocred
    make_redocred_data(data_types=['train', 'dev', 'test'], source_path="../data/docred/refine", save_path="../data/docred/refine")
    source_train = "../data/docred/refine/redocred_train.json"
    source_test = "../data/docred/refine/redocred_test.json"
    relation_count(source_file=source_test, save_file="../data/docred/infor/redocred_test_relation_count.json")
    fact_count(source_file=source_test, save_file="../data/docred/infor/redocred_test_fact_count.json")
    # make data for train_set and test_set for 1 lora
    version = "Head_fact"
    D_S_F(source_file=source_train, save_file=f"../data/train/{version}/train.json") 
    D_S_F(source_file=source_test, save_file=f"../data/train/{version}/test.json")
