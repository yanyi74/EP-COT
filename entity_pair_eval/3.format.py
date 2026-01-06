import json
import re
import ast
def read_json(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

def clean_string(output_str):
    # 使用正则表达式将未转义的单引号替换为双引号
    json_compatible_str = re.sub(r"(?<!\\)'", '"', output_str)

    json_compatible_str = re.sub(r'(?<=\w)"(?=\w)', "'", json_compatible_str)

    return json_compatible_str

if __name__ == "__main__":
    dev_result = read_json("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/yanyi17/llm/eval/result/dev_result_500.json")
    output = {}
    wrong = 0
    for i, item in enumerate(dev_result):
        output_str = item["output"]
        # json_compatible_str = output_str
        # # 清理字符串以符合JSON格式
        json_compatible_str = clean_string(output_str)
        
        try:
            # 解析为字典
            parsed_dict = json.loads(json_compatible_str)
            output[i] = parsed_dict 
        except json.JSONDecodeError as e:
            # 将无法解析的字符串也保存在输出文件中便于后续检查
            try:
                # 转换字符串为Python字典
                parsed_dict = ast.literal_eval(output_str)
                output[i] =  parsed_dict
            except (SyntaxError, ValueError) as e:
                wrong += 1
                print(output_str)
                output[i] =  output_str
                print(f"Error parsing string:{i}:{e}")

    # 保存输出到指定的文件路径
    save_json(output, "./result/dev_output_500.json")
    print("错误的有", wrong)
