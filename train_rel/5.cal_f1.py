import json
import re
from collections import defaultdict

def read_json(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    redoc_train_50_result = read_json("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/lingshou/yanyi17/llm/result/ds_llama3_8b_rel_pred_1.json")
    all_true = 0
    all_pred = 0
    all_num = 0
    all_label_num = 0
    save_wrong = []
    error = 0

    relation_stat = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'pred_count': 0, 'label_count': 0})
    mis_predict_stat = defaultdict(lambda: defaultdict(int))

    for item in redoc_train_50_result[:50]:
        inf_result = item["inf_result"]
        labels = item["labels"]
        item["preds"] = {}
        all_label_num += len(labels)
        wrong = []
        for key, value in inf_result.items():
            pattern = r'"relation":\s*(\[[^\]]+\])'
            match = re.search(pattern, value)
            if match:
                relations = match.group(1)
            else:
                relations = "[]"
                print("No relation data found.")
            try:
                relations = json.loads(relations)
            except Exception as e:
                error += 1
                relations = []
                print("---------")
            real_label = labels[key]
            pred_set = set(relations)
            label_set = set(real_label)

            # 统计每个relation的TP、FP、FN、预测数量、标签数量
            for rel in pred_set:
                relation_stat[rel]['pred_count'] += 1
            for rel in label_set:
                relation_stat[rel]['label_count'] += 1
            for rel in pred_set & label_set:
                relation_stat[rel]['tp'] += 1
            for rel in pred_set - label_set:
                relation_stat[rel]['fp'] += 1
            for rel in label_set - pred_set:
                relation_stat[rel]['fn'] += 1

            # 统计混淆（仅在标签有但预测没有的关系上，统计预测中多出来的关系）
            for miss_rel in label_set - pred_set:
                for extra_rel in pred_set - label_set:
                    mis_predict_stat[miss_rel][extra_rel] += 1

            true = len(pred_set & label_set)
            item["preds"][key] = relations
            if true != len(label_set):  
                wrong.append({key: relations})
            all_true += true
            all_pred += len(pred_set)
            all_num += len(label_set)
            print(f'预测为{relations}')
            print(f'标签为{real_label}')
        if wrong:
            item["wrong"] = wrong
            save_wrong.append(item)

    # 总体指标
    precision = all_true / all_pred if all_pred > 0 else 0
    recall = all_true / all_num if all_num > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f'error = {error}')
    print(f'precision={precision}, recall={recall}, f1={f1_score}')
    print(f'all_num = {all_num}, all_pred = {all_pred}')
    print(f'all_true = {all_true}')
    print(f'all_label_num = {all_label_num}')
    print(f'wrong_num = {len(save_wrong)}')

    print("\n每个关系的详细统计：")
    print("relation\tlabel_count\tpred_count\ttp\tprecision\trecall\tf1")
    for rel, stat in relation_stat.items():
        tp = stat['tp']
        fp = stat['fp']
        fn = stat['fn']
        pred_count = stat['pred_count']
        label_count = stat['label_count']
        rel_precision = tp / pred_count if pred_count > 0 else 0
        rel_recall = tp / label_count if label_count > 0 else 0
        rel_f1 = 2 * rel_precision * rel_recall / (rel_precision + rel_recall) if (rel_precision + rel_recall) > 0 else 0
        print(f"{rel}\t{label_count}\t{pred_count}\t{tp}\t{rel_precision:.4f}\t{rel_recall:.4f}\t{rel_f1:.4f}")

    save_json(save_wrong, "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/lingshou/yanyi17/llm/result/ds_llama3_70b_wrong.json")
    # save_json(redoc_train_50_result, "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-hmart-waimaiad/lingshou/yanyi17/llm/result/ds_llama3_70b_pred.json")

    # ----------------- 贪心法自动选择纠正关系组合 -----------------
    rel_list = list(relation_stat.keys())
    tp = all_true
    fp = all_pred - all_true
    fn = all_num - all_true

    simulate_tp = tp
    simulate_fp = fp
    simulate_fn = fn

    selected = set()
    reached = False

    while True:
        best_f1 = -1
        best_rel = None
        for rel in rel_list:
            if rel in selected:
                continue
            rel_stat = relation_stat[rel]
            tmp_tp = simulate_tp + rel_stat['fp'] + rel_stat['fn']
            tmp_fp = simulate_fp - rel_stat['fp']
            tmp_fn = simulate_fn - rel_stat['fn']
            tmp_precision = tmp_tp / (tmp_tp + tmp_fp) if (tmp_tp + tmp_fp) > 0 else 0
            tmp_recall = tmp_tp / (tmp_tp + tmp_fn) if (tmp_tp + tmp_fn) > 0 else 0
            tmp_f1 = 2 * tmp_precision * tmp_recall / (tmp_precision + tmp_recall) if (tmp_precision + tmp_recall) > 0 else 0
            if tmp_f1 > best_f1:
                best_f1 = tmp_f1
                best_rel = rel
        if best_rel is None:
            break
        selected.add(best_rel)
        rel_stat = relation_stat[best_rel]
        simulate_tp += rel_stat['fp'] + rel_stat['fn']
        simulate_fp -= rel_stat['fp']
        simulate_fn -= rel_stat['fn']
        precision_sim = simulate_tp / (simulate_tp + simulate_fp) if (simulate_tp + simulate_fp) > 0 else 0
        recall_sim = simulate_tp / (simulate_tp + simulate_fn) if (simulate_tp + simulate_fn) > 0 else 0
        f1_sim = 2 * precision_sim * recall_sim / (precision_sim + recall_sim) if (precision_sim + recall_sim) > 0 else 0
        if f1_sim >= 0.83:
            print("\n贪心选择后，只需纠正这些关系即可让F1达到83%：")
            print(list(selected))
            print(f"模拟后的F1: {f1_sim:.4f}")
            reached = True
            break

    if not reached:
        print("\n即使用贪心法纠正所有关系，F1也达不到83%")

    # ----------------- 混淆统计及错误数量输出 -----------------
    print("\n每个关系被误判成的其他关系统计（只显示前3个）：")
    for real_rel, pred_dict in mis_predict_stat.items():
        sorted_confuse = sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)
        print(f"{real_rel} 被误判为：", end='')
        for pred_rel, count in sorted_confuse[:3]:
            print(f"{pred_rel}({count}) ", end='')
        print()

    print("\n每个关系被误判成其他关系的总次数：")
    for real_rel, pred_dict in mis_predict_stat.items():
        total_mis = sum(pred_dict.values())
        print(f"{real_rel} 总被误判次数: {total_mis}")

    print("\n模型最容易漏掉（FN最多）的关系TOP5：")
    sorted_fn = sorted(relation_stat.items(), key=lambda x: x[1]['fn'], reverse=True)
    for rel, stat in sorted_fn[:5]:
        print(f"{rel}: FN(漏掉)次数 = {stat['fn']}")

    # ----------------- 多召回一些后的F1提升 -----------------
    # 假设全部FN都召回（FN->TP，FP不变）
    tp = all_true
    fp = all_pred - all_true
    fn = all_num - all_true

    max_tp = tp + fn
    max_fp = fp
    max_fn = 0

    max_precision = max_tp / (max_tp + max_fp) if (max_tp + max_fp) > 0 else 0
    max_recall = 1.0
    max_f1 = 2 * max_precision * max_recall / (max_precision + max_recall) if (max_precision + max_recall) > 0 else 0

    print(f"\n如果能把所有漏掉的关系都召回，F1可以达到: {max_f1:.4f}")

    # 假设只多召回N个FN
    N = 200 # 你可以自定义
    add_tp = min(N, fn)
    new_tp = tp + add_tp
    new_fn = fn - add_tp
    new_fp = fp

    new_precision = new_tp / (new_tp + new_fp) if (new_tp + new_fp) > 0 else 0
    new_recall = new_tp / (new_tp + new_fn) if (new_tp + new_fn) > 0 else 0
    new_f1 = 2 * new_precision * new_recall / (new_precision + new_recall) if (new_precision + new_recall) > 0 else 0

    print(f"如果能多召回 {add_tp} 个FN，F1可以提升到: {new_f1:.4f}")

    # ----------------- 你特别关心的统计 -----------------
    # 1. 错误误判多少个
    total_fp = sum(stat['fp'] for stat in relation_stat.values())
    total_fn = sum(stat['fn'] for stat in relation_stat.values())
    print(f"\n总误判(预测有但标签没有)的数量: {total_fp}")
    print(f"总漏判(标签有但预测没有)的数量: {total_fn}")
    print(f"总错误（FP+FN）: {total_fp + total_fn}")

    # 2. 没有预测出来的有多少个
    print(f"\n没有预测出来(漏判)的总数: {total_fn}")

    # 3. 误判的一般误判成啥
    print("\n每个关系最常被误判成的关系：")
    for real_rel, pred_dict in mis_predict_stat.items():
        if not pred_dict:
            continue
        sorted_confuse = sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)
        top_pred, top_cnt = sorted_confuse[0]
        print(f"{real_rel} 最常被误判为 {top_pred}，次数: {top_cnt}")

    # 4. 没有预测是什么关系没预测出来
    print("\n漏判最多的关系TOP5：")
    sorted_fn = sorted(relation_stat.items(), key=lambda x: x[1]['fn'], reverse=True)
    for rel, stat in sorted_fn[:5]:
        print(f"{rel}: 漏判(FN)次数 = {stat['fn']}")
