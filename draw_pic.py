import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# --------------------------
# ACL-Specific Styling Setup
# --------------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

# --------------------------
# 仅保留长尾关系（label_count ≤ 10）+ 数据严格对应
# --------------------------
data = pd.DataFrame([
    # 长尾关系：label_count ≤ 10，有升有降
    ("replaces", 1, 0.0000, 0.2900),
    ("replaced by", 1, 0.0000, 0.3300),
    ("place of death", 1, 0.6667, 0.6867),
    ("position held", 1, 0.0000, 0.0000),
    ("subclass of", 1, 0.0000, 0.2222),
    ("territory claimed by", 1, 0.0000, 0.0000),
    ("chairperson", 1, 1.0000, 1.0000),
    ("location of formation", 1, 0.2222, 0.3822),
    ("genre", 1, 0.0000, 0.0000),
    ("work location", 2, 0.5714, 0.7414),
    ("continent", 2, 0.2857, 0.4757),
    ("cast member", 2, 0.8000, 0.8500),
    ("characters", 2, 0.8000, 0.7700),
    ("production company", 2, 0.6667, 0.6967),
    ("founded by", 2, 0.4000, 0.5500),
    ("residence", 2, 0.6667, 0.6867),
    ("mouth of the watercourse", 2, 0.2500, 0.4257),
    ("ethnic group", 2, 0.0000, 0.1900),
    ("author", 2, 0.6667, 0.6867),
    ("end time", 3, 0.5000, 0.6900),
    ("award received", 3, 0.8571, 0.9271),
    ("instance of", 3, 0.4444, 0.6544),
    ("creator", 3, 0.4000, 0.5800),
    ("languages spoken, written or signed", 3, 0.2222, 0.4122),
    ("employer", 3, 0.8000, 0.8300),
    ("product or material produced", 3, 0.7500, 0.7700),
    ("lyrics by", 4, 0.5000, 0.5200),
    ("producer", 4, 0.8889, 0.9089),
    ("head of state", 4, 0.4000, 0.3700),
    ("start time", 5, 0.5000, 0.5700),
    ("headquarters location", 5, 0.1667, 0.1467),
    ("manufacturer", 5, 0.7500, 0.7300),
    ("point in time", 6, 0.6000, 0.6800),
    ("member of sports team", 6, 0.5714, 0.6314),
    ("located on terrain feature", 6, 0.1538, 0.1838),
    ("operator", 6, 0.7273, 0.7473),
    ("dissolved, abolished or demolished", 6, 0.6667, 0.6867),
    ("conflict", 8, 0.6667, 0.6367),
    ("official language", 8, 0.5600, 0.5300),
    ("basin country", 8, 0.7368, 0.7568),
    ("location", 10, 0.4000, 0.4700),
    ("legislative body", 10, 0.9091, 0.9191),
    ("capital", 10, 0.5000, 0.5300),
    ("capital of", 10, 0.0000, 0.0000),
    ("publisher", 9, 0.7778, 0.7878),
    ("member of political party", 9, 0.6923, 0.6723),
], columns=["relation", "label_count", "F1_without_RPD", "F1_with_RPD"])

# 按 label_count 升序排列（左侧=更稀缺的长尾关系）
data = data.sort_values("label_count").reset_index(drop=True)

# --------------------------
# 创建聚焦长尾关系的图表（无数值标签）
# --------------------------
fig, ax = plt.subplots(figsize=(20, 9))  # 适配45个长尾关系
x = np.arange(len(data))
bar_width = 0.35

# 绘制柱状图（ACL合规颜色，对比清晰）
bars1 = ax.bar(x - bar_width/2, data["F1_without_RPD"], bar_width,
               label="Without RPD", color="#e74c3c", alpha=0.8, edgecolor="black", linewidth=0.5)
bars2 = ax.bar(x + bar_width/2, data["F1_with_RPD"], bar_width,
               label="With RPD", color="#3498db", alpha=0.8, edgecolor="black", linewidth=0.5)

# --------------------------
# 图表样式优化（简洁、专业）
# --------------------------
# ax.set_xlabel("Impact of RPD Modeling on Long-Tail Relation Extraction", fontweight="bold", fontsize=13)
ax.set_ylabel("F1 Score", fontweight="bold", fontsize=13)
# ax.set_title("Impact of RPD Modeling on Long-Tail Relation Extraction (Label Count ≤ 10)",
#              fontweight="bold", pad=25, fontsize=15)

# 标签显示优化（每隔1个显示，避免重叠）
ax.set_xticks(x[::1])
ax.set_xticklabels(data["relation"].iloc[::1], rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
ax.set_ylim(0.0, 1.05)  # 适配F1分数范围

# 图例位置优化
ax.legend(loc="upper right", frameon=True, shadow=False, framealpha=1.0, fontsize=12)

# 弱化网格线（不干扰主体）
ax.yaxis.grid(True, linestyle="--", alpha=0.3, color="black")
ax.set_axisbelow(True)

# --------------------------
# 保存文件（ACL合规格式）
# --------------------------
plt.tight_layout()
plt.savefig("rpd_longtail_only_comparison.pdf", format="pdf", bbox_inches="tight")
plt.savefig("rpd_longtail_only_comparison.png", format="png", bbox_inches="tight")
plt.close()

print("✅ 聚焦长尾关系的图表保存成功！仅包含 label_count ≤ 10 的关系，无数值标签")