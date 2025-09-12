import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 配置
# -----------------------------
# file_path = "petri_machine_dataset/raw/train.pt"
file_path = "industrial_dataset/raw/graphs_data.pt"

max_graphs_to_show = 10

# 节点类型颜色映射
color_map = {
    "Place": "red",
    "Transition": "blue"
}

# -----------------------------
# 加载数据
# -----------------------------
data = torch.load(file_path)
print(f"Total graphs in dataset: {len(data)}")

# -----------------------------
# 可视化循环
# -----------------------------
num_graphs = min(len(data), max_graphs_to_show)
for i in range(num_graphs):
    g = data[i]

    adj = g["adjacency_matrix"]       # 邻接矩阵 (numpy array)
    node_types_dict = g["node_types"] # 节点类型字典
    nodes_order = g["nodes_order"]    # 节点顺序列表

    # 构建 NetworkX 图
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph)  # 有向图

    # 节点颜色
    colors = [color_map[node_types_dict[n]] for n in nodes_order]

    # 布局和绘图
    plt.figure(figsize=(6,6))
    pos = nx.spring_layout(G, seed=42)  # 固定布局，便于对比
    nx.draw(G, pos, with_labels=True, labels={idx: n for idx, n in enumerate(nodes_order)},
            node_color=colors, node_size=500, arrows=True)
    plt.title(f"Graph {i}")
    plt.show()
