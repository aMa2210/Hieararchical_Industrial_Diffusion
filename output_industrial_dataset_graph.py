import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

file_path = "industrial_dataset/raw/graphs_data.pt"
data = torch.load(file_path)

adjacency_data = data['adjacency_matrices']
node_types = data['node_types']

# 取第0个图
adj = adjacency_data[0]
G = nx.from_numpy_array(adj)

# node_types[0] 是一个 list，里面是字符串
types = node_types[0]
print(types)  # 打印前10个看看是什么

color_map = {
    "MACHINE": "skyblue",
    "ASSEMBLY": "deeppink",
    "DISASSEMBLY": "green",
    "BUFFER": "orange"
}

colors = [color_map[t] for t in types]

unique_types = list(set(types))
type_to_id = {t: i for i, t in enumerate(unique_types)}


plt.figure(figsize=(6,6))
nx.draw(G, with_labels=True, node_color=colors)
plt.show()
