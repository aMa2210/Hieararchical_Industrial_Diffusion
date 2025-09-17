import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

file_path = "industrial_dataset/raw/graphs_data.pt"
data = torch.load(file_path)

max_graphs_to_show = 10

adjacency_data = data['adjacency_matrices']
node_types = data['node_types']

# 定义节点样式
node_styles = {
    "MACHINE": {'marker': 's', 'color': 'skyblue', 'label': 'MACHINE'},
    "BUFFER": {'marker': 'v', 'color': 'orange', 'label': 'BUFFER'},
    "ASSEMBLY": {'marker': 'D', 'color': 'deeppink', 'label': 'ASSEMBLY'},
    "DISASSEMBLY": {'marker': 'h', 'color': 'green', 'label': 'DISASSEMBLY'},
}


for i in range(min(max_graphs_to_show, len(adjacency_data))):
    adj = adjacency_data[i]
    types = node_types[i]  # list of strings

    G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(8,6))

    # 按节点类型绘制
    for ntype, props in node_styles.items():
        nodelist = [idx for idx, t in enumerate(types) if t == ntype]
        nx.draw_networkx_nodes(G, pos,
                               nodelist=nodelist,
                               node_shape=props['marker'],
                               node_color=props['color'],
                               node_size=400,
                               label=props['label'])

    # 绘制边和节点标签
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=12)
    nx.draw_networkx_labels(G, pos)

    plt.title(f"Industrial Graph {i}")
    plt.axis('off')
    plt.legend(scatterpoints=1)
    plt.show()
    plt.close()
