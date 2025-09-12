import torch
import networkx as nx
import matplotlib.pyplot as plt

# 读取数据
data = torch.load('petri_net.pt')[0]
nodes = data['nodes'].numpy()
edges = data['edges'].squeeze(0).numpy()  # 去掉 batch 维度

n_nodes = len(nodes)
G = nx.DiGraph()

# 添加节点
for i in range(n_nodes):
    G.add_node(i, type=nodes[i])

# 添加边
for i in range(n_nodes):
    for j in range(n_nodes):
        if edges[i, j] != 0:
            G.add_edge(i, j)

# 可视化
plt.figure(figsize=(8, 6))
node_colors = ['lightblue' if t == 0 else 'lightgreen' for t in nodes]
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=800, arrowsize=20)
plt.show()
