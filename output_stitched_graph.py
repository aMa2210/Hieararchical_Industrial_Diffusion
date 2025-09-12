import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data

# 1️⃣ 加载保存的 pt 文件
stitched = torch.load("stitched_graph.pt")

# 2️⃣ 构建 NetworkX 图
# edge_index 是 shape [2, num_edges] 的 tensor
edge_index = stitched.edge_index
G = nx.DiGraph()  # 有向图

# 添加节点
num_nodes = stitched.x.size(0)
G.add_nodes_from(range(num_nodes))

# 添加边
edges = edge_index.t().tolist()  # 转成 list of [src, dst]
G.add_edges_from(edges)

# 3️⃣ 可选：根据节点类型设置颜色
# 假设 stitched.x 的每行是 one-hot 类型：PLACE=0, TRANS=1
node_types = stitched.x.argmax(dim=1).tolist()
colors = ['lightblue' if t == 0 else 'lightgreen' for t in node_types]

# 4️⃣ 可视化
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)  # 布局
nx.draw(G, pos, with_labels=True, node_color=colors, node_size=500, arrowsize=20)
plt.show()
