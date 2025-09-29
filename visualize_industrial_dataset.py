import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# 配置
# -----------------------------
file_path = 'industrial_dataset/raw/graphs_data.pt'
# file_path = 'ablation_runs_new/no_projector/samples.pt'
max_graphs_to_show = 10  # max visualization data

node_styles = {
    0: {'marker': 's', 'color': 'blue', 'label': 'MACHINE'},
    1: {'marker': 'v', 'color': 'red', 'label': 'BUFFER'},
    2: {'marker': 'D', 'color': 'green', 'label': 'ASSEMBLY'},
    3: {'marker': 'h', 'color': 'orange', 'label': 'DISASSEMBLY'},
}


# -----------------------------
# 加载图数据
# -----------------------------
data = torch.load(file_path)
# print(type(data), len(data))
# print(data[0])

# print(type(data))
# print(data.keys() if isinstance(data, dict) else len(data))
# adj_matrices = data["adjacency_matrices"]
# node_types   = data["node_types"]
# label2id     = data["label2id"]
# id2label     = {v:k for k,v in label2id.items()}
node_types   = [nt.numpy() if isinstance(nt, torch.Tensor) else np.array(nt)
                for nt in data["node_types"]]
adj_matrices = [am.numpy() if isinstance(am, torch.Tensor) else np.array(am)
                for am in data["adjacency_matrices"]]

print(f"Loaded {len(adj_matrices)} graphs from {file_path}")

# -----------------------------
# 可视化循环
# -----------------------------
num_graphs = min(len(adj_matrices), max_graphs_to_show)
for i in range(num_graphs):
    A = adj_matrices[i]            # 邻接矩阵
    types = node_types[i]          # 节点类型向量

    # 转为 NetworkX 图
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)

    # 节点颜色
    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(8, 6))

    # 按节点类型绘制
    for ntype, props in node_styles.items():
        label = props['label']
        nodelist = [idx for idx, t in enumerate(types) if t == label]
        nx.draw_networkx_nodes(G, pos,
                               nodelist=nodelist,
                               node_shape=props['marker'],
                               node_color=props['color'],
                               node_size=400,
                               label=props['label'])

    # 绘制边
    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=12)
    nx.draw_networkx_labels(G, pos)

    plt.title(f"Industrial Graph {i}")
    plt.axis('off')
    plt.legend(scatterpoints=1)

    # 保存图片
    # output_path = os.path.join(output_folder, f"industrial_graph_{i}.png")
    # plt.savefig(output_path, bbox_inches='tight')
    plt.show()
    plt.close()
