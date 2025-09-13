import torch
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

file_path = "industrial_dataset/raw/graphs_data.pt"
data = torch.load(file_path)

adjacency_data = data['adjacency_matrices']
node_types = data['node_types']


adj = adjacency_data[0]
G = nx.from_numpy_array(adj)


types = node_types[0]
print(types)

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
