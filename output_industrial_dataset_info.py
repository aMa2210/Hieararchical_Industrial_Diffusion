import torch
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt

# data = torch.load("petri_assembly_dataset/processed/train_processed.pt")

file_path = "industrial_dataset/raw/graphs_data.pt"
# file_path = "petri_machine_dataset/raw/train.pt"

data = torch.load(file_path)
print(type(data))
print(len(data))
print(data.keys())

adjacency_data = data['adjacency_matrices']
extra_info = data['node_types']

print(type(adjacency_data), len(adjacency_data))
print(type(extra_info), len(extra_info))

