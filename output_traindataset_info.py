import torch
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt

# data = torch.load("petri_assembly_dataset/processed/train_processed.pt")

file_path = "petri_machine_dataset/raw/train.pt"

data = torch.load(file_path)
print(type(data))
print(len(data))

# 如果是元组
if isinstance(data, tuple):
    print(type(data[0]), type(data[1]))


adjacency_data = data[0]  # 第0个元素
extra_info = data[1]      # 第1个元素，可选

print(adjacency_data)

dataset = data[0]
print(len(dataset))
