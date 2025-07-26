# ── Structure viewer for graphs_data_int.pt ────────────────────────────
import torch, numpy as np, pprint

PT_PATH = "/graphs_data_int.pt"   
data = torch.load(PT_PATH, map_location="cpu", weights_only=False)

adjacency_list  = data["adjacency_matrices"]   # list[np.ndarray]
node_id_list    = data["node_types"]           # list[np.ndarray] (dtype int8 / int32)
label2id        = data["label2id"]             # dict[str, int]

# 2. Global overview
num_graphs = len(adjacency_list)
all_nodes  = sum(A.shape[0] for A in adjacency_list)
all_edges  = int(sum(A.sum() for A in adjacency_list))

print("── File summary ──────────────────────────────────────────────────")
print(f"Graphs          : {num_graphs}")
print(f"Total nodes     : {all_nodes}")
print(f"Total directed edges: {all_edges}")
print("\nLabel ↔ ID mapping:")
pprint.pp(label2id)



idx = 0
A   = adjacency_list[idx]
ids = node_id_list[idx]


#  Label mapping: {'ASSEMBLY': 0, 'BUFFER': 1, 'DISASSEMBLY': 2, 'MACHINE': 3}

print(f"\n── Detailed view: graph {idx} ────────────────────────────────")
print("Adjacency matrix:")
print(A)
print("\nNode-type ID array:")
print(ids)