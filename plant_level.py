# ── Structure viewer for graphs_data_int.pt ────────────────────────────
import torch, numpy as np, pprint
from collections import Counter

PT_PATH = "graphs_data.pt"   
data = torch.load(PT_PATH, map_location="cpu", weights_only=False)

adjacency_list  = data["adjacency_matrices"]   # list[np.ndarray]
node_id_list    = data["node_types"]           # list[np.ndarray] (dtype int8 / int32)
print(f"Available keys in data: {data.keys()}")
# label2id        = data["label2id"]             # dict[str, int] # This line will be removed
label2id = {'ASSEMBLY': 0, 'BUFFER': 1, 'DISASSEMBLY': 2, 'MACHINE': 3} # Added hardcoded mapping
id2label = {v: k for k, v in label2id.items()}

# 2. Global overview
num_graphs = len(adjacency_list)
all_nodes  = sum(A.shape[0] for A in adjacency_list)
all_edges  = int(sum(A.sum() for A in adjacency_list))

print("── File summary ──────────────────────────────────────────────────")
print(f"Graphs          : {num_graphs}")
print(f"Total nodes     : {all_nodes}")
print(f"Total directed edges: {all_edges}")
print(f"Average nodes per graph: {all_nodes/num_graphs:.2f}")
print(f"Average edges per graph: {all_edges/num_graphs:.2f}")
print("\nLabel ↔ ID mapping:")

pprint.pp(label2id) # Re-adding pprint for the hardcoded dict

# Analyze node type distribution across all graphs
all_node_types = []
for nodes in node_id_list:
    all_node_types.extend(nodes)
    
node_type_counts = Counter(all_node_types)
print("\n── Node type distribution across all graphs ──────────────────────")
for node_type, count in node_type_counts.items():
    print(f"{node_type}: {count} ({count/len(all_node_types)*100:.1f}%)")

# Graph size distribution
graph_sizes = [len(nodes) for nodes in node_id_list]
print("\n── Graph size distribution ──────────────────────────────────────")
print(f"Min nodes: {min(graph_sizes)}")
print(f"Max nodes: {max(graph_sizes)}")
print(f"Mean nodes: {np.mean(graph_sizes):.2f}")
print(f"Median nodes: {np.median(graph_sizes)}")

# Edge density distribution
edge_counts = [int(A.sum()) for A in adjacency_list]
print("\n── Edge count distribution ──────────────────────────────────────")
print(f"Min edges: {min(edge_counts)}")
print(f"Max edges: {max(edge_counts)}")
print(f"Mean edges: {np.mean(edge_counts):.2f}")
print(f"Median edges: {np.median(edge_counts)}")

# Show details for a few sample graphs
sample_indices = [0, num_graphs//3, 2*num_graphs//3, num_graphs-1]
for i, idx in enumerate(sample_indices):
    if i > 0:  # Add separator between graphs
        print("\n" + "─" * 60)
    
    A = adjacency_list[idx]
    nodes = node_id_list[idx]
    
    # Convert numeric node types to labels if needed
    if isinstance(nodes[0], (int, np.integer)):
        node_labels = [id2label.get(int(n), f"Unknown-{n}") for n in nodes]
    else:
        node_labels = nodes
    
    node_type_counts = Counter(node_labels)
    
    print(f"\n── Sample graph {idx} ──────────────────────────────────────")
    print(f"Nodes: {len(nodes)}")
    print(f"Edges: {int(A.sum())}")
    print(f"Edge density: {A.sum() / (len(nodes)**2):.4f}")
    
    print("\nNode type counts:")
    for node_type, count in node_type_counts.items():
        print(f"  {node_type}: {count}")
    
    # Print a more compact adjacency matrix for larger graphs
    print("\nAdjacency matrix:")
    if len(nodes) > 15:
        # For large matrices, print a summary
        print(f"Shape: {A.shape}")
        print("First 5x5 submatrix:")
        print(A[:5, :5])
        print("...")
    else:
        print(A)
    
    # Print first few nodes with their types
    print("\nFirst few nodes with types:")
    for i, (node_type, connections) in enumerate(zip(node_labels, A)):
        if i >= 5:  # Limit to first 5 nodes
            print("...")
            break
        out_connections = np.where(connections > 0)[0]
        in_connections = np.where(A[:, i] > 0)[0]
        print(f"Node {i} ({node_type}):")
        print(f"  Outgoing connections to: {out_connections.tolist()}")
        print(f"  Incoming connections from: {in_connections.tolist()}")