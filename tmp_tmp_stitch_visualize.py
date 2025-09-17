import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data

def visualize_stitched(stitched: Data):
    edge_index = stitched.edge_index
    x = stitched.x.argmax(dim=1)  # 0=Place, 1=Transition

    G = nx.DiGraph()
    for i, t in enumerate(x.tolist()):
        node_type = 'PLACE' if t == 0 else 'TRANS'
        G.add_node(i, type=node_type)

    for u, v in edge_index.t().tolist():
        G.add_edge(u, v)

    pos = nx.spring_layout(G, seed=42)

    # 分开画两类节点
    place_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'PLACE']
    trans_nodes = [n for n, d in G.nodes(data=True) if d['type'] == 'TRANS']

    nx.draw_networkx_nodes(G, pos, nodelist=place_nodes,
                           node_color="#8BC34A", node_shape='o', node_size=600,
                           edgecolors="black", linewidths=1)
    nx.draw_networkx_nodes(G, pos, nodelist=trans_nodes,
                           node_color="#03A9F4", node_shape='s', node_size=600,
                           edgecolors="black", linewidths=1)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="black")

    plt.axis("off")
    plt.show()


save_path = "./stitched_graph22.pt"
stitched = torch.load(save_path, map_location="cpu")
visualize_stitched(stitched)