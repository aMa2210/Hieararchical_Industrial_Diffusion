import torch
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from torch_geometric.utils import to_networkx
import os

# Load generated graphs from final_graphs.pt
# graphs = torch.load('generated_graphs/final_graphs.pt')
graphs = torch.load('petri_net.pt')


print(type(graphs))

output_image_dir = 'petri_net_images'
os.makedirs(output_image_dir, exist_ok=True)

for idx, graph_data in enumerate(graphs):
    G = to_networkx(graph_data, to_undirected=False)

    node_labels = graph_data.x.argmax(dim=1).cpu().numpy()
    place_nodes = [i for i, label in enumerate(node_labels) if label == 0]
    transition_nodes = [i for i, label in enumerate(node_labels) if label == 1]

    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(10, 8))

    # Draw places (circles)
    nx.draw_networkx_nodes(G, pos,
                           nodelist=place_nodes,
                           node_shape='o',
                           node_color='#a6cee3',
                           edgecolors='black',
                           node_size=1000,
                           linewidths=1.5)

    # Draw transitions (thin rectangles)
    for node in transition_nodes:
        x, y = pos[node]
        plt.gca().add_patch(plt.Rectangle((x-0.025, y-0.07), 0.05, 0.14,
                                          edgecolor='black', facecolor='#fb9a99', linewidth=1.5, zorder=2))

    # Draw edges with clear arrows
    nx.draw_networkx_edges(
        G, pos,
        edge_color='gray',
        arrows=True,
        arrowstyle='-|>',
        arrowsize=20,
        connectionstyle='arc3,rad=0.05',
        node_size=1000
    )

    # Node labels
    nx.draw_networkx_labels(G, pos, font_size=12, font_color='black')

    # Custom legend clearly placed
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#a6cee3', markersize=15, markeredgecolor='black', label='Places'),
        Patch(facecolor='#fb9a99', edgecolor='black', label='Transitions')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    plt.title(f'Petri Net Graph {idx + 1}')
    plt.axis('off')

    image_path = os.path.join(output_image_dir, f'petri_net_{idx + 1}.png')
    # plt.savefig(image_path, bbox_inches='tight')
    plt.show()
    plt.close()

print(f'Successfully saved {len(graphs)} Petri net images to "{output_image_dir}".')
