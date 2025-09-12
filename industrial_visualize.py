import torch
import networkx as nx
import matplotlib.pyplot as plt
import os

# Carga los grafos generados
final_graphs = torch.load('allpinned_graphs/final_graphs.pt')

# final_graphs = torch.load("exp_outputs/E2_20250912_113248.pt")
# Crear carpeta para guardar imágenes
output_folder = 'industrial_graph_images_tmp'
os.makedirs(output_folder, exist_ok=True)

# Definir estilos por tipo de nodo
node_styles = {
    0: {'marker': 's', 'color': 'blue', 'label': 'MACHINE'},
    1: {'marker': 'v', 'color': 'red', 'label': 'BUFFER'},
    2: {'marker': 'D', 'color': 'green', 'label': 'ASSEMBLY'},
    3: {'marker': 'h', 'color': 'orange', 'label': 'DISASSEMBLY'},
}

for idx, data in enumerate(final_graphs):
    G = nx.DiGraph()

    # Añadir nodos con tipos específicos
    node_labels = data.x.argmax(dim=1).tolist()
    for i, label in enumerate(node_labels):
        G.add_node(i, type=label)

    # Añadir aristas
    edge_index = data.edge_index.cpu().numpy()
    for src, dst in edge_index.T:
        G.add_edge(src, dst)

    # Posiciones de nodos
    pos = nx.spring_layout(G, seed=42)

    # Dibujar grafo
    plt.figure(figsize=(8, 6))

    for ntype, props in node_styles.items():
        nodelist = [n for n, d in G.nodes(data=True) if d['type'] == ntype]
        nx.draw_networkx_nodes(G, pos,
                               nodelist=nodelist,
                               node_shape=props['marker'],
                               node_color=props['color'],
                               node_size=400,
                               label=props['label'])

    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrowsize=12)

    plt.title(f"Industrial Graph {idx}")
    plt.axis('off')
    plt.legend(scatterpoints=1)

    # Guardar imagen
    output_path = os.path.join(output_folder, f"industrial_graph_{idx}.png")
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

print(f"✅ Imágenes guardadas en la carpeta '{output_folder}'")
