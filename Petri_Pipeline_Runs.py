# Petri_Pipeline_Runs.py
# Simple runner: edit the config below and just run `python Petri_Pipeline_Runs.py`.
# It loads your trained weights, generates graphs, and ALWAYS renders PNGs.

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch

# Use your model class exactly as defined in your codebase
from Petri_Pipeline_Functions import LightweightPetriDiffusion

# --- only needed for rendering ---
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from torch_geometric.utils import to_networkx
# ---------------------------------


# ===================== EASY CONFIG =====================
NET_TYPE     = "disassembly"     # "machine" | "assembly" | "disassembly"
N_SAMPLES    = 8              # how many graphs to generate
N_NODES      = 8             # nodes per generated graph
OUT_DIR      = "exp_outputs"  # where to save final_graphs.pt + intermediate_steps.pt
IMAGE_DIR    = 'petri'           # None -> <OUT_DIR>/images ; or set a custom folder path
SHOW         = False          # show windows while saving PNGs (usually False on servers)
WEIGHTS_PATH = None           # None -> uses f"petri_{NET_TYPE}_model.pth"
# =======================================================


def render_graphs_pt(pt_path: str, out_img_dir: str, show: bool = False) -> None:
    """
    Render each PyG Data graph in pt_path (list[Data]) into a PNG.
    Places = circles (class 0), Transitions = thin rectangles (class 1).
    Saves as petri_graph_<i>.png in out_img_dir.
    """
    graphs = torch.load(pt_path, map_location="cpu")
    os.makedirs(out_img_dir, exist_ok=True)

    for idx, data in enumerate(graphs):
        # Convert to networkx directed graph
        G = to_networkx(data, to_undirected=False)

        # Node types: 0 = Place, 1 = Transition
        node_labels = data.x.argmax(dim=1).cpu().numpy()
        place_nodes = [i for i, lbl in enumerate(node_labels) if lbl == 0]
        transition_nodes = [i for i, lbl in enumerate(node_labels) if lbl == 1]

        # Layout (deterministic)
        pos = nx.spring_layout(G, seed=42)

        fig, ax = plt.subplots(figsize=(10, 8))

        # Draw places (circles)
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=place_nodes,
            node_shape='o',
            node_color='#a6cee3',
            edgecolors='black',
            node_size=1000,
            linewidths=1.5,
            ax=ax
        )

        # Draw transitions (thin rectangles)
        for n in transition_nodes:
            x, y = pos[n]
            ax.add_patch(Rectangle(
                (x - 0.025, y - 0.07), 0.05, 0.14,
                edgecolor='black',
                facecolor='#fb9a99',
                linewidth=1.5,
                zorder=2
            ))

        # Draw directed edges with arrows
        nx.draw_networkx_edges(
            G, pos,
            edge_color='gray',
            arrows=True,
            arrowstyle='-|>',
            arrowsize=20,
            connectionstyle='arc3,rad=0.05',
            node_size=1000,
            ax=ax
        )

        # Labels
        nx.draw_networkx_labels(G, pos, font_size=12, font_color='black', ax=ax)

        # Legend
        legend_elements = [
            Patch(facecolor='#a6cee3', edgecolor='black', label='Places'),
            Patch(facecolor='#fb9a99', edgecolor='black', label='Transitions')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        ax.set_title(f'Petri Net Graph {idx + 1}')
        ax.axis('off')

        out_path = os.path.join(out_img_dir, f'petri_graph_{idx + 1}.png')
        fig.savefig(out_path, bbox_inches='tight', dpi=150)
        if show:
            plt.show()
        plt.close(fig)

    print(f'✅ Saved {len(graphs)} Petri net images to "{out_img_dir}".')


@torch.no_grad()
def run():
    # Prepare folders
    os.makedirs(OUT_DIR, exist_ok=True)
    img_dir = IMAGE_DIR or os.path.join(OUT_DIR, "images")
    os.makedirs(img_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build your model with the same hyper-params used in training
    model = LightweightPetriDiffusion(T=100, hidden_dim=12, time_embed_dim=16).to(device)

    # Resolve weights path
    weights_path = WEIGHTS_PATH or f"petri_{NET_TYPE}_model.pth"
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Weights not found: {weights_path}\n"
            f"• Train first (producing {weights_path}) or set WEIGHTS_PATH to the correct file."
        )

    # Load weights
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Generate graphs
    model.sample_conditional_and_save(
        n_nodes=N_NODES,
        batch_size=N_SAMPLES,
        device=device,
        output_dir=OUT_DIR,
    )

    # Always render images
    pt_path = os.path.join(OUT_DIR, "final_graphs.pt")
    render_graphs_pt(pt_path, img_dir, show=SHOW)


if __name__ == "__main__":
    run()
