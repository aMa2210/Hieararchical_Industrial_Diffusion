# ----------------------------------------------------------------------
# Generates the industrial graph + Petri subgraphs, stitches into a single
# Petri net, and saves PNGs in:
#   Stitched_Graph/
#     ├─ Industrial_Graph/   --> one PNG of the industrial graph (run_and_plot.py style)
#     ├─ Petri_Subgraph/     --> one PNG per non-Buffer node (filename has index)
#     └─ Stitched_Graph/     --> stitched .pt + PNG (Petri style)
# ----------------------------------------------------------------------

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from pathlib import Path
import datetime as dt

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.data import Data

from Industrial_Pipeline_Functions import LightweightIndustrialDiffusion
from Petri_Pipeline_Functions import LightweightPetriDiffusion
from Integrated_Pipeline_Functions import IntegratedDiffusionPipeline

# constants (keep your notation)
MACHINE, BUFFER, ASSEMBLY, DISASSEMBLY = 0, 1, 2, 3
TYPE_NAME = {MACHINE: "MACHINE", BUFFER: "BUFFER", ASSEMBLY: "ASSEMBLY", DISASSEMBLY: "DISASSEMBLY"}

# folder structure
ROOT_IMG = Path("Stitched_Graph")
DIR_INDUSTRIAL = ROOT_IMG / "Industrial_Graph"
DIR_PETRI      = ROOT_IMG / "Petri_Subgraph"
DIR_STITCHED   = ROOT_IMG / "Stitched_Graph"
for d in [ROOT_IMG, DIR_INDUSTRIAL, DIR_PETRI, DIR_STITCHED]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------- Petri/stitched drawing (exact your style) ----------------------
def _draw_petri_style(G, pos):
    # PLACE green circle, TRANS blue square, sizes/edge styles identical to your original
    place_nodes = [n for n, d in G.nodes(data=True) if d.get('kind') == 'PLACE']
    trans_nodes = [n for n, d in G.nodes(data=True) if d.get('kind') == 'TRANS']

    nx.draw_networkx_nodes(G, pos, nodelist=place_nodes,
                           node_color="#8BC34A", node_shape='o', node_size=600,
                           edgecolors="black", linewidths=1)
    nx.draw_networkx_nodes(G, pos, nodelist=trans_nodes,
                           node_color="#03A9F4", node_shape='s', node_size=600,
                           edgecolors="black", linewidths=1)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="black")
    plt.axis("off")

def draw_petri_subgraph(subgraph: dict, idx: int, node_type_name: str, save_dir: Path):
    nt = np.array(subgraph["node_types"])
    A  = np.array(subgraph["adjacency_matrix"])

    G = nx.DiGraph()
    # convention: 0 or -1 => PLACE, 1 => TRANS
    for i, t in enumerate(nt.tolist()):
        kind = "PLACE" if t in (0, -1) else "TRANS"
        G.add_node(i, kind=kind)
    r, c = np.where(A > 0)
    for u, v in zip(r.tolist(), c.tolist()):
        G.add_edge(u, v)

    pos = nx.spring_layout(G, seed=42 + int(idx)) 
    plt.figure(figsize=(6.0, 4.6), dpi=180)
    _draw_petri_style(G, pos)
    out = save_dir / f"petri_{node_type_name.lower()}_{idx}.png"
    plt.tight_layout()
    plt.savefig(out)
    plt.close()
    return out

def draw_stitched_graph(stitched: Data, save_png: Path):
    if stitched is None:
        raise ValueError("Stitched graph is None. Ensure stitch() returns a torch_geometric.data.Data.")

    edge_index = stitched.edge_index
    x = stitched.x.argmax(dim=1)  # 0=Place, 1=Transition

    G = nx.DiGraph()
    for i, t in enumerate(x.tolist()):
        kind = 'PLACE' if t == 0 else 'TRANS'
        G.add_node(i, kind=kind)
    for u, v in edge_index.t().tolist():
        G.add_edge(u, v)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(7.0, 5.4), dpi=180)
    _draw_petri_style(G, pos)
    plt.tight_layout()
    plt.savefig(save_png)
    plt.close()

# ---------------------- Industrial drawing (run_and_plot.py style) ----------------------
_NODE_STYLES = {
    MACHINE:     {'marker': 's', 'color': 'blue',   'label': 'MACHINE'},
    BUFFER:      {'marker': 'v', 'color': 'red',    'label': 'BUFFER'},
    ASSEMBLY:    {'marker': 'D', 'color': 'green',  'label': 'ASSEMBLY'},
    DISASSEMBLY: {'marker': 'h', 'color': 'orange', 'label': 'DISASSEMBLY'},
}

def draw_industrial_graph(node_types: np.ndarray, adj: np.ndarray, save_to: Path):
    # Build directed graph from adjacency
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))

    handles = []
    for ntype, props in _NODE_STYLES.items():
        nodelist = [idx for idx, t in enumerate(node_types.tolist()) if t == ntype]
        if not nodelist:
            continue
        col = nx.draw_networkx_nodes(
            G, pos, nodelist=nodelist,
            node_shape=props['marker'],
            node_color=props['color'],
            node_size=400, label=props['label']
        )
        handles.append(col)

    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrows=True, arrowsize=12)
    nx.draw_networkx_labels(G, pos, font_size=9)
    plt.title("Industrial Graph")
    plt.axis('off')
    if handles:
        plt.legend(scatterpoints=1)

    plt.savefig(save_to, bbox_inches='tight', dpi=150)
    plt.close()

# --------------------------------------------- MAIN ---------------------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load models/weights
    plant_model = LightweightIndustrialDiffusion().to(device).eval()
    petri_models = {
        MACHINE:     LightweightPetriDiffusion().to(device).eval(),
        ASSEMBLY:    LightweightPetriDiffusion().to(device).eval(),
        DISASSEMBLY: LightweightPetriDiffusion().to(device).eval(),
    }
    plant_model.load_state_dict(torch.load('industrial_model.pth', map_location=device))
    petri_models[MACHINE].load_state_dict(torch.load('petri_machine_model.pth', map_location=device))
    petri_models[ASSEMBLY].load_state_dict(torch.load('petri_assembly_model.pth', map_location=device))
    petri_models[DISASSEMBLY].load_state_dict(torch.load('petri_disassembly_model.pth', map_location=device))

    pipeline = IntegratedDiffusionPipeline(plant_model, petri_models, device)

    # params
    n_nodes_global = 10
    n_nodes_petri  = 6
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    # build integrated dict (one sample)
    integ = pipeline.generate_full_integrated_graph(
        n_nodes_global=n_nodes_global,
        n_nodes_petri=n_nodes_petri,
        output_dir='integrated_graph'
    )

    # save copy near stitched outputs
    tmp_pt = DIR_STITCHED / f"graphs_data_{ts}.pt"
    torch.save(integ, tmp_pt)

    # industrial image (single, exact run_and_plot.py style)
    global_node_types = np.array(integ["global_graph"]["node_types"])
    global_adj        = np.array(integ["global_graph"]["adjacency_matrix"])
    industrial_png = DIR_INDUSTRIAL / f"industrial_graph_{ts}.png"
    draw_industrial_graph(global_node_types, global_adj, industrial_png)
    print(f"✓ Industrial graph image -> {industrial_png}")

    # petri subgraphs (one per non-Buffer), filenames include industrial idx
    count = 0
    for idx, sub in integ["petri_subgraphs"].items():
        # skip trivial buffer subgraph [-1]
        if len(sub["node_types"]) == 1 and sub["node_types"][0] == -1:
            continue
        node_type = int(global_node_types[idx])
        if node_type == BUFFER:
            continue
        _ = draw_petri_subgraph(sub, idx, TYPE_NAME[node_type], DIR_PETRI)
        count += 1
    print(f"✓ Petri subgraph images -> {count} files in {DIR_PETRI}")

    # stitched (.pt + PNG) in Petri style/notation
    stitched_pt_path  = DIR_STITCHED / f"stitched_{ts}.pt"
    stitched_png_path = DIR_STITCHED / f"stitched_{ts}.png"

    stitched_data = pipeline.stitch(
        path=str(tmp_pt),
        save_path=str(stitched_pt_path),
        device=device
    )
    draw_stitched_graph(stitched_data, stitched_png_path)
    print(f"✓ Stitched graph saved -> {stitched_pt_path}")
    print(f"✓ Stitched graph image -> {stitched_png_path}")

    print("\nAll done ✓")


if __name__ == "__main__":
    main()
