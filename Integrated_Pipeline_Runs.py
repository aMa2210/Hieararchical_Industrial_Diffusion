import os
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

# Costanti (mantieni la tua notazione)
MACHINE, BUFFER, ASSEMBLY, DISASSEMBLY = 0, 1, 2, 3
TYPE_NAME = {MACHINE: "MACHINE", BUFFER: "BUFFER", ASSEMBLY: "ASSEMBLY", DISASSEMBLY: "DISASSEMBLY"}

seed_local = 45

# Creazione della cartella principale "Stitched_Graph" se non esiste
ROOT_IMG = Path("Stitched_Graph")
if not ROOT_IMG.exists():
    ROOT_IMG.mkdir(parents=True)

# Crea la cartella con il timestamp all'interno di "Stitched_Graph"
timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
run_dir = ROOT_IMG / timestamp
run_dir.mkdir(parents=True, exist_ok=True)

# Crea le sottocartelle richieste
DIR_INDUSTRIAL = run_dir / "Industrial_Graph"
DIR_PETRI      = run_dir / "Petri_Subgraph"
DIR_STITCHED   = run_dir / "Stitched_Graph"
for d in [DIR_INDUSTRIAL, DIR_PETRI, DIR_STITCHED]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------- Petri/stitched drawing (exact your style) ----------------------
def _draw_petri_style(G, pos):
    # Disegna il grafico Petri secondo lo stile richiesto
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
    for i, t in enumerate(nt.tolist()):
        kind = "PLACE" if t in (0, -1) else "TRANS"
        G.add_node(i, kind=kind)
    r, c = np.where(A > 0)
    for u, v in zip(r.tolist(), c.tolist()):
        G.add_edge(u, v)

    pos = nx.spring_layout(G, seed=seed_local)
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

    pos = nx.spring_layout(G, seed=seed_local)
    plt.figure(figsize=(7.0, 5.4), dpi=180)
    _draw_petri_style(G, pos)
    plt.tight_layout()
    plt.savefig(save_png)
    plt.close()

# ---------------------- Industrial drawing (run_and_plot.py style) ----------------------
_NODE_STYLES = {
    0: {'marker': 's', 'color': 'blue',   'label': 'MACHINE'},
    1: {'marker': 'v', 'color': 'red',    'label': 'BUFFER'},
    2: {'marker': 'D', 'color': 'green',  'label': 'ASSEMBLY'},
    3: {'marker': 'h', 'color': 'orange', 'label': 'DISASSEMBLY'},
}

def _normalize_node_types(arr):
    arr = np.asarray(arr)
    if arr.ndim == 2 and arr.shape[1] in (2, 3, 4):
        return arr.argmax(axis=1).astype(int)
    if arr.dtype.kind in {'U', 'S', 'O'}:
        m = {'MACHINE':0, 'BUFFER':1, 'ASSEMBLY':2, 'DISASSEMBLY':3}
        return np.array([m[str(x).upper()] for x in arr], dtype=int)
    return arr.astype(int)

def draw_industrial_graph(node_types: np.ndarray, adj: np.ndarray, save_to: Path):
    G = nx.from_numpy_array(adj, create_using=nx.DiGraph)
    pos = nx.spring_layout(G, seed=seed_local)
    plt.figure(figsize=(8, 6))

    handles, labels = [], []

    for ntype, props in _NODE_STYLES.items():
        nodelist = [idx for idx, t in enumerate(node_types.tolist()) if int(t) == ntype]
        if not nodelist:
            continue
        coll = nx.draw_networkx_nodes(
            G, pos, nodelist=nodelist,
            node_shape=props['marker'],
            node_color=props['color'],
            node_size=500
        )
        handles.append(coll)
        labels.append(props['label'])

    nx.draw_networkx_edges(G, pos, arrowstyle='-|>', arrows=True, arrowsize=12, width=1.2)
    nx.draw_networkx_labels(G, pos, font_size=9)
    plt.title("Industrial Graph")
    plt.axis('off')

    if handles:
        plt.legend(handles=handles, labels=labels, scatterpoints=1, loc="best")

    # Salvataggio nella cartella timestamp senza timestamp nei nomi dei file
    industrial_png_path = DIR_INDUSTRIAL / "industrial_graph.png"
    plt.savefig(industrial_png_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"✓ Industrial graph image saved in: {run_dir}")

# --------------------------------------------- MAIN ---------------------------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Carica modelli/pesi
    plant_model = LightweightIndustrialDiffusion().to(device).eval()
    petri_models = {
        MACHINE:     LightweightPetriDiffusion().to(device).eval(),
        ASSEMBLY:    LightweightPetriDiffusion().to(device).eval(),
        DISASSEMBLY: LightweightPetriDiffusion().to(device).eval(),
    }
    plant_model.load_state_dict(torch.load('industrial_model.pth', map_location=device))

    # plant_model.load_state_dict(torch.load('ablation_runs_new/baseline/model.pth', map_location=device))
    petri_models[MACHINE].load_state_dict(torch.load('petri_machine_model.pth', map_location=device))
    petri_models[ASSEMBLY].load_state_dict(torch.load('petri_assembly_model.pth', map_location=device))
    petri_models[DISASSEMBLY].load_state_dict(torch.load('petri_disassembly_model.pth', map_location=device))

    pipeline = IntegratedDiffusionPipeline(plant_model, petri_models, device)

    # Parametri
    n_nodes_global = 10
    n_nodes_petri  = 6
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Costruzione del grafo integrato (un campione)
    integ = pipeline.generate_full_integrated_graph(
        n_nodes_global=n_nodes_global,
        n_nodes_petri=n_nodes_petri,
        output_dir=DIR_INDUSTRIAL
    )

    # Salvataggio della copia vicino agli output cuciti
    tmp_pt = DIR_STITCHED / f"graphs_data_unstitched.pt"
    torch.save(integ, tmp_pt)

    # Immagine industriale (singolo, esatto come run_and_plot.py)
    global_node_types = _normalize_node_types(integ["global_graph"]["node_types"])  # <-- NORMALIZZATO
    global_adj        = np.array(integ["global_graph"]["adjacency_matrix"])
    industrial_png = DIR_INDUSTRIAL / f"industrial_graph.png"
    draw_industrial_graph(global_node_types, global_adj, industrial_png)

    torch.save(integ["petri_subgraphs"], Path(DIR_PETRI) / "petri_subgraphs.pt")
    # Petri subgraph (uno per ogni nodo non-Buffer), i nomi dei file includono l'indice industriale
    count = 0
    for idx, sub in integ["petri_subgraphs"].items():
        if len(sub["node_types"]) == 1 and sub["node_types"][0] == -1:
            continue
        node_type = int(global_node_types[idx])
        if node_type == BUFFER:
            continue
        _ = draw_petri_subgraph(sub, idx, TYPE_NAME[node_type], DIR_PETRI)
        count += 1

    # Grafo cucito (.pt + PNG) in stile Petri
    stitched_pt_path  = DIR_STITCHED / f"stitched.pt"
    stitched_png_path = DIR_STITCHED / f"stitched.png"

    stitched_data = pipeline.stitch(
        path=str(tmp_pt),
        save_path=str(stitched_pt_path),
        device=device
    )
    draw_stitched_graph(stitched_data, stitched_png_path)



if __name__ == "__main__":
    main()
