# experiments.py  – Main-results evaluation for the hierarchical diffusion model
# ------------------------------------------------------------------------------

import time, random, collections
import torch, networkx as nx
from industrial_diffusion import (LightweightIndustrialDiffusion,
                                  validate_constraints)        # C1–C6 checker
from diffusion_model          import strict_projector_petri     # Petri checker
from integrated_diffusion     import IntegratedDiffusionPipeline
from diffusion_model          import LightweightPetriDiffusion
import numpy as np
from torch_geometric.data     import Batch
from torch_geometric.nn       import global_mean_pool, GINConv
from torch.utils.data         import DataLoader
# --------------------------------------------------------------------------
# Helpers for hashing and validity
# --------------------------------------------------------------------------
def plant_valid(node_labels: torch.Tensor, edge_mat: torch.Tensor, device):
    """True iff plant-level constraints C1–C6 hold."""
    return validate_constraints(edge_mat, node_labels, device)

def petri_valid(node_labels: torch.Tensor, edge_mat: torch.Tensor, device):
    """True iff no self-loops nor same-type arcs exist (Petri constraints)."""
    projected = strict_projector_petri(node_labels, edge_mat, device)
    return torch.equal(projected, edge_mat)

def wl_hash(node_labels: torch.Tensor, edge_mat: torch.Tensor) -> str:
    """Deterministic hash ( Weisfeiler-Lehman ) for isomorphism tests."""
    G = nx.DiGraph()
    n = len(node_labels)
    for i in range(n):
        G.add_node(i, t=int(node_labels[i]))
    src, dst = torch.nonzero(edge_mat, as_tuple=True)
    for s, d in zip(src.tolist(), dst.tolist()):
        G.add_edge(s, d)
    return nx.weisfeiler_lehman_graph_hash(G, node_attr='t')


# --------------------------------------------------------------------------
# NUEVO – util para persistir los grafos en formato <adjacency,node_types,…>
# --------------------------------------------------------------------------
from pathlib import Path
import datetime as dt
import numpy as np

_SAVE_DIR = Path("exp_outputs")          # carpeta destino
_SAVE_DIR.mkdir(exist_ok=True)

# 1-solo: mapping fijo que ya usamos en el script
LABEL2ID = {"MACHINE": 3,
            "BUFFER":  1,
            "ASSEMBLY":0,
            "DISASSEMBLY":2}

def _save_graphs_pt(tag: str, batch: list[dict]) -> None:
    """
    Guarda el batch en un .pt con el mismo esquema que graphs_data_int.pt
    Keys:
      ├ adjacency_matrices : list[np.ndarray]  (int8/uint8)
      ├ node_types         : list[np.ndarray]  (int8)
      └ label2id           : dict[str,int]
    """
    adj_list, node_list = [], []
    for g in batch:
        #  (n,n) matriz de adyacencia → numpy uint8
        adj_list.append(g["edges"].cpu().numpy().astype(np.uint8))
        #  vector de etiquetas de nodo → numpy int8
        node_list.append(g["nodes"].cpu().numpy().astype(np.int8))

    payload = {
        "adjacency_matrices": adj_list,
        "node_types":         node_list,
        "label2id":           LABEL2ID
    }

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = _SAVE_DIR / f"{tag}_{stamp}.pt"
    torch.save(payload, fname)
    print(f"   ↳ Grafos guardados en {fname}")


# --------------------------------------------------------------------------
# Load training graphs – needed for Novelty %
# --------------------------------------------------------------------------

#below only for evaluating
#####################################
print("Loading training graphs …")
train = torch.load("graphs_data.pt")
train_hash_by_size = collections.defaultdict(set)
label_map = {"MACHINE":0,"BUFFER":1,"ASSEMBLY":2,"DISASSEMBLY":3}
for A, types in zip(train["adjacency_matrices"], train["node_types"]):
    nlab = torch.tensor([label_map[t] for t in types])
    emat = torch.tensor(A, dtype=torch.long)
    h    = wl_hash(nlab, emat)
    train_hash_by_size[len(types)].add(h)
######################################
# --------------------------------------------------------------------------
# Load pretrained models
# --------------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

plant_model = LightweightIndustrialDiffusion().to(device).eval()
# plant_model.load_state_dict(torch.load("industrial_model.pth",
#                                        map_location=device))

plant_model.load_state_dict(torch.load("ablation_runs_new/baseline/model.pth",
                                       map_location=device))

petri_models = {
    0: LightweightPetriDiffusion().to(device).eval(),
    2: LightweightPetriDiffusion().to(device).eval(),
    3: LightweightPetriDiffusion().to(device).eval(),
}
petri_models[0].load_state_dict(torch.load("petri_machine_model.pth",
                                           map_location=device))
petri_models[2].load_state_dict(torch.load("petri_assembly_model.pth",
                                           map_location=device))
petri_models[3].load_state_dict(torch.load("petri_disassembly_model.pth",
                                           map_location=device))


pipeline = IntegratedDiffusionPipeline(plant_model, petri_models, device)
print('model loaded')

# --------------------------------------------------------------------------
# Metric evaluator
# --------------------------------------------------------------------------
def evaluate(batch):
    """
    batch = list of dicts:
      {'nodes': Tensor[n], 'edges': Tensor[n,n],
       'petri': Dict[int, subgraph]  (optional), 'success': bool (optional)}
    """
    valid = unique = novel = success = 0
    seen_hashes = collections.defaultdict(set)      # uniqueness per size

    for g in batch:
        nlab  = g["nodes"]
        emat  = g["edges"]
        ok    = plant_valid(nlab, emat, device)

        # internal Petri nets, if provided
        for sub in g.get("petri", {}).values():
            if sub is None:       # buffer
                continue
            pn  = torch.tensor(sub["node_types"],        device=device)
            pe  = torch.tensor(sub["adjacency_matrix"],  device=device)
            ok &= petri_valid(pn, pe, device)

        if ok:
            valid += 1
            h      = wl_hash(nlab.cpu(), emat.cpu())
            nsize  = len(nlab)
            if h not in seen_hashes[nsize]:
                unique += 1
                seen_hashes[nsize].add(h)
            if h not in train_hash_by_size[nsize]:
                novel += 1
        if g.get("success", False):
            success += 1

    total = len(batch)
    res = {
        "valid":   100 * valid   / total,
        "unique":  100 * unique  / valid if valid else 0,
        "novel":   100 * novel   / valid if valid else 0,
    }
    if any("success" in g for g in batch):
        res["success"] = 100 * success / total
    return res

# --------------------------------------------------------------------------
# Experiment E1 – free generation
# --------------------------------------------------------------------------
def experiment_free(n_samples=300, n_nodes=15):
    batch = []
    t0 = time.time()
    for _ in range(n_samples):
        nodes, edges = pipeline.generate_global_graph(n_nodes)
        batch.append({"nodes": nodes,
                      "edges": edges.squeeze(0)})
    runtime = time.time() - t0
    print(f"[E1-Free]   {n_samples} samples in {runtime:.1f}s")
    #print(evaluate(batch), "\n")
    #extra_metrics(batch, tag="[E1]")
    _save_graphs_pt("E1", batch)

# --------------------------------------------------------------------------
# Experiment E2 – all-pinned inventory
# --------------------------------------------------------------------------
def experiment_allpinned(n_samples=300,
                         inv=(3,4,2,1)):   # (M, B, A, D)
    numM,numB,numA,numD = inv
    batch = []
    t0 = time.time()
    for _ in range(n_samples):
        nodes, edges = pipeline.generate_global_graph_all_pinned(
            num_machines=numM,
            num_buffers=numB,
            num_assemblies=numA,
            num_disassemblies=numD)
        ok_inv = ( (nodes==0).sum()==numM and
                   (nodes==1).sum()==numB and
                   (nodes==2).sum()==numA and
                   (nodes==3).sum()==numD )
        batch.append({"nodes": nodes,
                      "edges": edges.squeeze(0),
                      "success": ok_inv})
    runtime = time.time() - t0
    print(f"[E2-AllPinned] {n_samples} samples in {runtime:.1f}s")
    #print(evaluate(batch), "\n")
    #extra_metrics(batch, tag="[E2]")
    _save_graphs_pt("E2", batch)

# --------------------------------------------------------------------------
# Experiment E3 – partial-pinned (30 % nodes)
# --------------------------------------------------------------------------
def experiment_partial(n_samples=300, n_nodes=20, pin_ratio=0.3):
    batch = []
    t0 = time.time()
    for _ in range(n_samples):
        pin_counts = {"MACHINE": 1,
                      "ASSEMBLY": 1,
                      "BUFFER": int(pin_ratio*n_nodes) - 2}
        nodes, edges = pipeline.generate_global_graph_partial_pinned(
            num_nodes=n_nodes,
            pinned_info=pin_counts)
        batch.append({"nodes": nodes,
                      "edges": edges.squeeze(0)})
    runtime = time.time() - t0
    print(f"[E3-Partial] {n_samples} samples in {runtime:.1f}s")
    #print(evaluate(batch), "\n")
    #extra_metrics(batch, tag="[E3]")
    _save_graphs_pt("E3", batch)


# ───────────── extra: FID / MMD ───────────────────────────────

class GraphEncoder(torch.nn.Module):
    """Mini-GIN → mean-pool → linear  (128-D por defecto)."""
    def __init__(self, in_dim=4, hid=64, out=128):
        super().__init__()
        mlp = torch.nn.Sequential(torch.nn.Linear(in_dim, hid),
                                  torch.nn.ReLU(),
                                  torch.nn.Linear(hid, hid))
        self.conv = GINConv(mlp)
        self.lin  = torch.nn.Linear(hid, out)

    def forward(self, batch):
        h = self.conv(batch.x, batch.edge_index)
        h = global_mean_pool(h, batch.batch)      # (B, hid)
        return self.lin(h)                        # (B, out)


@torch.no_grad()
def encode_graphs(list_dicts, encoder, device='cpu', bs=64):
    """Convert your list of dicts {'nodes','edges'} into embeddings."""
    data_objs = []
    for g in list_dicts:
        x = torch.nn.functional.one_hot(g["nodes"], num_classes=4).float()
        edge_idx = (g["edges"] > 0).nonzero(as_tuple=False).t().contiguous()
        from torch_geometric.data import Data
        data_objs.append(Data(x=x, edge_index=edge_idx))
    loader = DataLoader(data_objs, bs, shuffle=False,
                        collate_fn=Batch.from_data_list)
    Z = []
    for batch in loader:
        Z.append(encoder(batch.to(device)).cpu())
    return torch.cat(Z, 0)                # (N, d)


def frechet(mu1, cov1, mu2, cov2):
    diff = mu1 - mu2
    covmean = cov_sqrt(cov1 @ cov2)
    return diff.dot(diff) + torch.trace(cov1 + cov2 - 2 * covmean)

def cov_sqrt(mat, eps=1e-8):
    # mat: (d,d) simétrica PSD
    evals, evecs = torch.linalg.eigh(mat)
    evals_clamped = torch.clamp(evals, min=0.)          # num. safety
    return (evecs * evals_clamped.sqrt()) @ evecs.t()



def mmd_rbf(X, Y):
    # heuristic bandwidth (median)
    Z = torch.cat([X, Y], 0)
    sq = torch.cdist(Z, Z, p=2.0)**2
    sigma = torch.sqrt(0.5*sq[sq>0].median())
    k = lambda A,B: torch.exp(-torch.cdist(A,B,p=2.0)**2 / (2*sigma**2))
    m, n = len(X), len(Y)
    return (k(X,X).sum() - m)/(m*(m-1)) \
         + (k(Y,Y).sum() - n)/(n*(n-1)) \
         - 2*k(X,Y).mean()


def extra_metrics(batch, tag=""):
    """Calcula FID y MMD de esta tanda frente al set de training."""
    # 1)  preparar encoder (si ya lo tienes entrenado, cárgale pesos)
    enc = GraphEncoder(in_dim=4).to(device).eval()

    # 2)  embeddings de training   (usa los que ya cargaste al principio)
    global _Z_train          # cache en memoria
    if '_Z_train' not in globals():
        train_objs = []
        lmap = {"MACHINE":0,"BUFFER":1,"ASSEMBLY":2,"DISASSEMBLY":3}
        for A, types in zip(train["adjacency_matrices"], train["node_types"]):
            xlabs = torch.tensor([lmap[t] for t in types])
            train_objs.append({"nodes": xlabs,
                               "edges": torch.tensor(A)})
        _Z_train = encode_graphs(train_objs, enc, device)

    # 3)  embeddings de la muestra
    Z_samp = encode_graphs(batch, enc, device)

    # 4)  FID
    mu_t, cov_t = _Z_train.mean(0), torch.from_numpy(np.cov(_Z_train.T.numpy()))
    mu_s, cov_s = Z_samp.mean(0), torch.from_numpy(np.cov(Z_samp.T.numpy()))
    fid  = frechet(mu_t, cov_t, mu_s, cov_s).item()

    # 5)  MMD
    mmd = mmd_rbf(_Z_train, Z_samp).item()

    print(f"   ↳ FID={fid:7.2f}   MMD={mmd:7.4f}   {tag}")
# ──────────────────────────────────────────────────────────────


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
if __name__ == "__main__":

############### generate industrial graphs
    # torch.manual_seed(42); random.seed(42)
    experiment_free(n_samples=300)       # E1
    experiment_allpinned(n_samples=300, inv=(3,4,2,1))  # E2
    experiment_partial(n_samples=300)    # E3

################ generate petrinet
    # from integrated_diffusion import IntegratedDiffusionPipeline
    # from torch_geometric.data import Data
    #
    # pipeline = IntegratedDiffusionPipeline(plant_model, petri_models, device)
    #
    # all_graphs = []
    #
    # for i in range(10):
    #     petri_nodes, petri_edges = pipeline.generate_petri_subgraph(node_type=0, n_nodes_petri=8)
    #
    #     n_nodes = len(petri_nodes)
    #     edge_index = (petri_edges[0] != 0).nonzero(as_tuple=False).t().contiguous()  # shape [2, num_edges]
    #     x = torch.nn.functional.one_hot(petri_nodes, num_classes=2).float()  # shape [n_nodes, 2]
    #     graph_data = Data(x=x, edge_index=edge_index)
    #
    #     all_graphs.append(graph_data)
    #
    # torch.save(all_graphs, 'petri_net_917_10.pt')



# ################ generate integrated graph
#     integ = pipeline.generate_full_integrated_graph(n_nodes_global=10, n_nodes_petri=6)
#     torch.save(integ, "graphs_data_tmp.pt")
#     #
#     pipeline.stitch("graphs_data_tmp.pt", save_path="./stitched_graph22.pt")
