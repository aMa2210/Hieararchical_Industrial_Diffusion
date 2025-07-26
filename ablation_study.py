# ablation_study.py 
"""
Lanza varios entrenamientos con distintos switches de pérdida
y recolecta métricas de validez, unicidad y novedad.

Estructura de carpetas creada:
./ablation_runs/
    └── <run_name>/
          ├── model.pth
          ├── samples.pt
          └── metrics.json
También se genera ./ablation_summary.csv con todas las filas.
"""

import json, csv, os, shutil, torch
import torch_geometric                       # alias global
from torch_geometric.data import Data        # clase Data
from torch_geometric.utils import to_dense_adj
from datetime import datetime
from pathlib import Path
from torch_geometric.loader import DataLoader
from industrial_dataset import IndustrialGraphDataset
from industrial_diffusion import (
    LightweightIndustrialDiffusion,
    train_model, validate_constraints                               # <-- sí vive en industrial_diffusion.py
)

from random import randint
from experiments import encode_graphs, GraphEncoder, frechet, mmd_rbf, cov_sqrt

# Estas dos utilidades están en train_industrial.py
from train_industrial import (
    compute_edge_weights,
    compute_marginal_probs,
)

import networkx as nx
# ---------- 1. Configuración global ----------
ROOT_OUT = Path("ablation_runs")
ROOT_OUT.mkdir(exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET = IndustrialGraphDataset(root="industrial_dataset")
LOADER  = DataLoader(DATASET, batch_size=4, shuffle=True)

EDGE_W  = compute_edge_weights(DATASET, DEVICE)
NODE_M, EDGE_M = compute_marginal_probs(DATASET, DEVICE)

EPOCHS = 30         # ↓ Ajusta para tu GPU
T_STEPS = 100
N_SAMP  = 300        # muestras para métrica


from torch_geometric.data import Batch
from torch_geometric.nn   import GINConv, global_mean_pool
import numpy as np

device = DEVICE  # reutilizamos la misma GPU

ENC = GraphEncoder(in_dim=4).to(device).eval()

def _embed_dataset(dataset):
    objs = []
    for d in dataset:
        x_lbl = d.x.argmax(1)
        dense = to_dense_adj(d.edge_index, max_num_nodes=d.x.size(0))[0]
        objs.append({"nodes": x_lbl, "edges": dense})
    return encode_graphs(objs, ENC, device)

_Z_TRAIN = _embed_dataset(DATASET)   # se guarda en RAM
MU_T, COV_T = _Z_TRAIN.mean(0), torch.from_numpy(np.cov(_Z_TRAIN.T.numpy()))

# ---------- 2. Lista de variantes ----------
ABLATIONS = [
    # modelo completo (24 dim)
    {"name": "baseline",
     "hidden_dim": 12, "kl": 0.1, "constraint": 1.0, "use_projector": True},

    # mismo tamaño pero sin projector
    {"name": "no_projector",
     "hidden_dim": 12, "kl": 0.1, "constraint": 0.0, "use_projector": False},

    # red más pequeña: 32 → ½ de 64
    {"name": "half_hidden",
     "hidden_dim": 6, "kl": 0.1, "constraint": 1.0, "use_projector": True},

    # pesos aleatorios + projector
    {"name": "random_projector",
     "hidden_dim": 12, "kl": 0.0, "constraint": 0.0,
     "use_projector": True, "skip_training": True},
]

# ---------- 3. Métricas simples ----------

def wl_hash(nodes, edges):
    G = nx.Graph()
    for i, t in enumerate(nodes.tolist()):
        G.add_node(int(i), label=int(t))
    for i, j in (edges.nonzero(as_tuple=False)):
        G.add_edge(int(i), int(j))
    return nx.weisfeiler_lehman_graph_hash(G, node_attr="label")


def compute_metrics(model, out_dir, n_samples=N_SAMP):
    model.eval()
    samples, hashes = [], []
    with torch.no_grad():
        for _ in range(n_samples):
            n = 15
            edge_idx = torch.tensor([(i, j) for i in range(n) for j in range(n)
                                     if i != j], dtype=torch.long).t().contiguous().to(DEVICE)
            data = Data(x=torch.zeros(n, model.node_num_classes, device=DEVICE),
                        edge_index=edge_idx)
            data.batch = torch.zeros(n, dtype=torch.long, device=DEVICE)
            nodes, edges, _ = model.reverse_diffusion_single(
                data, DEVICE, save_intermediate=False)
            samples.append((nodes.cpu(), edges.squeeze(0).cpu()))
            hashes.append(wl_hash(nodes.cpu(), edges.squeeze(0).cpu()))  # ← uno por muestra

    # validity
    viol = sum(not validate_constraints(e, n, DEVICE) for n, e in samples)
    validity = 1 - viol / n_samples

    # uniqueness (WL)  – colisiones entre muestras
    uniqueness = len(set(hashes)) / n_samples

    # novelty (WL) – cada muestra vs dataset
    train_hashes = { wl_hash(d.x.argmax(1).cpu(),
                              to_dense_adj(d.edge_index,
                                           max_num_nodes=d.x.size(0))[0].cpu())
                     for d in DATASET }
    novelty = sum(h not in train_hashes for h in hashes) / n_samples

    # FID / MMD
    batch_objs = [{"nodes": n_lbls, "edges": e_mat} for n_lbls, e_mat in samples]
    Z_samp = encode_graphs(batch_objs, ENC, device)
    mu_s, cov_s = Z_samp.mean(0), torch.from_numpy(np.cov(Z_samp.T.numpy()))
    fid = frechet(MU_T, COV_T, mu_s, cov_s).item()
    mmd = mmd_rbf(_Z_TRAIN, Z_samp).item()

    torch.save(samples, out_dir/"samples.pt")
    return {"validity": validity, "uniqueness": uniqueness,
            "novelty": novelty, "fid": fid, "mmd": mmd}

# ---------- 4. Bucle principal ----------
summary_rows = []
from industrial_diffusion import compute_batch_loss

def train_loop(model, loader, device,
               edge_w, node_m, edge_m,
               epochs, T, kl, c_lam):
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for batch in loader:
            batch = batch.to(device)
            opt.zero_grad()
            loss = compute_batch_loss(model, batch, T, device,
                                      edge_w, node_m, edge_m,
                                      kl_lambda=kl,
                                      constraint_lambda=c_lam)
            loss.backward()
            opt.step()
            tot += loss.item()
        print(f"Epoch {ep+1}/{epochs}, Loss: {tot/len(loader):.4f}")

for cfg in ABLATIONS:
    run_name = f"{cfg['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir  = ROOT_OUT/run_name
    out_dir.mkdir()

    print(f"▶️  Entrenando {run_name}")
    hidden = cfg.get("hidden_dim", 12)
    model = LightweightIndustrialDiffusion(
    T=T_STEPS,
    hidden_dim=hidden,
    use_projector=cfg.get("use_projector", True)
    ).to(DEVICE)


    kl_l  = cfg.get("kl", 0.1)
    c_lam = cfg.get("constraint", 1.0)

    if not cfg.get("skip_training", False):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train_loop(model, LOADER, DEVICE,
               EDGE_W, NODE_M, EDGE_M,
               EPOCHS, T_STEPS,
               cfg.get("kl", 0.1),
               cfg.get("constraint", 1.0))

    torch.save(model.state_dict(), out_dir/"model.pth")

    # ---------- 4.1 Métricas ----------
    metrics = compute_metrics(model, out_dir)
    metrics.update({"run": run_name, **cfg})
    print("→ Resultados:", {k: round(v, 4) if isinstance(v, float) else v
                            for k, v in metrics.items()})
    summary_rows.append(metrics)  
    # Liberar GPU intermedio
    del model; torch.cuda.empty_cache()

print("\n=== Resumen final ===")
for row in summary_rows:
    print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})