# train_industrial.py  ── versión cronometrada
import os, time, argparse, torch
from torch_geometric.loader import DataLoader
from industrial_dataset   import IndustrialGraphDataset
from industrial_diffusion import (LightweightIndustrialDiffusion,
                                  train_model)          # ya definido
from torch_geometric.utils import to_dense_adj


# ---------- utilidades para pesos y marginales ----------
def compute_edge_weights(dataset, device):
    total_edges = 0
    class_counts = torch.zeros(2, device=device)
    for data in dataset:
        dense = to_dense_adj(data.edge_index,
                             max_num_nodes=data.x.size(0))[0]
        e0 = (dense > 0).long()
        class_counts += torch.bincount(e0.view(-1), minlength=2).to(device)
        total_edges  += e0.numel()
    class_counts[class_counts == 0] = 1.0
    w = total_edges / (2.0 * class_counts)
    return w / w.sum()

def compute_marginal_probs(dataset, device):
    node_counts = torch.zeros(4, device=device)   # 4 tipos de nodo
    edge_counts = torch.zeros(2, device=device)
    n_nodes = n_edges = 0
    for data in dataset:
        labels = data.x.argmax(dim=1)
        node_counts += torch.bincount(labels, minlength=4).float().to(device)
        n_nodes += data.x.size(0)
        dense = to_dense_adj(data.edge_index,
                             max_num_nodes=data.x.size(0))[0]
        e0 = (dense > 0).long()
        edge_counts += torch.bincount(e0.view(-1), minlength=2).float().to(device)
        n_edges += e0.numel()
    return node_counts / n_nodes, edge_counts / n_edges
# --------------------------------------------------------


def run_training(epochs=30, batch=4, lr=1e-3):
    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset  = IndustrialGraphDataset(root='industrial_dataset')
    loader   = DataLoader(dataset, batch_size=batch, shuffle=True)

    edge_w   = compute_edge_weights(dataset, device)
    node_m, edge_m = compute_marginal_probs(dataset, device)

    model     = LightweightIndustrialDiffusion().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    print(f"\n▶ Training INDUSTRIAL model  ({len(dataset)} graphs)")
    start = time.perf_counter()

    train_model(model, loader, optimizer, device,
                edge_weight=edge_w,
                node_marginal=node_m,
                edge_marginal=edge_m,
                epochs=epochs, T=100)

    elapsed = time.perf_counter() - start
    print(f"⏱  Finished in {elapsed/60:.1f} min  ({elapsed:.1f} s)\n")

    torch.save(model.state_dict(), 'industrial_model.pth')
    print("✅ Weights saved to  industrial_model.pth")


# -------------- entry point con argparse -----------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30,
                        help='Número de épocas de entrenamiento')
    parser.add_argument('--batch',  type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr',     type=float, default=1e-3,
                        help='Learning rate')
    args = parser.parse_args()

    run_training(epochs=args.epochs,
                 batch=args.batch,
                 lr=args.lr)
