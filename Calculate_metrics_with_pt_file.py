import networkx as nx
from Industrial_Pipeline_Functions import validate_constraints
from torch_geometric.utils import to_dense_adj
import torch
from torch_geometric.nn import GINConv, global_mean_pool
from Industrial_Pipeline_Functions import IndustrialGraphDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import numpy as np
import os
import pandas as pd


def convert_samples_compare(samples_compare, device="cpu"):
    adj_list = samples_compare["adjacency_matrices"]
    node_list = samples_compare["node_types"]

    assert len(adj_list) == len(node_list)

    converted = []
    for adj, nodes in zip(adj_list, node_list):
        if not torch.is_tensor(adj):
            adj = torch.as_tensor(adj, dtype=torch.float32, device=device)
        if not torch.is_tensor(nodes):
            nodes = torch.as_tensor(nodes, dtype=torch.long, device=device)

        converted.append((nodes, adj))
    return converted

def main():

    # pt_path_compare = 'exp_outputs/300_samples/E3.pt'
    # samples_compare = torch.load(pt_path_compare)
    # converted_samples = convert_samples_compare(samples_compare)
    #
    # results = calculate_metrics_pt_file(converted_samples)
    # print(results)
#############################
    folder_path = 'exp_outputs/300_samples'
    output_csv = 'exp_outputs/300_samples/results.csv'

    # folder_path = 'exp_outputs/E1/pt_file'
    # output_csv = 'exp_outputs/E1/pt_file/results.csv'

    results_list = []

    for filename in os.listdir(folder_path):
        if filename.endswith('.pt'):
            file_path = os.path.join(folder_path, filename)
            print(f"Processing {file_path} ...")

            samples = torch.load(file_path)
            converted_samples = convert_samples_compare(samples)
            result = calculate_metrics_pt_file(converted_samples)

            result['file'] = filename
            print(result)
            results_list.append(result)

    df = pd.DataFrame(results_list)
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

Dataset_path = 'industrial_dataset'

class GraphEncoder(torch.nn.Module):
    """Mini-GIN → mean-pool → linear  (128-D por defecto)."""

    def __init__(self, in_dim=4, hid=64, out=128):
        super().__init__()
        mlp = torch.nn.Sequential(torch.nn.Linear(in_dim, hid),
                                  torch.nn.ReLU(),
                                  torch.nn.Linear(hid, hid))
        self.conv = GINConv(mlp)
        self.lin = torch.nn.Linear(hid, out)

    def forward(self, batch):
        h = self.conv(batch.x, batch.edge_index)
        h = global_mean_pool(h, batch.batch)  # (B, hid)
        return self.lin(h)  # (B, out)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ENC = GraphEncoder(in_dim=4).to(device).eval()


def wl_hash(nodes, edges):
    G = nx.Graph()
    for i, t in enumerate(nodes.tolist()):
        G.add_node(int(i), label=int(t))
    for i, j in (edges.nonzero(as_tuple=False)):
        G.add_edge(int(i), int(j))
    return nx.weisfeiler_lehman_graph_hash(G, node_attr="label")


@torch.no_grad()
def encode_graphs(list_dicts, encoder, device='cpu', bs=64):
    """Convierte tu lista de dicts {'nodes','edges'} en embeddings."""
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
    return torch.cat(Z, 0)  # (N, d)


def cov_sqrt(mat, eps=1e-8):
    # mat: (d,d) simétrica PSD
    evals, evecs = torch.linalg.eigh(mat)
    evals_clamped = torch.clamp(evals, min=0.)  # num. safety
    return (evecs * evals_clamped.sqrt()) @ evecs.t()


def frechet(mu1, cov1, mu2, cov2):
    diff = mu1 - mu2
    covmean = cov_sqrt(cov1 @ cov2)
    return diff.dot(diff) + torch.trace(cov1 + cov2 - 2 * covmean)


def mmd_rbf(X, Y):
    # bandwidth heurístico (mediana)
    Z = torch.cat([X, Y], 0)
    sq = torch.cdist(Z, Z, p=2.0) ** 2
    sigma = torch.sqrt(0.5 * sq[sq > 0].median())
    k = lambda A, B: torch.exp(-torch.cdist(A, B, p=2.0) ** 2 / (2 * sigma ** 2))
    m, n = len(X), len(Y)
    return (k(X, X).sum() - m) / (m * (m - 1)) \
           + (k(Y, Y).sum() - n) / (n * (n - 1)) \
           - 2 * k(X, Y).mean()


def _embed_dataset(dataset):
    objs = []
    for d in dataset:
        x_lbl = d.x.argmax(1)
        dense = to_dense_adj(d.edge_index, max_num_nodes=d.x.size(0))[0]
        objs.append({"nodes": x_lbl, "edges": dense})
    return encode_graphs(objs, ENC, device)


def calculate_metrics_pt_file(samples):
    DATASET = IndustrialGraphDataset(root=Dataset_path)

    hashes = [wl_hash(n_lbls.cpu(), e_mat.cpu()) for (n_lbls, e_mat) in samples]
    # validity
    viol = sum(not validate_constraints(e_mat, n_lbls, device) for (n_lbls, e_mat) in samples)
    validity = 1 - viol / len(samples) if len(samples) > 0 else 0.0

    # uniqueness (WL)
    uniqueness = len(set(hashes)) / len(samples) if len(samples) > 0 else 0.0

    # novelty (WL) vs dataset
    train_hashes = {
        wl_hash(d.x.argmax(1).cpu(),
                to_dense_adj(d.edge_index, max_num_nodes=d.x.size(0))[0].cpu())
        for d in DATASET
    }
    novelty = sum(h not in train_hashes for h in hashes) / len(samples) if len(samples) > 0 else 0.0

    _Z_TRAIN = _embed_dataset(DATASET)
    MU_T, COV_T = _Z_TRAIN.mean(0), torch.from_numpy(np.cov(_Z_TRAIN.T.numpy()))
    # FID / MMD
    batch_objs = [{"nodes": n_lbls, "edges": e_mat} for (n_lbls, e_mat) in samples]
    Z_samp = encode_graphs(batch_objs, ENC, device)
    mu_s = Z_samp.mean(0)
    cov_s = torch.from_numpy(np.cov(Z_samp.T.numpy())) if Z_samp.shape[0] > 1 else torch.eye(Z_samp.shape[1])
    fid = frechet(MU_T, COV_T, mu_s, cov_s).item()
    mmd = mmd_rbf(_Z_TRAIN, Z_samp).item()

    return {"validity": validity, "uniqueness": uniqueness,
            "novelty": novelty, "fid": fid, "mmd": mmd}


if __name__ == '__main__':
    main()
