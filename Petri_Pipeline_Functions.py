# Petri Graph Pipeline

# 1 Petri Dataset Creation
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils
from torch_geometric.data import InMemoryDataset, Data
from typing import Optional
import networkx as nx


class PetriNetDataset(InMemoryDataset):
    def __init__(self, root, split='train', transform=None, pre_transform=None, pre_filter=None):
        self.split = split
        super(PetriNetDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [f'{self.split}.pt']

    @property
    def processed_file_names(self):
        return [f'{self.split}_processed.pt']

    def download(self):
        raw_split_path = os.path.join(self.raw_dir, f'{self.split}.pt')
        if not os.path.exists(raw_split_path):
            raise FileNotFoundError(f"Local file not found: {raw_split_path}")

    def process(self):
        raw_path = os.path.join(self.raw_dir, f'{self.split}.pt')
        print("🪵 Reading raw from:", raw_path)
        raw_dataset = torch.load(raw_path)
        print("🪵 Type of the first sample:", type(raw_dataset[0]))

        data_list = []
        for sample in raw_dataset:
            adj_matrix = torch.tensor(sample['adjacency_matrix'], dtype=torch.float)
            node_types = sample['node_types']
            nodes_order = sample['nodes_order']

            # Encode nodes (Place=0, Transition=1)
            node_labels = torch.tensor([0 if node_types[node] == "Place" else 1 for node in nodes_order], dtype=torch.long)

            # One-hot encode node types
            x = F.one_hot(node_labels, num_classes=2).float()

            # Convert adjacency matrix to edge_index
            edge_index, _ = pyg_utils.dense_to_sparse(adj_matrix)
            edge_index = edge_index.long()

            d = Data(x=x, edge_index=edge_index)
            d.n_nodes = x.size(0)
            data_list.append(d)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# 2 Petri Diffusion Model
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import to_dense_batch, to_dense_adj, subgraph
from torch_geometric.data import Data, DataLoader, Batch
import os


def get_forbidden_mask(node_labels, device):
    """
    Forbidden edges for Petri nets:
      - self-loops
      - same-type edges (Place-Place or Transition-Transition)
    """
    n = node_labels.size(0)
    diag_mask = torch.eye(n, dtype=torch.bool, device=device)
    allowed_mask = (node_labels.unsqueeze(1) != node_labels.unsqueeze(0))
    forbidden_mask = diag_mask | (~allowed_mask)
    return forbidden_mask.float()  # 1.0 forbidden, 0.0 allowed


def get_forbidden_edges_petri(node_labels):
    """Return forbidden (i<=j) edges: self-loops + same-type."""
    n = node_labels.size(0)
    forbidden = []
    for i in range(n):
        for j in range(i, n):
            if i == j or node_labels[i] == node_labels[j]:
                forbidden.append([i, j])
    if len(forbidden) > 0:
        return torch.tensor(forbidden, dtype=torch.long, device=node_labels.device)
    else:
        return torch.empty((0, 2), dtype=torch.long, device=node_labels.device)


def strict_projector_petri(node_labels, candidate_edge_matrix, device):
    """Zero out edges that are forbidden (self-loops or same-type)."""
    forbidden_edges = get_forbidden_edges_petri(node_labels)
    projected = candidate_edge_matrix.clone()
    if forbidden_edges.numel() > 0:
        for edge in forbidden_edges:
            i, j = edge[0].item(), edge[1].item()
            projected[i, j] = 0
            projected[j, i] = 0
    return projected


# ---------- NEW: extra constraints helpers (no-isolated, connected, source transition) ----------
def _undirected_adj(E: torch.Tensor) -> torch.Tensor:
    """Binary undirected adjacency from directed 0/1 matrix E."""
    return ((E > 0) | (E.t() > 0)).to(E.dtype)

def _components(E_und: torch.Tensor):
    """Return list of components (as python lists of node indices) using DFS/BFS."""
    n = E_und.size(0)
    visited = [False] * n
    comps = []
    for s in range(n):
        if visited[s]:
            continue
        stack = [s]
        visited[s] = True
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            nbrs = torch.where(E_und[u] > 0)[0].tolist()
            for v in nbrs:
                if not visited[v]:
                    visited[v] = True
                    stack.append(v)
        comps.append(comp)
    return comps

def _ensure_min_degree(node_labels: torch.Tensor, E: torch.Tensor):
    """
    Guarantee every node has at least one incident edge by adding a valid cross-type edge if needed.
    """
    n = E.size(0)
    y = node_labels
    for i in range(n):
        deg = int((E[i] > 0).sum().item() + (E[:, i] > 0).sum().item())
        if deg == 0:
            # connect to any node of opposite type
            opp = torch.where(y != y[i])[0]
            if opp.numel() == 0:
                continue
            j = int(opp[0].item())  # deterministic choice
            # add a single directed edge (either direction is allowed)
            if y[i] == 0 and y[j] == 1:
                E[i, j] = 1  # Place -> Transition
            elif y[i] == 1 and y[j] == 0:
                E[i, j] = 1  # Transition -> Place
    return E

def _ensure_connected(node_labels: torch.Tensor, E: torch.Tensor):
    """
    Make the graph weakly connected by adding a cross-type bridge between components.
    """
    y = node_labels
    while True:
        und = _undirected_adj(E)
        comps = _components(und)
        if len(comps) <= 1:
            break
        # connect comp[0] with comp[k]
        A = comps[0]
        B = comps[-1]
        # find any cross-type pair (u in B, v in A) and add a directed edge
        added = False
        for u in B:
            for v in A:
                if y[u] != y[v]:
                    # choose u -> v (valid by bipartite)
                    E[u, v] = 1
                    added = True
                    break
            if added:
                break
        if not added:
            # fallback (shouldn't happen): stop to avoid infinite loop
            break
    return E

def _ensure_source_transition(node_labels: torch.Tensor, E: torch.Tensor):
    """
    Ensure at least one Transition has in-degree 0.
    If none, pick a transition t*, take one incoming place p->t*, re-route p to another transition (or add t*->p) and remove p->t*.
    """
    y = node_labels
    trans_idx = torch.where(y == 1)[0]
    if trans_idx.numel() == 0:
        return E  # no transitions to constrain

    indeg_T = (E[:, trans_idx] > 0).sum(dim=0)  # per-transition in-degree
    if (indeg_T == 0).any():
        return E  # already satisfied

    # choose a transition with minimal in-degree (>=1 here)
    t_pos = int(torch.argmin(indeg_T).item())
    t = int(trans_idx[t_pos].item())

    # find any incoming place p -> t
    p_cands = torch.where((E[:, t] > 0) & (y == 0))[0]
    if p_cands.numel() == 0:
        return E  # nothing to remove; unusual—but keep safe

    p = int(p_cands[0].item())

    # try to redirect p to another transition v != t
    other_T = trans_idx[trans_idx != t]
    if other_T.numel() > 0:
        v = int(other_T[0].item())
        E[p, v] = 1  # keep p connected
        E[p, t] = 0  # remove incoming to t -> now t loses one input
    else:
        # only one transition exists; make it source by removing its incoming
        # keep p connected by adding t -> p if missing
        if E[t, p] == 0:
            E[t, p] = 1
        E[p, t] = 0

    return E

def enforce_petri_global_constraints(node_labels: torch.Tensor, E: torch.Tensor) -> torch.Tensor:
    """
    Enforce:
      - no self-loops, no same-type edges (bipartite)
      - every node has at least one incident edge
      - graph is weakly connected
      - at least one Transition has in-degree 0
    """
    device = E.device
    y = node_labels.to(device)
    n = y.size(0)

    # 1) remove forbidden edges (also clean diagonal)
    E = strict_projector_petri(y, E, device)

    # 2) min degree >=1
    E = _ensure_min_degree(y, E)

    # 3) weak connectivity
    E = _ensure_connected(y, E)

    # 4) at least one Transition with in-degree 0
    E = _ensure_source_transition(y, E)

    # final clean-up (never hurts)
    E = strict_projector_petri(y, E, device)
    return E

# -----------------------------------------------------------------------------------------------


def get_sinusoidal_embedding(t, embedding_dim):
    """Sinusoidal time embedding."""
    if t.dim() == 1:
        t = t.unsqueeze(1)
    device = t.device
    half_dim = embedding_dim // 2
    emb_scale = math.log(10000) / (half_dim - 1)
    dims = torch.arange(half_dim, device=device).float()
    dims = torch.exp(-dims * emb_scale)
    emb = t * dims.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros(t.size(0), 1, device=device)], dim=1)
    return emb


def strict_validation_petri(node_labels, candidate_edge_matrix, device):
    """
    Ensure at least one Transition has no input, and fix isolated nodes.
    Prevent isolated nodes from connecting to the marked Transition.
    """
    n = node_labels.size(0)
    projected = candidate_edge_matrix.clone()
    transitions = torch.where(node_labels == 1)[0]
    in_degrees = projected[:, transitions].sum(0)
    min_input_idx = transitions[torch.argmin(in_degrees)]
    projected[:, min_input_idx] = 0
    marked_transition = min_input_idx.item()

    for i in range(n):
        if projected[i, :].sum() == 0 and projected[:, i].sum() == 0:
            node_type = node_labels[i].item()
            other_type = 1 - node_type

            other_indices = torch.where(node_labels == other_type)[0]
            if node_type == 0:
                other_indices = other_indices[other_indices != marked_transition]
            else:  # Transition -> Place
                pass  # no restriction needed for Places

            if len(other_indices) > 0:
                degrees = projected[other_indices, :].sum(1) + projected[:, other_indices].sum(0)
                min_deg_idx = other_indices[torch.argmin(degrees)]
                projected[i, min_deg_idx] = 1

    G = nx.from_numpy_array(projected.cpu().numpy(), create_using=nx.DiGraph)
    components = list(nx.weakly_connected_components(G))

    while len(components) > 1:
        comp1 = components[0]
        comp2 = components[1]

        # Pick one node from comp1 and one from comp2 (with different types)
        u_candidates = list(comp1)
        v_candidates = list(comp2)

        # Ensure Place -> Transition or Transition -> Place
        edge_added = False
        for u in u_candidates:
            for v in v_candidates:
                if node_labels[u] != node_labels[v]:
                    if v == marked_transition:
                        projected[v, u] = 1
                    else:
                        projected[u, v] = 1
                    edge_added = True
                    break
            if edge_added:
                break
        G = nx.from_numpy_array(projected.cpu().numpy(), create_using=nx.DiGraph)
        components = list(nx.weakly_connected_components(G))

    return projected


class LightweightPetriDiffusion(nn.Module):
    def __init__(self, T=100, hidden_dim=12, beta_start=0.0001, beta_end=0.02, time_embed_dim=16, nhead=4, dropout=0.1, use_projector=True):
        super().__init__()
        self.T = T
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_schedule = torch.linspace(beta_start, beta_end, T)
        self.alpha = 1 - self.beta_schedule
        self.register_buffer('alpha_bar', torch.cumprod(self.alpha, dim=0))
        self.use_projector = use_projector
        
        self.node_num_classes = 2
        self.edge_num_classes = 2
        
        self.time_embed_dim = time_embed_dim
        self.time_linear = nn.Linear(time_embed_dim, time_embed_dim)
        
        in_channels = self.node_num_classes + time_embed_dim
        self.transformer1 = TransformerConv(in_channels, hidden_dim, heads=nhead, concat=False, dropout=dropout)
        self.transformer2 = TransformerConv(hidden_dim, hidden_dim, heads=nhead, concat=False, dropout=dropout)
        self.node_out = nn.Linear(hidden_dim, self.node_num_classes)
        
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.edge_num_classes)
        )
    
    def forward(self, x, edge_index, batch, t):
        t_tensor = torch.tensor([t], dtype=torch.float, device=x.device) / self.T
        t_embed = get_sinusoidal_embedding(t_tensor, self.time_embed_dim)
        t_embed = self.time_linear(t_embed)
        t_embed_full = t_embed.repeat(x.size(0), 1)
        
        x_input = torch.cat([x, t_embed_full], dim=1)
        h = F.relu(self.transformer1(x_input, edge_index))
        h = F.relu(self.transformer2(h, edge_index))
        node_logits = self.node_out(h)
        
        h_dense, mask = to_dense_batch(h, batch)  # (B, maxN, H)
        edge_logits_list = []
        for i in range(h_dense.size(0)):
            num_nodes = int(mask[i].sum().item())
            if num_nodes == 0:
                edge_logits_list.append(torch.empty(0))
                continue
            h_i = h_dense[i, :num_nodes, :]
            h_i_exp1 = h_i.unsqueeze(1).expand(-1, num_nodes, -1)
            h_i_exp2 = h_i.unsqueeze(0).expand(num_nodes, -1, -1)
            edge_input = torch.cat([h_i_exp1, h_i_exp2], dim=-1)
            edge_logits = self.edge_mlp(edge_input)
            edge_logits_list.append(edge_logits)
        return node_logits, edge_logits_list
    
    def forward_diffusion(self, x0, e0, t, device):
        p_keep = self.alpha_bar[t].item()

        # Nodes
        rand_vals = torch.rand(x0.shape, device=device)
        random_node = torch.randint(0, self.node_num_classes, x0.shape, device=device)
        x_t = torch.where(rand_vals < p_keep, x0, random_node)
        x_t_onehot = F.one_hot(x_t, num_classes=self.node_num_classes).float()

        # Edges
        rand_vals_e = torch.rand(e0.shape, device=device)
        random_edge = torch.randint(0, self.edge_num_classes, e0.shape, device=device)
        e_t_raw = torch.where(rand_vals_e < p_keep, e0, random_edge)

        if self.use_projector:
            projected_edges = strict_projector_petri(x_t, e_t_raw, device)
        else:
            projected_edges = e_t_raw

        e_t_onehot = F.one_hot(projected_edges, num_classes=self.edge_num_classes).float()
        return x_t_onehot, e_t_onehot

    def reverse_diffusion_single(self, data, device, save_intermediate=True):
        num_nodes = data.x.size(0)

        x = torch.randint(0, self.node_num_classes, (num_nodes,), device=device)
        x = F.one_hot(x, num_classes=self.node_num_classes).float()
        e = torch.randint(0, self.edge_num_classes, (num_nodes, num_nodes), device=device)
        e = F.one_hot(e, num_classes=self.edge_num_classes).float()

        intermediate_graphs = []

        for t in range(self.T - 1, -1, -1):
            node_logits, edge_logits_list = self.forward(x, data.edge_index, data.batch, t)
            node_probs = F.softmax(node_logits, dim=1)
            x_labels = torch.multinomial(node_probs, num_samples=1).squeeze(1)
            x = F.one_hot(x_labels, num_classes=self.node_num_classes).float()

            if edge_logits_list and edge_logits_list[0].numel() > 0:
                edge_logits = edge_logits_list[0]
                edge_probs = F.softmax(edge_logits, dim=-1)
                flat_probs = edge_probs.view(-1, self.edge_num_classes)
                sampled_flat = torch.multinomial(flat_probs, num_samples=1).view(-1)
                candidate_edge_matrix = sampled_flat.view(num_nodes, num_nodes)
                current_node_labels = x.argmax(dim=1)

                if self.use_projector:
                    projected_edges = strict_projector_petri(current_node_labels, candidate_edge_matrix, device)
                    if t == 0:
                        projected_edges = strict_validation_petri(current_node_labels, projected_edges, device)
                else:
                    projected_edges = candidate_edge_matrix

                e = F.one_hot(projected_edges, num_classes=self.edge_num_classes).float()

            if save_intermediate:
                intermediate_graphs.append(Data(
                    x=x.clone(),
                    edge_index=(e.argmax(dim=-1) > 0).nonzero(as_tuple=False).t().contiguous()
                ))

        final_node_labels = x.argmax(dim=1)
        final_edge_labels = e.argmax(dim=-1)

        return final_node_labels, final_edge_labels.unsqueeze(0), intermediate_graphs

   
    def sample(self, data, device):
        return self.reverse_diffusion_single(data, device)
    

    def sample_conditional_and_save(self, n_nodes, batch_size, device,
                                output_dir='exp_outputs'):
        """
        Uses reverse diffusion to sample graphs and saves:
          - output_dir/final_graphs.pt         (list[Data])
          - output_dir/intermediate_steps.pt   (list[list[Data]])
        """
        os.makedirs(output_dir, exist_ok=True)

        final_graphs = []
        all_intermediate_steps = []

        for _ in range(batch_size):
            # Dense scaffold for message passing
            edge_list = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if i != j]
            edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
            x = torch.zeros(n_nodes, self.node_num_classes, device=device)
            data = Data(x=x, edge_index=edge_index)
            data.batch = torch.zeros(n_nodes, dtype=torch.long, device=device)

            final_nodes, final_edges, intermediate_steps = self.reverse_diffusion_single(
                data, device, save_intermediate=True
            )
            n_nodes = len(final_nodes)
            edge_index = (final_edges[0] != 0).nonzero(as_tuple=False).t().contiguous()  # shape [2, num_edges]
            x = torch.nn.functional.one_hot(final_nodes, num_classes=2).float()  # shape [n_nodes, 2]
            graph_data = Data(x=x, edge_index=edge_index)

            # graph_data = Data(
            #     x=F.one_hot(final_nodes, num_classes=self.node_num_classes).float(),
            #     edge_index=(final_edges[0] > 0).nonzero(as_tuple=False).t().contiguous()
            # )
            final_graphs.append(graph_data)
            all_intermediate_steps.append(intermediate_steps)

        torch.save(final_graphs, os.path.join(output_dir, 'final_graphs.pt'))
        torch.save(all_intermediate_steps, os.path.join(output_dir, 'intermediate_steps.pt'))

        print(f"✅ Saved {len(final_graphs)} final graphs to {output_dir}/final_graphs.pt")
        print(f"✅ Also saved intermediate diffusion steps at '{output_dir}/intermediate_steps.pt'")

        return final_nodes, final_edges


def compute_batch_loss(model, batch_data, T, device, edge_weight, node_marginal, edge_marginal, kl_lambda=0.1, constraint_lambda=1.0):
    """Training loss with node/edge CE, KL to marginals, and constraint penalty."""
    data_list = batch_data.to_data_list()
    total_loss = 0.0
    count = 0
    for data in data_list:
        true_n = data.n_nodes.item() if hasattr(data, 'n_nodes') else data.x.size(0)
        if true_n == 0:
            continue

        x0 = data.x[:true_n].argmax(dim=1)
        dense_adj = to_dense_adj(data.edge_index, max_num_nodes=true_n)[0]
        e0 = (dense_adj > 0).long()

        t_i = torch.randint(0, T, (1,)).item()
        x_t, e_t = model.forward_diffusion(x0, e0, t_i, device)

        edge_index_noisy = (e_t.argmax(dim=-1) > 0).nonzero(as_tuple=False).t().contiguous()
        data_i = Data(x=x_t, edge_index=edge_index_noisy)
        data_i.batch = torch.zeros(x_t.size(0), dtype=torch.long, device=device)
        node_logits, edge_logits_list = model(data_i.x, data_i.edge_index, data_i.batch, t=t_i)
        
        loss_node = F.cross_entropy(node_logits, x0.to(device))
        
        if edge_logits_list and edge_logits_list[0].numel() > 0:
            edge_logits = edge_logits_list[0]
            loss_edge = F.cross_entropy(edge_logits.view(-1, model.edge_num_classes),
                                        e0.to(device).view(-1),
                                        weight=edge_weight)
        else:
            loss_edge = 0.0

        node_probs = F.softmax(node_logits, dim=1)
        kl_node = kl_divergence(node_probs, node_marginal.to(device))
        if edge_logits_list and edge_logits_list[0].numel() > 0:
            edge_logits = edge_logits_list[0]
            edge_probs = F.softmax(edge_logits, dim=-1)
            edge_probs_avg = edge_probs.view(-1, model.edge_num_classes)
            kl_edge = kl_divergence(edge_probs_avg, edge_marginal.to(device))
        else:
            kl_edge = 0.0

        if edge_logits_list and edge_logits_list[0].numel() > 0:
            edge_probs = F.softmax(edge_logits, dim=-1)
            pred_edge_prob = edge_probs[..., 1]
            forbidden_mask = get_forbidden_mask(x0, device)
            forbidden_mask = forbidden_mask[:true_n, :true_n]
            constraint_loss = F.mse_loss(pred_edge_prob * forbidden_mask, torch.zeros_like(pred_edge_prob))
        else:
            constraint_loss = 0.0

        loss = loss_node + loss_edge + kl_lambda * (kl_node + kl_edge) + constraint_lambda * constraint_loss
        total_loss += loss
        count += 1

    avg_loss = total_loss / count if count > 0 else torch.tensor(0.0, device=device)
    return avg_loss


def train_model(model, dataloader, optimizer, device, edge_weight, node_marginal, edge_marginal, epochs=20, T=100):
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = compute_batch_loss(model, batch, T, device, edge_weight, node_marginal, edge_marginal, kl_lambda=0.1, constraint_lambda=1.0)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")


def compute_marginal_probs(dataset, device):
    """Marginal probabilities for nodes and edges from the dataset."""
    node_counts = torch.zeros(2, device=device)
    edge_counts = torch.zeros(2, device=device)
    total_nodes = 0
    total_edges = 0
    from torch_geometric.utils import to_dense_adj
    for data in dataset:
        true_n = data.n_nodes.item() if hasattr(data, 'n_nodes') else data.x.size(0)
        if true_n == 0:
            continue
        node_labels = data.x[:true_n].argmax(dim=1)
        node_counts += torch.bincount(node_labels, minlength=2).float().to(device)
        total_nodes += true_n
        
        dense_adj = to_dense_adj(data.edge_index, max_num_nodes=true_n)[0]
        e0 = (dense_adj > 0).long()
        total_edges += e0.numel()
        edge_counts += torch.bincount(e0.view(-1), minlength=2).float().to(device)
    
    node_probs = node_counts / total_nodes if total_nodes > 0 else torch.ones(2, device=device) / 2.0
    edge_probs = edge_counts / total_edges if total_edges > 0 else torch.ones(2, device=device) / 2.0
    return node_probs, edge_probs


def compute_edge_weights(dataset, device):
    total_edges = 0
    class_counts = torch.zeros(2, device=device)
    for data in dataset:
        true_n = data.n_nodes.item() if hasattr(data, 'n_nodes') else data.x.size(0)
        if true_n == 0:
            continue
        dense_adj = to_dense_adj(data.edge_index, max_num_nodes=true_n)[0]
        e0 = (dense_adj > 0).long()
        counts = torch.bincount(e0.view(-1), minlength=2).to(device)
        class_counts += counts
        total_edges += e0.numel()
    
    class_counts[class_counts == 0] = 1.0
    weights = total_edges / (2.0 * class_counts)
    weights = weights / weights.sum()
    return weights


def kl_divergence(pred_probs, marginal_probs):
    """KL(pred || marginal) averaged over items."""
    marginal_probs = marginal_probs.unsqueeze(0)
    kl = torch.sum(pred_probs * (torch.log(pred_probs + 1e-8) - torch.log(marginal_probs + 1e-8)), dim=1)
    return kl.mean()


# 3 Petri Training + Sampling Script (CLI)
from torch_geometric.loader import DataLoader
import argparse, time

def main_train(net_type='machine'):           # 'machine' | 'assembly' | 'disassembly'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_root = f'petri_{net_type}_dataset'
    os.makedirs(os.path.join(dataset_root, 'raw'),       exist_ok=True)
    os.makedirs(os.path.join(dataset_root, 'processed'), exist_ok=True)

    dataset = PetriNetDataset(root=dataset_root, split='train')

    edge_weight             = compute_edge_weights(dataset, device)
    node_marg, edge_marg    = compute_marginal_probs(dataset, device)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    model      = LightweightPetriDiffusion(T=100, hidden_dim=12, time_embed_dim=16).to(device)
    optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3)

    print(f"\n▶ Starting training for {net_type.upper()} ({len(dataset)} graphs)")
    start = time.perf_counter()

    train_model(model, dataloader, optimizer, device,
                edge_weight=edge_weight,
                node_marginal=node_marg,
                edge_marginal=edge_marg,
                epochs=30, T=100)

    elapsed = time.perf_counter() - start
    print(f"⏱  Finished in {elapsed/60:.1f} min ({elapsed:.1f} s)\n")

    torch.save(model.state_dict(), f'petri_{net_type}_model.pth')
    print(f"✅ Weights saved to petri_{net_type}_model.pth")


@torch.no_grad()
def main_sample(net_type: str, n_nodes: int, n_samples: int, out_dir: str, weights_path: Optional[str] = None):
    """Minimal sampling entrypoint that reuses the model's own sample_conditional_and_save."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LightweightPetriDiffusion(T=100, hidden_dim=12, time_embed_dim=16).to(device)

    if weights_path is None:
        weights_path = f'petri_{net_type}_model.pth'
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Could not find weights: {weights_path}")

    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    model.sample_conditional_and_save(
        n_nodes=n_nodes,
        batch_size=n_samples,
        device=device,
        output_dir=out_dir
    )


# ── main.py (final part) ─────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'sample'],
                        help="Select 'train' to train models or 'sample' to generate graphs.")
    parser.add_argument('--net_type', type=str, default='machine',
                        choices=['machine', 'assembly', 'disassembly', 'all'],
                        help="Type of Petri-net for training or sampling.")
    parser.add_argument('--n_nodes', type=int, default=None, help="Number of nodes when sampling.")
    parser.add_argument('--n_samples', type=int, default=5, help="How many graphs to sample.")
    parser.add_argument('--out_dir', type=str, default='exp_outputs',
                        help="Output directory for sampling (default: exp_outputs).")
    parser.add_argument('--weights_path', type=str, default=None, help="Optional explicit path to weights for sampling.")

    args = parser.parse_args()

    if args.mode == 'train':
        if args.net_type == 'all':
            for t in ['machine', 'assembly', 'disassembly']:
                main_train(net_type=t)
        else:
            main_train(net_type=args.net_type)
    else:
        if args.net_type == 'all':
            raise ValueError("Sampling with --net_type all is not supported. Choose one of {machine, assembly, disassembly}.")
        if args.n_nodes is None:
            raise ValueError("Please provide --n_nodes when using --mode sample.")
        os.makedirs(args.out_dir, exist_ok=True)
        main_sample(
            net_type=args.net_type,
            n_nodes=args.n_nodes,
            n_samples=args.n_samples,
            out_dir=args.out_dir,
            weights_path=args.weights_path
        )
        print(f"✅ Sampling finished. Files saved in '{args.out_dir}'.")
