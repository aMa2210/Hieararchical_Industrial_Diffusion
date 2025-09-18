# final_petri_diffusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import to_dense_batch, to_dense_adj, subgraph
from torch_geometric.data import Data, DataLoader, Batch
import os
import networkx as nx



def get_forbidden_mask(node_labels, device):
    """
    Compute a binary mask of shape (n, n) indicating forbidden edges.
    For Petri nets, an edge is forbidden if:
      - It is a self-loop (i.e., diagonal elements must be 0).
      - It connects nodes of the same type.
      
    Args:
        node_labels (torch.Tensor): Tensor of shape (n,) with node types (e.g., 0 for Place, 1 for Transition).
        device: Torch device.
        
    Returns:
        A tensor mask of shape (n, n) with 1 indicating a forbidden edge, and 0 allowed.
    """
    n = node_labels.size(0)
    diag_mask = torch.eye(n, dtype=torch.bool, device=device)
    allowed_mask = (node_labels.unsqueeze(1) != node_labels.unsqueeze(0))
    forbidden_mask = diag_mask | (~allowed_mask)
    # !!tbd add punishment about isolated nodes
    return forbidden_mask.float()  # float mask: 1.0 for forbidden, 0.0 for allowed


def get_forbidden_edges_petri(node_labels):
    """
    Given a tensor of node labels (shape: (n,)), returns a tensor of forbidden edge indices.
    For Petri nets, an edge is forbidden if it is a self-loop or if it connects nodes of the same type.
    
    Args:
        node_labels (torch.Tensor): Tensor of shape (n,) with node types (e.g., 0 for Place, 1 for Transition).
        
    Returns:
        A tensor of shape (k, 2) listing the forbidden edges (only upper triangular, i.e., i < j).
    """
    n = node_labels.size(0)
    forbidden = []
    for i in range(n):
        for j in range(i, n):  # include self-loops
            if i == j or node_labels[i] == node_labels[j]:
                forbidden.append([i, j])
    if len(forbidden) > 0:
        return torch.tensor(forbidden, device=node_labels.device)
    else:
        return torch.empty((0, 2), dtype=torch.long, device=node_labels.device)

def strict_projector_petri(node_labels, candidate_edge_matrix, device):
    """
    Adapt the projector idea to Petri nets:
    Given the current node labels and a candidate binary edge matrix (shape: (n, n)),
    zero out the edges that are forbidden (i.e., self-loops or same-type connections).
    
    This is a simpler projector that uses a forbidden-edges mask computed from node_labels.
    
    Args:
        node_labels (torch.Tensor): Tensor of shape (n,) with predicted node types.
        candidate_edge_matrix (torch.Tensor): Tensor of shape (n, n) (binary, 0/1) from candidate predictions.
        device: Torch device.
        
    Returns:
        A new edge matrix (n, n) with forbidden edges zeroed out.
    """
    n = node_labels.size(0)
    # Get forbidden edge indices (only for i <= j)
    forbidden_edges = get_forbidden_edges_petri(node_labels)
    projected = candidate_edge_matrix.clone()
    if forbidden_edges.numel() > 0:
        for edge in forbidden_edges:
            i, j = edge[0].item(), edge[1].item()
            # Zero out both (i, j) and (j, i) to enforce undirected constraint if needed.
            projected[i, j] = 0
            projected[j, i] = 0
    return projected



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


def get_sinusoidal_embedding(t, embedding_dim):
    """
    Compute sinusoidal embeddings.
    Args:
        t: A tensor of shape (B, 1) or (1,) representing the timesteps (normalized).
        embedding_dim: Dimensionality of the embedding.
    Returns:
        Tensor of shape (B, embedding_dim) with sinusoidal embeddings.
    """
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

class LightweightPetriDiffusion(nn.Module):
    def __init__(self, T=100, hidden_dim=12, beta_start=0.0001, beta_end=0.02, time_embed_dim=16, nhead=4, dropout=0.1, use_projector=True):
        """
        Args:
            T (int): Total diffusion steps.
            hidden_dim (int): Hidden dimension.
            beta_start, beta_end: Noise schedule parameters.
            time_embed_dim (int): Dimensionality for time embedding.
            nhead (int): Number of heads in the transformer.
            dropout (float): Dropout rate.
        """
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
        # Instead of a learned MLP for time, we use sinusoidal embeddings (optionally with a linear projection).
        self.time_linear = nn.Linear(time_embed_dim, time_embed_dim)
        
        # Transformer-based layers for node denoising.
        # The input dimension is node_num_classes + time_embed_dim.
        in_channels = self.node_num_classes + time_embed_dim
        self.transformer1 = TransformerConv(in_channels, hidden_dim, heads=nhead, concat=False, dropout=dropout)
        self.transformer2 = TransformerConv(hidden_dim, hidden_dim, heads=nhead, concat=False, dropout=dropout)
        self.node_out = nn.Linear(hidden_dim, self.node_num_classes)
        
        # MLP for edge prediction remains similar.
        self.edge_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.edge_num_classes)
        )
    
    def forward(self, x, edge_index, batch, t):
        """
        Forward pass (denoising).
        Args:
            x (torch.Tensor): Noisy node features (one-hot, shape (total_nodes, node_num_classes)).
            edge_index (torch.Tensor): Edge index.
            batch (torch.Tensor): Batch assignment.
            t (int): Diffusion step.
        Returns:
            node_logits: (total_nodes, node_num_classes)
            edge_logits_list: List of edge logits (each tensor of shape (n_i, n_i, edge_num_classes)).
        """
        # Normalize t and compute sinusoidal embedding.
        t_tensor = torch.tensor([t], dtype=torch.float, device=x.device) / self.T
        t_embed = get_sinusoidal_embedding(t_tensor, self.time_embed_dim)
        t_embed = self.time_linear(t_embed)  # Optional linear projection.
        t_embed_full = t_embed.repeat(x.size(0), 1)
        
        # Concatenate node features with time embedding.
        x_input = torch.cat([x, t_embed_full], dim=1)
        h = F.relu(self.transformer1(x_input, edge_index))
        h = F.relu(self.transformer2(h, edge_index))
        node_logits = self.node_out(h)
        
        # For edge prediction, convert node embeddings to dense batch.
        h_dense, mask = to_dense_batch(h, batch)  # shape: (batch_size, max_nodes, hidden_dim)
        batch_size, max_nodes, _ = h_dense.shape
        edge_logits_list = []
        for i in range(batch_size):
            num_nodes = int(mask[i].sum().item())
            if num_nodes == 0:
                edge_logits_list.append(torch.empty(0))
                continue
            h_i = h_dense[i, :num_nodes, :]  # (num_nodes, hidden_dim)
            h_i_exp1 = h_i.unsqueeze(1).expand(-1, num_nodes, -1)
            h_i_exp2 = h_i.unsqueeze(0).expand(num_nodes, -1, -1)
            edge_input = torch.cat([h_i_exp1, h_i_exp2], dim=-1)
            edge_logits = self.edge_mlp(edge_input)
            edge_logits_list.append(edge_logits)
        return node_logits, edge_logits_list
    
    def forward_diffusion(self, x0, e0, t, device):
        """
        Constraint-enforced forward diffusion: Adds noise to ground-truth while ensuring that the constraints
        (no forbidden edges such as self-loops or same-type connections) remain satisfied.

        Args:
            x0: Tensor of ground truth node labels (shape: [n_nodes])
            e0: Tensor of ground truth edge labels (shape: [n_nodes, n_nodes])
            t: Current timestep
            device: Torch device

        Returns:
            Tuple (x_t_onehot, e_t_onehot) with constraints enforced.
        """
        p_keep = self.alpha_bar[t].item()

        # Nodes: random noising as usual
        rand_vals = torch.rand(x0.shape, device=device)
        random_node = torch.randint(0, self.node_num_classes, x0.shape, device=device)
        x_t = torch.where(rand_vals < p_keep, x0, random_node)
        x_t_onehot = F.one_hot(x_t, num_classes=self.node_num_classes).float()

        # Edges: random noising with constraint enforcement
        rand_vals_e = torch.rand(e0.shape, device=device)
        random_edge = torch.randint(0, self.edge_num_classes, e0.shape, device=device)
        e_t_raw = torch.where(rand_vals_e < p_keep, e0, random_edge)

        # Enforce constraints by projecting edges
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
                                output_dir='generated_graphs'):
        os.makedirs(output_dir, exist_ok=True)

        final_graphs = []
        all_intermediate_steps = []

        for graph_idx in range(batch_size):
            # Create initial dummy fully-connected graph
            edge_list = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if i != j]
            edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
            x = torch.zeros(n_nodes, self.node_num_classes, device=device)
            data = Data(x=x, edge_index=edge_index)
            data.batch = torch.zeros(n_nodes, dtype=torch.long, device=device)

            # Run reverse diffusion with intermediate steps collection
            final_nodes, final_edges, intermediate_steps = self.reverse_diffusion_single(
                data, device, save_intermediate=True
            )

            # Append final graph
            graph_data = Data(
                x=F.one_hot(final_nodes, num_classes=self.node_num_classes).float(),
                edge_index=(final_edges[0] > 0).nonzero(as_tuple=False).t().contiguous()
            )
            final_graphs.append(graph_data)

            # Append intermediate steps (for this specific graph)
            all_intermediate_steps.append(intermediate_steps)

        # Save all final graphs clearly into one file
        torch.save(final_graphs, os.path.join(output_dir, 'final_graphs.pt'))

        # Save all intermediate steps clearly into another file
        torch.save(all_intermediate_steps, os.path.join(output_dir, 'intermediate_steps.pt'))

        print(f"✅ Saved {len(final_graphs)} final graphs to {output_dir}/final_graphs.pt")
        print(f"✅ Also saved intermediate diffusion steps at '{output_dir}/intermediate_steps.pt'")

        return final_nodes, final_edges





def compute_batch_loss(model, batch_data, T, device, edge_weight, node_marginal, edge_marginal, kl_lambda=0.1, constraint_lambda=1.0):
    """
    Computes the training loss on a batch of graphs using cross-entropy losses for nodes and edges,
    KL divergence regularization based on the marginal distributions, and an additional constraint loss
    that penalizes predictions of forbidden edges.
    
    Args:
        model: The diffusion model.
        batch_data: A batched PyG Data object.
        T: Total diffusion steps.
        device: Torch device.
        edge_weight: Weight tensor for edge cross-entropy.
        node_marginal: Marginal distribution for node classes.
        edge_marginal: Marginal distribution for edge classes.
        kl_lambda (float): Weight for the KL divergence loss terms.
        constraint_lambda (float): Weight for the forbidden edge penalty term.
    
    Returns:
        Average loss over the batch.
    """
    data_list = batch_data.to_data_list()
    total_loss = 0.0
    count = 0
    for data in data_list:
        true_n = data.n_nodes.item() if hasattr(data, 'n_nodes') else data.x.size(0)
        if true_n == 0:
            continue

        # Ground-truth node labels from one-hot data.
        x0 = data.x[:true_n].argmax(dim=1)  # shape: (true_n,)
        # Compute dense adjacency (ground-truth edge labels; binary: 0=no edge, 1=edge exists)
        dense_adj = to_dense_adj(data.edge_index, max_num_nodes=true_n)[0]
        e0 = (dense_adj > 0).long()  # shape: (true_n, true_n)

        t_i = torch.randint(0, T, (1,)).item()
        x_t, e_t = model.forward_diffusion(x0, e0, t_i, device)

        edge_index_noisy = (e_t.argmax(dim=-1) > 0).nonzero(as_tuple=False).t().contiguous() # format the e_t so that it matches the input style of Data
        data_i = Data(x=x_t, edge_index=edge_index_noisy)
        data_i.batch = torch.zeros(x_t.size(0), dtype=torch.long, device=device) # the tensor to indicate the belongings of the edges, since here all the nodes are in the same graph(i.e. graph[0]), the value of the tensor is all 0
        node_logits, edge_logits_list = model(data_i.x, data_i.edge_index, data_i.batch, t=t_i)
        
        # Node loss.
        loss_node = F.cross_entropy(node_logits, x0.to(device))
        
        # Edge loss.
        if edge_logits_list and edge_logits_list[0].numel() > 0:
            edge_logits = edge_logits_list[0]  # shape: (true_n, true_n, edge_num_classes)
            loss_edge = F.cross_entropy(edge_logits.view(-1, model.edge_num_classes),
                                        e0.to(device).view(-1),
                                        weight=edge_weight) #the rarer the kind of edge is (exist or not), the more loss is imposed
                                                            #in this case, since for most of the nodes, edge doesn't exist between them,
                                                            #if the model doesn't generate the edge that was originally exist, it gets much loss.
        else:
            loss_edge = 0.0

        # KL divergence losses:
        node_probs = F.softmax(node_logits, dim=1)
        kl_node = kl_divergence(node_probs, node_marginal.to(device))
        if edge_logits_list and edge_logits_list[0].numel() > 0:
            edge_logits = edge_logits_list[0]
            edge_probs = F.softmax(edge_logits, dim=-1)
            edge_probs_avg = edge_probs.view(-1, model.edge_num_classes)
            kl_edge = kl_divergence(edge_probs_avg, edge_marginal.to(device))
        else:
            kl_edge = 0.0

        # Constraint loss: penalize forbidden edge predictions.
        # Get predicted edge probabilities for "edge exists" (assume class 1).
        if edge_logits_list and edge_logits_list[0].numel() > 0:
            edge_probs = F.softmax(edge_logits, dim=-1)  # shape: (true_n, true_n, edge_num_classes)
            pred_edge_prob = edge_probs[..., 1]  # probability for class 1
            # Compute forbidden mask from ground-truth node labels.
            # tbd
            # !!!here, there may exists a problem, the code use x0 to get the forbidden_mask instead of the graph the one it just generated
            # !!!but if t is a large number, the noise is heavy, the generated graph is not much as similar as the ground truth, in this case
            # !!!using the constraint of the ground truth graph is inappropriate
            forbidden_mask = get_forbidden_mask(x0, device)  # shape: (true_n, true_n)
            forbidden_mask = forbidden_mask[:true_n, :true_n]
            # Use MSE loss to push predicted probability to 0 where forbidden.
            constraint_loss = F.mse_loss(pred_edge_prob * forbidden_mask, torch.zeros_like(pred_edge_prob))
        else:
            constraint_loss = 0.0

        loss = loss_node + loss_edge + kl_lambda * (kl_node + kl_edge) + constraint_lambda * constraint_loss
        total_loss += loss
        count += 1

    if count > 0:
        avg_loss = total_loss / count
    else:
        avg_loss = torch.tensor(0.0, device=device)
    return avg_loss





def train_model(model, dataloader, optimizer, device, edge_weight, node_marginal, edge_marginal, epochs=20, T=100):
        model.train()   # switch model to train mode, not the train function itself
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
    """
    Compute marginal probabilities for nodes and edges from the dataset.
    
    Args:
        dataset: A PyTorch Geometric Dataset (e.g., PetriNetDataset).
        device: Torch device.
    
    Returns:
        node_probs: Tensor of shape (2,) with marginal probabilities for node classes.
        edge_probs: Tensor of shape (2,) with marginal probabilities for edge classes.
    """
    node_counts = torch.zeros(2, device=device)
    edge_counts = torch.zeros(2, device=device)
    total_nodes = 0
    total_edges = 0
    from torch_geometric.utils import to_dense_adj
    for data in dataset:
        # Get true number of nodes.
        true_n = data.n_nodes.item() if hasattr(data, 'n_nodes') else data.x.size(0)
        if true_n == 0:
            continue
        # For nodes: data.x is one-hot, so get labels via argmax.
        node_labels = data.x[:true_n].argmax(dim=1)
        node_counts += torch.bincount(node_labels, minlength=2).float().to(device)
        total_nodes += true_n
        
        # For edges: compute dense adjacency and binarize.
        dense_adj = to_dense_adj(data.edge_index, max_num_nodes=true_n)[0]  # (true_n, true_n)
        e0 = (dense_adj > 0).long()
        edge_counts += torch.bincount(e0.view(-1), minlength=2).float().to(device)
        total_edges += e0.numel()

    
    # Avoid division by zero.
    if total_nodes > 0:
        node_probs = node_counts / total_nodes
    else:
        node_probs = torch.ones(2, device=device) / 2.0
    if total_edges > 0:
        edge_probs = edge_counts / total_edges
    else:
        edge_probs = torch.ones(2, device=device) / 2.0
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
        counts = torch.bincount(e0.view(-1), minlength=2).to(device)  # <-- corrección aquí
        class_counts += counts
        total_edges += e0.numel()
    
    class_counts[class_counts == 0] = 1.0
    weights = total_edges / (2.0 * class_counts)
    weights = weights / weights.sum()
    return weights


def kl_divergence(pred_probs, marginal_probs):
    """
    Compute KL divergence between predicted probabilities and marginal distribution.
    pred_probs: Tensor of shape (N, num_classes)
    marginal_probs: Tensor of shape (num_classes,)
    Returns:
        KL divergence (scalar)
    """
    # Ensure marginal_probs is broadcastable.
    marginal_probs = marginal_probs.unsqueeze(0)
    kl = torch.sum(pred_probs * (torch.log(pred_probs + 1e-8) - torch.log(marginal_probs + 1e-8)), dim=1)
    return kl.mean()




