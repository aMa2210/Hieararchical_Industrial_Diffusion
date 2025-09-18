# industrial_diffusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch_geometric.data import Data
import os

# 0: MACHINE, 1: BUFFER, 2: ASSEMBLY, 3: DISASSEMBLY
def get_forbidden_mask(node_labels, device):
    n = node_labels.size(0)
    diag_mask = torch.eye(n, dtype=torch.bool, device=device)

    allowed_mask = torch.ones((n, n), dtype=torch.bool, device=device)
    for i in range(n):
        for j in range(n):
            if node_labels[i] == 1 and node_labels[j] == 1:  # BUFFER-BUFFER forbidden
                allowed_mask[i, j] = False

    forbidden_mask = diag_mask | (~allowed_mask)
    return forbidden_mask.float()

def validate_constraints(edge_matrix, node_labels, device):
    """
    Devuelve True si `edge_matrix` cumple todas las restricciones industriales:
    - No self‑loops.
    - No Buffer→Buffer.
    - Máximos de conexiones según tipo.
    - No bidireccionales.
    """
    forbidden = get_forbidden_mask(node_labels, device)
    # 1) Ninguna arista donde forbidden==1
    if (edge_matrix * forbidden).any():
        return False
    # 2) Máximos por tipo (ejemplo MACHINE→BUFFER <=1)
    n = node_labels.size(0)
    # Cuenta salidas MACHINE→BUFFER
    for i in range(n):
        if node_labels[i]==0:
            if edge_matrix[i, node_labels==1].sum() > 1:
                return False
    # (añade aquí otras comprobaciones específicas si lo deseas)
    return True

    



def strict_projector_industrial(node_labels, candidate_edge_matrix, device):
    """
    Proyecta candidate_edge_matrix (binaria) a una matriz de adyacencia
    que respete las constraints:
      - MACHINE -> BUFFER (máx 1 vez por MACHINE como fuente).
      - BUFFER -> MACHINE (máx 1 vez por MACHINE como destino).
      - ASSEMBLY con 2 entradas (BUFFER -> ASSEMBLY) y 1 salida (ASSEMBLY -> BUFFER).
      - DISASSEMBLY con 1 entrada (BUFFER -> DISASSEMBLY) y 2 salidas (DISASSEMBLY -> BUFFER).
      - SIN EDGES BIDIRECCIONALES (si ya existe j->i, no se acepta i->j).
    """
    n = node_labels.size(0)
    projected_edges = torch.zeros((n, n), dtype=torch.long, device=device)

    # Para máquinas, un uso como fuente y un uso como destino
    machine_sources_available = set(i for i in range(n) if node_labels[i] == 0)
    machine_targets_available = set(i for i in range(n) if node_labels[i] == 0)

    # Assembly: 2 entradas (BUFFERS) y 1 salida (BUFFER)
    assembly_inputs = {i: 0 for i in range(n) if node_labels[i] == 2}
    assembly_outputs = {i: 0 for i in range(n) if node_labels[i] == 2}

    # Disassembly: 1 entrada (BUFFER) y 2 salidas (BUFFER)
    disassembly_inputs = {i: 0 for i in range(n) if node_labels[i] == 3}
    disassembly_outputs = {i: 0 for i in range(n) if node_labels[i] == 3}

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # Evitar bidireccionalidad: si j->i ya está aprobado, no permitir i->j
            if projected_edges[j, i] == 1:
                continue

            if candidate_edge_matrix[i, j] == 1:
                src_type = node_labels[i].item()
                dst_type = node_labels[j].item()

                # MACHINE -> BUFFER
                if src_type == 0 and dst_type == 1 and i in machine_sources_available:
                    projected_edges[i, j] = 1
                    machine_sources_available.remove(i)

                # BUFFER -> MACHINE
                elif src_type == 1 and dst_type == 0 and j in machine_targets_available:
                    projected_edges[i, j] = 1
                    machine_targets_available.remove(j)

                # BUFFER -> ASSEMBLY (máx 2)
                elif src_type == 1 and dst_type == 2 and j in assembly_inputs and assembly_inputs[j] < 2:
                    projected_edges[i, j] = 1
                    assembly_inputs[j] += 1

                # ASSEMBLY -> BUFFER (máx 1)
                elif src_type == 2 and dst_type == 1 and i in assembly_outputs and assembly_outputs[i] < 1:
                    projected_edges[i, j] = 1
                    assembly_outputs[i] += 1

                # BUFFER -> DISASSEMBLY (máx 1)
                elif src_type == 1 and dst_type == 3 and j in disassembly_inputs and disassembly_inputs[j] < 1:
                    projected_edges[i, j] = 1
                    disassembly_inputs[j] += 1

                # DISASSEMBLY -> BUFFER (máx 2)
                elif src_type == 3 and dst_type == 1 and i in disassembly_outputs and disassembly_outputs[i] < 2:
                    projected_edges[i, j] = 1
                    disassembly_outputs[i] += 1

    return projected_edges




def get_sinusoidal_embedding(t, embedding_dim):
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


def kl_divergence(pred_probs, marginal_probs):
    marginal_probs = marginal_probs.unsqueeze(0)
    kl = torch.sum(pred_probs * (torch.log(pred_probs + 1e-8) - torch.log(marginal_probs + 1e-8)), dim=1)
    return kl.mean()

def compute_batch_loss(model, batch_data, T, device, edge_weight, node_marginal, edge_marginal, kl_lambda=0.1, constraint_lambda=1.0):
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
            loss = compute_batch_loss(model, batch, T, device, edge_weight, node_marginal, edge_marginal)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")


class LightweightIndustrialDiffusion(nn.Module):
    def __init__(self, T=100, hidden_dim=12, beta_start=0.0001, beta_end=0.02, time_embed_dim=16, nhead=4, dropout=0.1, use_projector=True):
        super().__init__()
        self.T = T
        self.beta_schedule = torch.linspace(beta_start, beta_end, T)
        self.alpha = 1 - self.beta_schedule
        self.register_buffer('alpha_bar', torch.cumprod(self.alpha, dim=0))
        self.use_projector = use_projector
    
        self.node_num_classes = 4  # MACHINE, BUFFER, ASSEMBLY, DISASSEMBLY
        self.edge_num_classes = 2

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
        t_embed = get_sinusoidal_embedding(t_tensor, self.time_linear.in_features)
        t_embed = self.time_linear(t_embed).repeat(x.size(0), 1)

        x_input = torch.cat([x, t_embed], dim=1)
        h = F.relu(self.transformer1(x_input, edge_index))
        h = F.relu(self.transformer2(h, edge_index))
        node_logits = self.node_out(h)

        h_dense, mask = to_dense_batch(h, batch)
        batch_size, max_nodes, _ = h_dense.shape
        edge_logits_list = []
        for i in range(batch_size):
            num_nodes = int(mask[i].sum().item())
            h_i = h_dense[i, :num_nodes, :]
            edge_input = torch.cat([h_i.unsqueeze(1).expand(-1, num_nodes, -1), h_i.unsqueeze(0).expand(num_nodes, -1, -1)], dim=-1)
            edge_logits = self.edge_mlp(edge_input)
            edge_logits_list.append(edge_logits)

        return node_logits, edge_logits_list

    def forward_diffusion(self, x0, e0, t, device):
        p_keep = self.alpha_bar[t].item()
        rand_vals = torch.rand(x0.shape, device=device)
        random_node = torch.randint(0, self.node_num_classes, x0.shape, device=device)
        x_t = torch.where(rand_vals < p_keep, x0, random_node)
        x_t_onehot = F.one_hot(x_t, num_classes=self.node_num_classes).float()

        rand_vals_e = torch.rand(e0.shape, device=device)
        random_edge = torch.randint(0, self.edge_num_classes, e0.shape, device=device)
        e_t_raw = torch.where(rand_vals_e < p_keep, e0, random_edge)

        if self.use_projector:
            projected_edges = strict_projector_industrial(x_t, e_t_raw, device)
        else:
            projected_edges = e_t_raw # No se aplica projector

        e_t_onehot = F.one_hot(projected_edges.long(), num_classes=self.edge_num_classes).float()

        return x_t_onehot, e_t_onehot

    def sample(self, data, device):
        return self.reverse_diffusion_single(data, device)

    def sample_conditional_and_save(self, n_nodes, batch_size, device, output_dir='generated_graphs'):
        os.makedirs(output_dir, exist_ok=True)

        final_graphs = []
        all_intermediate_steps = []

        for graph_idx in range(batch_size):
            edge_list = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if i != j]
            edge_index = torch.tensor(edge_list, dtype=torch.long, device=device).t().contiguous()
            x = torch.zeros(n_nodes, self.node_num_classes, device=device)
            data = Data(x=x, edge_index=edge_index)
            data.batch = torch.zeros(n_nodes, dtype=torch.long, device=device)

            final_nodes, final_edges, intermediate_steps = self.reverse_diffusion_single(
                data, device, save_intermediate=True
            )

            graph_data = Data(
                x=F.one_hot(final_nodes, num_classes=self.node_num_classes).float(),
                edge_index=(final_edges[0] > 0).nonzero(as_tuple=False).t().contiguous()
            )
            final_graphs.append(graph_data)
            all_intermediate_steps.append(intermediate_steps)

        torch.save(final_graphs, os.path.join(output_dir, 'final_graphs.pt'))
        torch.save(all_intermediate_steps, os.path.join(output_dir, 'intermediate_steps.pt'))

        print(f"✅ Saved {len(final_graphs)} final graphs to {output_dir}/final_graphs.pt")
        print(f"✅ Saved intermediate steps at '{output_dir}/intermediate_steps.pt'")

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

                max_attempts = 5
                current_node_labels = x.argmax(dim=1)
                for attempt in range(max_attempts):
                    sampled_flat = torch.multinomial(flat_probs, num_samples=1).view(-1)
                    candidate_edge_matrix = sampled_flat.view(num_nodes, num_nodes)
                    if self.use_projector:
                        projected = strict_projector_industrial(current_node_labels, candidate_edge_matrix, device)
                    else:
                        projected = candidate_edge_matrix
                    if validate_constraints(projected, current_node_labels, device):
                        e = F.one_hot(projected, num_classes=self.edge_num_classes).float()
                        break
                else:
                    # Si tras MAX_ATTEMPTS sigue inválido, usamos el último proyectado
                    e = F.one_hot(projected, num_classes=self.edge_num_classes).float()


            if save_intermediate:
                intermediate_graphs.append(Data(
                    x=x.clone(),
                    edge_index=(e.argmax(dim=-1) > 0).nonzero(as_tuple=False).t().contiguous()
                ))

        final_node_labels = x.argmax(dim=1)
        final_edge_labels = e.argmax(dim=-1)

        return final_node_labels, final_edge_labels.unsqueeze(0), intermediate_graphs



    def reverse_diffusion_single_counts(self, data, pinned_mask, device, save_intermediate=True):
        """
        Similar to reverse_diffusion_single, but it ONLY re-samples those nodes that are NOT fixed (pinned_mask[i] = False).
        pinned_mask is a boolean tensor of size [num_nodes].

        Example:
        pinned_mask = [True, True, False, ...] ⇒ the types of nodes 0 and 1 are preserved, and the rest are re-sampled.
        """
        import torch
        import torch.nn.functional as F
        from torch_geometric.data import Data
        
        num_nodes = data.x.size(0)

        # Inicializa e sin aristas (todas clase 0 = no-edge).
        e = torch.zeros((num_nodes, num_nodes, self.edge_num_classes), device=device)
        e[:, :, 0] = 1

        # x actual (arranca en data.x, que ya contiene la info pineada)
        x = data.x.clone()

        intermediate_graphs = []

        for t in range(self.T - 1, -1, -1):
            node_logits, edge_logits_list = self.forward(x, data.edge_index, data.batch, t)

            node_probs = F.softmax(node_logits, dim=1)
            sampled_labels = torch.multinomial(node_probs, num_samples=1).squeeze(1)  # (num_nodes,)

            # Actualizamos x SOLO para nodos que no estén pineados
            current_x_int = x.argmax(dim=1)  # entero
            for i in range(num_nodes):
                if not pinned_mask[i]:
                    current_x_int[i] = sampled_labels[i]

            x = F.one_hot(current_x_int, num_classes=self.node_num_classes).float()

            # Muestreo de edges
            if edge_logits_list and edge_logits_list[0].numel() > 0:
                edge_logits = edge_logits_list[0]  # (num_nodes, num_nodes, 2)
                edge_probs = F.softmax(edge_logits, dim=-1)
                candidate_edge_matrix = torch.multinomial(edge_probs.view(-1, self.edge_num_classes), 1).view(num_nodes, num_nodes)

                current_node_labels = x.argmax(dim=1)
                projected_edges = strict_projector_industrial(current_node_labels, candidate_edge_matrix, device)
                e = F.one_hot(projected_edges.long(), num_classes=self.edge_num_classes).float()

            if save_intermediate:
                new_data = Data(
                    x=x.clone(),
                    edge_index=(e.argmax(dim=-1) > 0).nonzero(as_tuple=False).t().contiguous()
                )
                intermediate_graphs.append(new_data)

        final_node_labels = x.argmax(dim=1)
        final_edge_labels = e.argmax(dim=-1)

        return final_node_labels, final_edge_labels.unsqueeze(0), intermediate_graphs
    
    def sample_conditional_allpinned_and_save(
        self,
        num_machines: int,
        num_assemblies: int,
        num_disassemblies: int,
        num_buffers: int,
        batch_size: int,
        device,
        output_dir='guided_graphs_allpinned'
    ):
        import os
        import torch
        import torch.nn.functional as F
        from torch_geometric.data import Data

        os.makedirs(output_dir, exist_ok=True)

        final_graphs = []
        all_intermediate_steps = []

        total_nodes = num_machines + num_assemblies + num_disassemblies + num_buffers

        # Arma la lista de tipos de nodo que quieres (por ejemplo, 0=Machine,1=Buffer,2=Assembly,3=Disassembly)
        node_types_list = ([0] * num_machines +
                        [1] * num_buffers +
                        [2] * num_assemblies +
                        [3] * num_disassemblies)

        for graph_idx in range(batch_size):
            # Mezcla la lista, para que el orden sea aleatorio
            perm = torch.randperm(total_nodes)
            pinned_types = torch.tensor(node_types_list, device=device)[perm]

            # Convierte a one-hot
            pinned_x = F.one_hot(pinned_types, num_classes=self.node_num_classes).float()

            # Todos los nodos están pineados => pinned_mask = True para todos
            pinned_mask = torch.ones(total_nodes, dtype=torch.bool, device=device)

            # Grafo inicial sin aristas
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            data = Data(x=pinned_x.clone(), edge_index=edge_index)
            data.batch = torch.zeros(total_nodes, dtype=torch.long, device=device)

            # Llama a reverse_diffusion_single_counts con pinned_mask = True
            final_nodes, final_edges, intermediate_steps = self.reverse_diffusion_single_counts(
                data, pinned_mask, device, save_intermediate=True
            )

            # Construye el grafo final
            graph_data = Data(
                x=F.one_hot(final_nodes, num_classes=self.node_num_classes).float(),
                edge_index=(final_edges[0] > 0).nonzero(as_tuple=False).t().contiguous()
            )
            final_graphs.append(graph_data)
            all_intermediate_steps.append(intermediate_steps)

        torch.save(final_graphs, os.path.join(output_dir, 'final_graphs.pt'))
        torch.save(all_intermediate_steps, os.path.join(output_dir, 'intermediate_steps.pt'))

        print(f"✅ [ALL PINNED] Saved {len(final_graphs)} final graphs to {output_dir}/final_graphs.pt")
        print(f"✅ [ALL PINNED] Saved intermediate steps at '{output_dir}/intermediate_steps.pt'")

    def sample_conditional_partialpinned_and_save(
        self,
        num_nodes: int,
        pinned_info: dict,
        batch_size: int,
        device,
        output_dir='guided_graphs_partialpinned'
    ):
        """
        'num_nodes' total de nodos.
        'pinned_info' es un diccionario {\"MACHINE\": #, \"ASSEMBLY\": #, ...} de cuántos nodos fijos hay de cada tipo.
        'batch_size' indica cuántos grafos generar.
        
        Ejemplo de pinned_info: {\"MACHINE\": 2, \"ASSEMBLY\": 1}
        => 2 nodos fijados a MACHINE, 1 nodo fijado a ASSEMBLY, y los demás libres.
        """
        import os
        import torch
        import torch.nn.functional as F
        from torch_geometric.data import Data

        os.makedirs(output_dir, exist_ok=True)

        final_graphs = []
        all_intermediate_steps = []

        # Mapeo de tipo string -> índice int
        type_map = {"MACHINE\": 0, \"BUFFER\": 1, \"ASSEMBLY\": 2, \"DISASSEMBLY": 3}

        # Construye una lista de nodos fijos
        pinned_list = []
        for type_str, count in pinned_info.items():
            pinned_list.extend([type_map[type_str]] * count)
        
        # Ver cuántos nodos quedaron fijos
        pinned_count = len(pinned_list)
        if pinned_count > num_nodes:
            raise ValueError("Has pineado más nodos de los que existen en total!")

        # El resto de nodos son libres => no se especifica su tipo
        free_count = num_nodes - pinned_count

        # Ej: pinned_list = [0,0,2] (2 machines, 1 assembly), free_count= (lo que falte)
        # Barajamos pinned_list
        pinned_list = torch.tensor(pinned_list, device=device)

        for graph_idx in range(batch_size):
            # Mezclamos pinned_list
            perm_pinned = torch.randperm(pinned_count, device=device)
            pinned_list_shuf = pinned_list[perm_pinned]

            # Crea un vector node_labels de tamaño num_nodes con -1 para nodos libres
            # y type para nodos pineados
            node_labels = torch.full((num_nodes,), -1, device=device, dtype=torch.long)

            # Elige posiciones aleatorias donde colocar estos pinned
            pos_pinned = torch.randperm(num_nodes, device=device)[:pinned_count]
            node_labels[pos_pinned] = pinned_list_shuf

            pinned_mask = (node_labels != -1)  # True para nodos pineados, False para libres

            # Donde no hay pinned => se inicializa con un label aleatorio (0..3) si quieres
            # O con -1 y luego le dejas al modelo. Pero PyTorch no maneja one-hot de -1.
            # Así que conviene asignar un label random (opcional) a los libres.
            for i in range(num_nodes):
                if node_labels[i] == -1:
                    node_labels[i] = torch.randint(0, self.node_num_classes, (1,), device=device)

            # Convierte a one-hot
            pinned_x = F.one_hot(node_labels, num_classes=self.node_num_classes).float()

            # Grafo inicial sin aristas
            edge_index = torch.empty((2, 0), dtype=torch.long, device=device)
            data = Data(x=pinned_x.clone(), edge_index=edge_index)
            data.batch = torch.zeros(num_nodes, dtype=torch.long, device=device)

            final_nodes, final_edges, intermediate_steps = self.reverse_diffusion_single_counts(
                data, pinned_mask, device, save_intermediate=True
            )

            # Construye el grafo final
            graph_data = Data(
                x=F.one_hot(final_nodes, num_classes=self.node_num_classes).float(),
                edge_index=(final_edges[0] > 0).nonzero(as_tuple=False).t().contiguous()
            )
            final_graphs.append(graph_data)
            all_intermediate_steps.append(intermediate_steps)

        torch.save(final_graphs, os.path.join(output_dir, 'final_graphs.pt'))
        torch.save(all_intermediate_steps, os.path.join(output_dir, 'intermediate_steps.pt'))

        print(f"✅ [PARTIALLY PINNED] Saved {len(final_graphs)} final graphs to {output_dir}/final_graphs.pt")
        print(f"✅ [PARTIALLY PINNED] Saved intermediate steps at '{output_dir}/intermediate_steps.pt'")
