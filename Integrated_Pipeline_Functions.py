import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from Industrial_Pipeline_Functions import LightweightIndustrialDiffusion
from Petri_Pipeline_Functions import LightweightPetriDiffusion

# Define constants for clarity
MACHINE, BUFFER, ASSEMBLY, DISASSEMBLY = 0, 1, 2, 3


class IntegratedDiffusionPipeline:
    def __init__(self, model_high, petri_models, device):
        self.model_high = model_high.to(device)
        self.petri_models = petri_models
        self.device = device

    # --------------------------- Global (industrial) ---------------------------
    def generate_global_graph(self, n_nodes):
        edge_list = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if i != j]
        edge_index = torch.tensor(edge_list, dtype=torch.long, device=self.device).t().contiguous()
        x = torch.zeros(n_nodes, self.model_high.node_num_classes, device=self.device)
        data = Data(x=x, edge_index=edge_index)
        data.batch = torch.zeros(n_nodes, dtype=torch.long, device=self.device)

        final_nodes, final_edges, _ = self.model_high.reverse_diffusion_single(
            data, self.device, False
        )
        node_types = final_nodes
        return node_types, final_edges

    # --------------------------- Petri subgraph -------------------------------
    def generate_petri_subgraph(self, node_type, n_nodes_petri=10):
        petri_model = self.petri_models[node_type].to(self.device)
        edge_list = [(i, j) for i in range(n_nodes_petri) for j in range(n_nodes_petri) if i != j]
        edge_index = torch.tensor(edge_list, dtype=torch.long, device=self.device).t().contiguous()
        x = torch.zeros(n_nodes_petri, petri_model.node_num_classes, device=self.device)
        data = Data(x=x, edge_index=edge_index)
        data.batch = torch.zeros(n_nodes_petri, dtype=torch.long, device=self.device)

        petri_nodes, petri_edges, _ = petri_model.reverse_diffusion_single(
            data, self.device, False
        )
        return petri_nodes, petri_edges

    # --------------------- Build integrated dict (global+petri) ----------------
    def generate_full_integrated_graph(self, n_nodes_global, n_nodes_petri=10, output_dir='integrated_graph'):
        os.makedirs(output_dir, exist_ok=True)

        # global_node_types, global_edges = self.generate_global_graph(n_nodes_global)
        global_node_types, global_edges = self.generate_global_graph_all_pinned(3, 4, 1, 2)
        global_adj_matrix = global_edges.squeeze(0).cpu().numpy()

        # also persist an industrial-only payload for external stitching/debug
        os.makedirs('industrial_graph_for_stitch', exist_ok=True)
        global_graph_to_save = [({"nodes": global_node_types, "edges": global_adj_matrix})]
        import numpy as np, datetime as dt
        adj_list, node_list = [], []
        for g in global_graph_to_save:
            adj_list.append(g["edges"].astype(np.uint8))                       # (n,n) -> uint8
            node_list.append(g["nodes"].cpu().numpy().astype(np.int8))         # (n,)  -> int8
        LABEL2ID = {"MACHINE": 3, "BUFFER": 1, "ASSEMBLY": 0, "DISASSEMBLY": 2}
        payload = {"adjacency_matrices": adj_list, "node_types": node_list, "label2id": LABEL2ID}
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f'industrial_graph_for_stitch/{stamp}.pt'
        torch.save(payload, fname)

        integrated_graph = {
            "global_graph": {
                "node_types": global_node_types.cpu().numpy(),
                "adjacency_matrix": global_adj_matrix
            },
            "petri_subgraphs": {}
        }

        for idx, node_type in enumerate(global_node_types):
            if node_type.item() != BUFFER:
                petri_nodes, petri_edges = self.generate_petri_subgraph(node_type.item(), n_nodes_petri)
                petri_adj_matrix = petri_edges.squeeze(0).cpu().numpy()
                petri_node_types = petri_nodes.cpu().numpy()
                integrated_graph["petri_subgraphs"][idx] = {
                    "node_types": petri_node_types,
                    "adjacency_matrix": petri_adj_matrix
                }
            else:  # BUFFER → trivial subgraph
                integrated_graph["petri_subgraphs"][idx] = {
                    "node_types":  [-1],           # a single place
                    "adjacency_matrix": [[0]],     # no edges
                    "in_places":  [0],
                    "out_places": [0]
                }

        file_path = os.path.join(output_dir, 'integrated_graph.pt')
        torch.save(integrated_graph, file_path)
        print(f"✅ Integrated graph saved correctly in: {file_path}")
        return integrated_graph

    # ------------------------ All pinned (inventory) ---------------------------
    def generate_global_graph_all_pinned(self, num_machines, num_buffers, num_assemblies, num_disassemblies):
        total_nodes = num_machines + num_buffers + num_assemblies + num_disassemblies
        node_types_list = (
            [MACHINE] * num_machines +
            [BUFFER] * num_buffers +
            [ASSEMBLY] * num_assemblies +
            [DISASSEMBLY] * num_disassemblies
        )

        perm = torch.randperm(total_nodes)
        pinned_types = torch.tensor(node_types_list, device=self.device)[perm]
        pinned_x = F.one_hot(pinned_types, num_classes=self.model_high.node_num_classes).float()
        pinned_mask = torch.ones(total_nodes, dtype=torch.bool, device=self.device)

        edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        data = Data(x=pinned_x, edge_index=edge_index)
        data.batch = torch.zeros(total_nodes, dtype=torch.long, device=self.device)

        final_nodes, final_edges, _ = self.model_high.reverse_diffusion_single_counts(
            data, pinned_mask, self.device, False
        )
        return final_nodes, final_edges

    # --------------------------- Partial pinned -------------------------------
    def generate_global_graph_partial_pinned(self, num_nodes, pinned_info):
        type_map = {"MACHINE": MACHINE, "BUFFER": BUFFER, "ASSEMBLY": ASSEMBLY, "DISASSEMBLY": DISASSEMBLY}
        pinned_list = []
        for type_str, count in pinned_info.items():
            pinned_list.extend([type_map[type_str]] * count)

        pinned_count = len(pinned_list)
        if pinned_count > num_nodes:
            raise ValueError("You pinned more nodes than the existing ones.")

        pinned_list = torch.tensor(pinned_list, device=self.device)
        perm_pinned = torch.randperm(pinned_count, device=self.device)
        pinned_list_shuf = pinned_list[perm_pinned]

        node_labels = torch.full((num_nodes,), -1, device=self.device, dtype=torch.long)
        pos_pinned = torch.randperm(num_nodes, device=self.device)[:pinned_count]
        node_labels[pos_pinned] = pinned_list_shuf
        pinned_mask = (node_labels != -1)

        for i in range(num_nodes):
            if node_labels[i] == -1:
                node_labels[i] = torch.randint(0, self.model_high.node_num_classes, (1,), device=self.device)

        pinned_x = F.one_hot(node_labels, num_classes=self.model_high.node_num_classes).float()

        edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        data = Data(x=pinned_x, edge_index=edge_index)
        data.batch = torch.zeros(num_nodes, dtype=torch.long, device=self.device)

        final_nodes, final_edges, _ = self.model_high.reverse_diffusion_single_counts(
            data, pinned_mask, self.device, False
        )
        return final_nodes, final_edges

    # ------------------------------ Batch builder -----------------------------
    def generate_multiple_integrated_graphs(self, num_graphs, n_nodes_global, n_nodes_petri=10, output_dir='integrated_graphs_batch'):
        os.makedirs(output_dir, exist_ok=True)
        for graph_idx in range(num_graphs):
            global_node_types, global_edges = self.generate_global_graph(n_nodes_global)
            global_adj_matrix = global_edges.squeeze(0).cpu().numpy()
            integrated_graph = {
                "global_graph": {
                    "node_types": global_node_types.cpu().numpy(),
                    "adjacency_matrix": global_adj_matrix
                },
                "petri_subgraphs": {}
            }

            for idx, node_type in enumerate(global_node_types):
                if node_type.item() != BUFFER:
                    petri_nodes, petri_edges = self.generate_petri_subgraph(node_type.item(), n_nodes_petri)
                    petri_adj_matrix = petri_edges.squeeze(0).cpu().numpy()
                    petri_node_types = petri_nodes.cpu().numpy()
                    integrated_graph["petri_subgraphs"][idx] = {
                        "node_types": petri_node_types,
                        "adjacency_matrix": petri_adj_matrix
                    }
                else:
                    integrated_graph["petri_subgraphs"][idx] = {
                        "node_types":  [-1],
                        "adjacency_matrix": [[0]],
                        "in_places":  [0],
                        "out_places": [0]
                    }

            file_path = os.path.join(output_dir, f'integrated_graph_{graph_idx}.pt')
            torch.save(integrated_graph, file_path)
            print(f"✅ Integrated graph #{graph_idx} saved correctly in: {file_path}")

    # ------------------------------- Stitch -----------------------------------
    def stitch(self, path, save_path=None, device='cpu'):
        """
        Merge the global plant graph with all Petri subgraphs into a single torch_geometric.data.Data.

        * Each BUFFER is modeled only with a single Place (P_buf).
        * External connections:
            - BUFFER as source:  P_buf -> in-transitions(dst)
            - BUFFER as target:  out-transition(src) -> P_buf
        * Normal subgraphs: all out_places(src) connect to all in_transitions(dst).
        """
        PLACE, TRANS = 0, 1
        integ = torch.load(path, map_location=device)
        g_adj = torch.tensor(integ["global_graph"]["adjacency_matrix"])

        all_types, edges, next_id = [], [], 0
        out_places, in_trans = {}, {}
        buffer_place, trans_src = {}, {}

        # 1) nodes + internal edges
        for n_id, sub in integ["petri_subgraphs"].items():
            if len(sub["node_types"]) == 1 and sub["node_types"][0] == -1:
                p = next_id
                all_types.append(PLACE)
                next_id += 1
                buffer_place[n_id] = p
                out_places[n_id] = [p]
                in_trans[n_id] = []
                trans_src[n_id] = []
                continue

            loc2glob = {}
            for i, t in enumerate(sub["node_types"]):
                all_types.append(PLACE if t in (0, -1) else TRANS)
                loc2glob[i] = next_id
                next_id += 1

            A = torch.tensor(sub["adjacency_matrix"])
            r, c = torch.where(A)
            edges += [(loc2glob[u.item()], loc2glob[v.item()]) for u, v in zip(r, c)]

            places = [i for i, t in enumerate(sub["node_types"]) if t in (0, -1)]
            trans  = [i for i, t in enumerate(sub["node_types"]) if t == 1]

            out_p = [i for i in places if A[i].sum() == 0] or [places[-1]]
            in_t  = [j for j in trans if A[:, j].sum() == 0] or ([trans[0]] if trans else [])

            out_places[n_id] = [loc2glob[i] for i in out_p]
            in_trans[n_id]  = [loc2glob[j] for j in in_t]
            trans_src[n_id] = [loc2glob[j] for j in trans]

        # 2) wire according to global graph
        rows, cols = torch.where(g_adj)
        for src, dst in zip(rows.tolist(), cols.tolist()):
            # normal → normal
            for p in out_places[src]:
                for t in in_trans[dst]:
                    edges.append((p, t))
            # BUFFER as source
            if src in buffer_place:
                p_buf = buffer_place[src]
                for t in in_trans.get(dst, []):
                    edges.append((p_buf, t))
                continue
            # BUFFER as target
            if dst in buffer_place:
                p_buf = buffer_place[dst]
                if trans_src[src]:
                    degs = [(t, sum(1 for u, _ in edges if u == t)) for t in trans_src[src]]
                    best_t = min(degs, key=lambda x: x[1])[0]
                    edges.append((best_t, p_buf))
                continue

        # 3) Data
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        x = torch.zeros(len(all_types), 2, device=device)
        x[torch.arange(len(all_types)), all_types] = 1
        stitched = Data(x=x, edge_index=edge_index)

        # 4) save optional
        if save_path:
            save_dir = os.path.dirname(save_path)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            torch.save(stitched, save_path)
            print(f"✅ Global Petri net saved in {save_path}")

        return stitched