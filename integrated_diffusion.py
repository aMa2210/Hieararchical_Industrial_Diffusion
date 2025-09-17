import torch
from torch_geometric.data import Data
import os 
import torch.nn.functional as F  
from industrial_diffusion import LightweightIndustrialDiffusion
from diffusion_model import LightweightPetriDiffusion

# Definir constantes para claridad
MACHINE, BUFFER, ASSEMBLY, DISASSEMBLY = 0, 1, 2, 3

class IntegratedDiffusionPipeline:
    def __init__(self, model_high, petri_models, device):
        self.model_high = model_high.to(device)
        self.petri_models = petri_models
        self.device = device

    def generate_global_graph(self, n_nodes):
        edge_list = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if i != j]
        edge_index = torch.tensor(edge_list, dtype=torch.long, device=self.device).t().contiguous()
        x = torch.zeros(n_nodes, self.model_high.node_num_classes, device=self.device)
        data = Data(x=x, edge_index=edge_index)
        data.batch = torch.zeros(n_nodes, dtype=torch.long, device=self.device)

        final_nodes, final_edges, _ = self.model_high.reverse_diffusion_single(data, self.device, False)
        node_types = final_nodes
        return node_types, final_edges

    def generate_petri_subgraph(self, node_type, n_nodes_petri=10):
        petri_model = self.petri_models[node_type].to(self.device)
        edge_list = [(i, j) for i in range(n_nodes_petri) for j in range(n_nodes_petri) if i != j]
        edge_index = torch.tensor(edge_list, dtype=torch.long, device=self.device).t().contiguous()
        x = torch.zeros(n_nodes_petri, petri_model.node_num_classes, device=self.device)
        data = Data(x=x, edge_index=edge_index)
        data.batch = torch.zeros(n_nodes_petri, dtype=torch.long, device=self.device)

        petri_nodes, petri_edges, _ = petri_model.reverse_diffusion_single(data, self.device, False)
        return petri_nodes, petri_edges

    def generate_full_integrated_graph(self, n_nodes_global, n_nodes_petri=10, output_dir='integrated_graph'):
        os.makedirs(output_dir, exist_ok=True)

        global_node_types, global_edges = self.generate_global_graph(n_nodes_global) ## generate again
        # global_node_types, global_edges = self.generate_global_graph_all_pinned(3,4,2,1) ## generate again

        # Convertir edges globales a matriz de adyacencia claramente
        global_adj_matrix = global_edges.squeeze(0).cpu().numpy()
        ########################
        global_graph_to_save = [({"nodes": global_node_types,
                                "edges": global_adj_matrix})]
        adj_list, node_list = [], []
        import numpy as np
        for g in global_graph_to_save:
            #  (n,n) matriz de adyacencia → numpy uint8
            adj_list.append(g["edges"].astype(np.uint8))
            #  vector de etiquetas de nodo → numpy int8
            node_list.append(g["nodes"].cpu().numpy().astype(np.int8))
        LABEL2ID = {"MACHINE": 3,
                    "BUFFER": 1,
                    "ASSEMBLY": 0,
                    "DISASSEMBLY": 2}
        payload = {
            "adjacency_matrices": adj_list,
            "node_types": node_list,
            "label2id": LABEL2ID
        }
        import datetime as dt
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f'industrial_graph_for_stitch/{stamp}.pt'
        torch.save(payload, fname)
        ########################
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
            else:                               # BUFFER  →  subgrafo trivial
                    integrated_graph["petri_subgraphs"][idx] = {
                        "node_types":  [-1],    # un solo place
                        "adjacency_matrix": [[0]],        # sin aristas
                        "in_places":  [0],                # el mismo actúa de entrada
                        "out_places": [0]                 # …y de salida
                    }

        # Guardar el grafo integrado de forma ordenada
        file_path = os.path.join(output_dir, 'integrated_graph.pt')
        torch.save(integrated_graph, file_path)

        print(f"✅ Grafo integrado guardado correctamente en: {file_path}")

        return integrated_graph

    def generate_multiple_integrated_graphs(self, num_graphs, n_nodes_global, n_nodes_petri=10, output_dir='integrated_graphs_batch'):
        os.makedirs(output_dir, exist_ok=True)
        
        for graph_idx in range(num_graphs):
            global_node_types, global_edges = self.generate_global_graph(n_nodes_global)

            # Convertir edges globales a matriz de adyacencia
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
                else:                               # BUFFER  →  subgrafo trivial
                    integrated_graph["petri_subgraphs"][idx] = {
                        "node_types":  [-1],    # un solo place
                        "adjacency_matrix": [[0]],        # sin aristas
                        "in_places":  [0],                # el mismo actúa de entrada
                        "out_places": [0]                 # …y de salida
                    }

            # Guardar claramente cada grafo en un archivo separado
            file_path = os.path.join(output_dir, f'integrated_graph_{graph_idx}.pt')
            torch.save(integrated_graph, file_path)

            print(f"✅ Grafo integrado #{graph_idx} guardado correctamente en: {file_path}")

    def generate_global_graph_all_pinned(self, num_machines, num_buffers, num_assemblies, num_disassemblies): #2 8 2 2
        total_nodes = num_machines + num_buffers + num_assemblies + num_disassemblies
        node_types_list = ([MACHINE] * num_machines +
                        [BUFFER] * num_buffers +
                        [ASSEMBLY] * num_assemblies +
                        [DISASSEMBLY] * num_disassemblies)

        perm = torch.randperm(total_nodes)
        pinned_types = torch.tensor(node_types_list, device=self.device)[perm]
        pinned_x = F.one_hot(pinned_types, num_classes=self.model_high.node_num_classes).float()

        pinned_mask = torch.ones(total_nodes, dtype=torch.bool, device=self.device)

        edge_index = torch.empty((2, 0), dtype=torch.long, device=self.device)
        data = Data(x=pinned_x, edge_index=edge_index)
        data.batch = torch.zeros(total_nodes, dtype=torch.long, device=self.device)

        final_nodes, final_edges, _ = self.model_high.reverse_diffusion_single_counts(data, pinned_mask, self.device, False)
        return final_nodes, final_edges

    def generate_global_graph_partial_pinned(self, num_nodes, pinned_info):
        type_map = {"MACHINE": MACHINE, "BUFFER": BUFFER, "ASSEMBLY": ASSEMBLY, "DISASSEMBLY": DISASSEMBLY}
        pinned_list = []
        for type_str, count in pinned_info.items():
            pinned_list.extend([type_map[type_str]] * count)

        pinned_count = len(pinned_list)
        if pinned_count > num_nodes:
            raise ValueError("Has pineado más nodos de los existentes.")

        free_count = num_nodes - pinned_count
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

        final_nodes, final_edges, _ = self.model_high.reverse_diffusion_single_counts(data, pinned_mask, self.device, False)
        return final_nodes, final_edges
    

    def generate_batch_integrated_graphs(self, 
                                     num_graphs, 
                                     n_nodes_global=None, 
                                     n_nodes_petri=10, 
                                     output_dir='batch_integrated_graphs', 
                                     pinned=None):
        os.makedirs(output_dir, exist_ok=True)

        for graph_idx in range(num_graphs):
            # Manejo claro de casos pinneados
            if pinned is not None:
                if pinned.get("all_pinned"):
                    # Aquí defines automáticamente n_nodes_global
                    num_machines = pinned["num_machines"]
                    num_buffers = pinned["num_buffers"]
                    num_assemblies = pinned["num_assemblies"]
                    num_disassemblies = pinned["num_disassemblies"]
                    n_nodes_total = num_machines + num_buffers + num_assemblies + num_disassemblies

                    final_nodes, final_edges = self.generate_global_graph_all_pinned(
                        num_machines=num_machines,
                        num_buffers=num_buffers,
                        num_assemblies=num_assemblies,
                        num_disassemblies=num_disassemblies
                    )

                elif pinned.get("partial_pinned"):
                    if n_nodes_global is None:
                        raise ValueError("Debes definir claramente 'n_nodes_global' para pinneado parcial.")

                    # Verifica que la suma sea consistente
                    total_pinned = sum(pinned["pinned_info"].values())
                    if total_pinned > n_nodes_global:
                        raise ValueError("Claramente hay más nodos pinneados que el total definido en 'n_nodes_global'.")

                    n_nodes_total = n_nodes_global

                    final_nodes, final_edges = self.generate_global_graph_partial_pinned(
                        num_nodes=n_nodes_global,
                        pinned_info=pinned["pinned_info"]
                    )
                else:
                    raise ValueError("Estructura de pinned incorrecta.")
            else:
                if n_nodes_global is None:
                    raise ValueError("Debes especificar claramente 'n_nodes_global' para generación libre.")
                n_nodes_total = n_nodes_global
                final_nodes, final_edges = self.generate_global_graph(n_nodes_global)

            global_adj_matrix = final_edges.squeeze(0).cpu().numpy()
            global_node_types = final_nodes.cpu().numpy()

            integrated_graph = {
                "global_graph": {
                    "node_types": global_node_types,
                    "adjacency_matrix": global_adj_matrix
                },
                "petri_subgraphs": {}
            }
            BUFFER_PLACE = -1 

            for idx, node_type in enumerate(final_nodes):
                if node_type.item() != BUFFER:
                    petri_nodes, petri_edges = self.generate_petri_subgraph(node_type.item(), n_nodes_petri)
                    petri_adj_tensor = petri_edges.squeeze(0).cpu()
                    petri_node_types = petri_nodes.cpu().numpy()
                    in_places  = torch.where(petri_adj_tensor.sum(0) == 0)[0]
                    out_places = torch.where(petri_adj_tensor.sum(1) == 0)[0]

                    integrated_graph["petri_subgraphs"][idx] = {
                        "node_types": petri_node_types,
                        "adjacency_matrix": petri_adj_tensor.numpy(), 
                        "in_places":  in_places.cpu().numpy().tolist(),
                        "out_places": out_places.cpu().numpy().tolist()
                    }
                else:                               # BUFFER  →  subgrafo trivial
                    integrated_graph["petri_subgraphs"][idx] = {
                        "node_types":  [BUFFER_PLACE],    # un solo place
                        "adjacency_matrix": [[0]],        # sin aristas
                        "in_places":  [0],                # el mismo actúa de entrada
                        "out_places": [0]                 # …y de salida
                    }

            file_path = os.path.join(output_dir, f'integrated_graph_{graph_idx}.pt')
            torch.save(integrated_graph, file_path)

            print(f"✅ Grafo integrado #{graph_idx} guardado correctamente en: {file_path}")



    def stitch(self, path, save_path=None, device='cpu'):
        """
        Une el grafo global con todos los subgrafos Petri en un único `Data`.

        * Cada **BUFFER** se modela ahora **sólo** con un `Place` (`P_buf`).
        * Las conexiones externas siguen la semántica estándar:
            - **Origen BUFFER**  : `P_buf  →  T_in(dest)`  (place→transition)
            - **Destino BUFFER** : `T_out(src) → P_buf`   (transition→place)
        * Para los subgrafos normales se mantiene: todos los `out_places(src)` se
        conectan con todos los `in_transitions(dst)`.
        """

        PLACE, TRANS = 0, 1
        integ = torch.load(path, map_location=device)
        g_adj = torch.tensor(integ["global_graph"]["adjacency_matrix"])



        # contenedores globales
        all_types, edges, next_id = [], [], 0
        out_places, in_trans = {}, {}      # normales
        buffer_place, trans_src = {}, {}   # buffers + lista de transiciones de cada nodo

        # ------------------------------------------------------------------
        # 1) construir nodos y aristas internas
        # ------------------------------------------------------------------
        for n_id, sub in integ["petri_subgraphs"].items():
            # ------------------ BUFFER  (solo un place) -------------------
            if len(sub["node_types"]) == 1 and sub["node_types"][0] == -1:
                p = next_id; all_types.append(PLACE); next_id += 1
                buffer_place[n_id] = p
                # no hay aristas internas ni transiciones
                out_places[n_id] = [p]   # saldrá como place
                in_trans[n_id]  = []     # no se usa, se maneja aparte
                trans_src[n_id] = []
                continue

            # ----------- MACHINE / ASSEMBLY / DISASSEMBLY -----------------
            loc2glob = {}
            for i, t in enumerate(sub["node_types"]):
                all_types.append(PLACE if t in (0, -1) else TRANS)
                loc2glob[i] = next_id; next_id += 1

            A = torch.tensor(sub["adjacency_matrix"])
            r, c = torch.where(A)
            edges += [(loc2glob[u.item()], loc2glob[v.item()]) for u, v in zip(r, c)]

            # interfaces
            places = [i for i, t in enumerate(sub["node_types"]) if t in (0, -1)]
            trans  = [i for i, t in enumerate(sub["node_types"]) if t == 1]

            out_p = [i for i in places if A[i].sum() == 0] or [places[-1]]
            in_t  = [j for j in trans if A[:, j].sum() == 0] or ([trans[0]] if trans else [])

            out_places[n_id] = [loc2glob[i] for i in out_p] #places without output
            in_trans[n_id]  = [loc2glob[j] for j in in_t]   #transitions without input
            trans_src[n_id] = [loc2glob[j] for j in trans]

        # ------------------------------------------------------------------
        # 2) conectar según grafo global
        # ------------------------------------------------------------------
        rows, cols = torch.where(g_adj)
        for src, dst in zip(rows.tolist(), cols.tolist()):
            # --- Ambos nodos normales ----------------------------------------
            # for every place without output and every transition without input, connect them
            for p in out_places[src]:
                for t in in_trans[dst]:
                    edges.append((p, t))

            # --- SRC es BUFFER ------------------------------------------------ place connect to every transition
            # without input of the dst subgraph
            # !!!! here is a problem, after connecting the 'in_trans', maybe this in_trans element should be removed
            # from the in_trans list, otherwise, it will be used to connect again later.
            if src in buffer_place:
                p_buf = buffer_place[src]
                for t in in_trans.get(dst, []):
                    edges.append((p_buf, t))          # place → transition
                continue

            # --- DST es BUFFER ------------------------------------------------
            # the transition node in the src subgraph with least output connect to this buffer
            if dst in buffer_place:
                p_buf = buffer_place[dst]
                # elige la transición saliente de src con menor grado saliente
                if trans_src[src]:
                    degs = [(t, sum(1 for u, _ in edges if u == t)) for t in trans_src[src]]
                    best_t = min(degs, key=lambda x: x[1])[0]
                    edges.append((best_t, p_buf))     # transition → place
                continue



        # ------------------------------------------------------------------
        # 3) construir Data
        # ------------------------------------------------------------------
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        x = torch.zeros(len(all_types), 2, device=device)
        x[torch.arange(len(all_types)), all_types] = 1
        stitched = Data(x=x, edge_index=edge_index)

        # ------------------------------------------------------------------
        # 4) guardar opcionalmente
        # ------------------------------------------------------------------
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(stitched, save_path)
            print(f"✅ Petri net global guardado en {save_path}")

        return stitched










def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_high = LightweightIndustrialDiffusion().to(device).eval()
    petri_models = {
        MACHINE: LightweightPetriDiffusion().to(device).eval(),
        ASSEMBLY: LightweightPetriDiffusion().to(device).eval(),
        DISASSEMBLY: LightweightPetriDiffusion().to(device).eval()
    }

    model_high.load_state_dict(torch.load('industrial_model.pth'))
    petri_models[MACHINE].load_state_dict(torch.load('petri_machine_model.pth'))
    petri_models[ASSEMBLY].load_state_dict(torch.load('petri_assembly_model.pth'))
    petri_models[DISASSEMBLY].load_state_dict(torch.load('petri_disassembly_model.pth'))

    pipeline = IntegratedDiffusionPipeline(model_high, petri_models, device)

    # Ejemplo batch TOTALMENTE pinneado (sin definir n_nodes_global!)
    g = pipeline.stitch(
        path='batch_integrated_free/integrated_graph_0.pt',
        save_path='batch_integrated_free/stitched_graph_0.pt')
    
    pinned_args_all = {
        "all_pinned": True,
        "num_machines": 3,
        "num_buffers": 4,
        "num_assemblies": 2,
        "num_disassemblies": 1
    }

    pipeline.generate_batch_integrated_graphs(
        num_graphs=5,
        n_nodes_petri=8,
        output_dir='batch_integrated_all_pinned',
        pinned=pinned_args_all
    )

    g = pipeline.stitch('batch_integrated_all_pinned/integrated_graph_0.pt')
    print(g) 

    # Ejemplo batch PARCIALMENTE pinneado (requiere n_nodes_global claramente definido!)
    pinned_args_partial = {
        "partial_pinned": True,
        "pinned_info": {"MACHINE": 3, "ASSEMBLY": 1, "BUFFER": 3}
    }

    pipeline.generate_batch_integrated_graphs(
        num_graphs=5,
        n_nodes_global=7,  # claramente mayor o igual a la suma de pinned_info
        n_nodes_petri=8,
        output_dir='batch_integrated_partial_pinned',
        pinned=pinned_args_partial
    )

    # Ejemplo batch SIN pinnear
    pipeline.generate_batch_integrated_graphs(
        num_graphs=5,
        n_nodes_global=12,
        n_nodes_petri=8,
        output_dir='batch_integrated_free'
    )

if __name__ == "__main__":
    main()
