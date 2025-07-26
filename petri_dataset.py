# petri_dataset.py (modificado claramente para archivos locales)

import os
import torch
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils
from torch_geometric.data import InMemoryDataset, Data

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
        # Ya no descargarás nada desde una URL.
        # Solo verificamos que el archivo esté presente localmente.
        raw_split_path = os.path.join(self.raw_dir, f'{self.split}.pt')
        if not os.path.exists(raw_split_path):
            raise FileNotFoundError(f"No se encontró el archivo local: {raw_split_path}")

    def process(self):
        raw_path = os.path.join(self.raw_dir, f'{self.split}.pt')
        print("🪵 Leyendo raw de:", raw_path)          # ←  línea de depuración
        raw_dataset = torch.load(raw_path)
        print("🪵 Tipo de la primera muestra:", type(raw_dataset[0]))

        data_list = []
        for sample in raw_dataset:
            adj_matrix = torch.tensor(sample['adjacency_matrix'], dtype=torch.float)
            node_types = sample['node_types']
            nodes_order = sample['nodes_order']

            # Codifica nodos (Place=0, Transition=1)
            node_labels = torch.tensor([0 if node_types[node] == "Place" else 1 for node in nodes_order])

            # One-hot encoding de los tipos de nodo
            x = F.one_hot(node_labels, num_classes=2).float()

            # Conversión a edge_index desde matriz de adyacencia
            edge_index, _ = pyg_utils.dense_to_sparse(adj_matrix)

            # Edge attributes dummy (todos 1)
            edge_attr = torch.ones((edge_index.size(1), 1), dtype=torch.float)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, n_nodes=torch.tensor([len(nodes_order)]))
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# Para probar localmente
if __name__ == '__main__':
    root = 'petri_machine_dataset'  # Cambia esto a assembly o disassembly según corresponda
    os.makedirs(os.path.join(root, 'raw'), exist_ok=True)
    os.makedirs(os.path.join(root, 'processed'), exist_ok=True)

    dataset = PetriNetDataset(root=root, split='train')
    print(f"✅ Dataset cargado correctamente: {len(dataset)} grafos disponibles.")
