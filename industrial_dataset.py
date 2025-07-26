# industrial_dataset.py
import torch
import torch.nn.functional as F
from torch_geometric.data import InMemoryDataset, Data
import torch_geometric.utils as pyg_utils

class IndustrialGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['graphs_data.pt']

    @property
    def processed_file_names(self):
        return ['processed_graphs.pt']

    def download(self):
        pass  # Tus datos ya fueron generados localmente, no es necesario descargar.

    def process(self):
        raw_data = torch.load(self.raw_paths[0])
        adj_matrices = raw_data['adjacency_matrices']
        node_types = raw_data['node_types']

        node_type_to_idx = {'MACHINE': 0, 'BUFFER': 1, 'ASSEMBLY': 2, 'DISASSEMBLY': 3}

        data_list = []
        for adj, types in zip(adj_matrices, node_types):
            node_labels = torch.tensor([node_type_to_idx[t] for t in types])
            x = F.one_hot(node_labels, num_classes=4).float()

            edge_index, _ = pyg_utils.dense_to_sparse(torch.tensor(adj))
            edge_attr = torch.ones((edge_index.size(1), 1), dtype=torch.float)

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, n_nodes=x.size(0))
            data_list.append(data)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
