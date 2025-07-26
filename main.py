# run_diffusion_on_petri.py
import torch
from torch_geometric.loader import DataLoader
from diffusion_model import LightweightPetriDiffusion, train_model, compute_edge_weights, compute_marginal_probs
from petri_dataset import PetriNetDataset
import os
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load your dataset (PetriNetDataset)
    root = 'petri_dataset'
    os.makedirs(os.path.join(root, 'raw'), exist_ok=True)
    os.makedirs(os.path.join(root, 'processed'), exist_ok=True)
    from petri_dataset import PetriNetDataset
    dataset = PetriNetDataset(root=root, split='train')
    
    # Compute edge weights and marginal probabilities from the dataset.
    edge_weight = compute_edge_weights(dataset, device)
    node_marginal, edge_marginal = compute_marginal_probs(dataset, device)
    print("Computed edge weights:", edge_weight)
    print("Computed node marginals:", node_marginal)
    print("Computed edge marginals:", edge_marginal)
    
    from torch_geometric.loader import DataLoader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = LightweightPetriDiffusion(T=100, hidden_dim=12, time_embed_dim=16).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    print("Starting training on Petri nets...")
    train_model(model, dataloader, optimizer, device,
                edge_weight=edge_weight, node_marginal=node_marginal, edge_marginal=edge_marginal, epochs=20, T=100)
    
    model.eval()
    model.sample_conditional_and_save(
    n_nodes=10,
    batch_size=10,
    device=device,
    output_dir='generated_graphs'
)
    
    # Loading final graphs
    final_graphs = torch.load('generated_graphs/final_graphs.pt')
    print("Number of final graphs:", len(final_graphs))

    # Load intermediate steps
    intermediate_steps = torch.load('generated_graphs/intermediate_steps.pt')
    print(f"Number of graphs with intermediate steps: {len(intermediate_steps)}")
    print(f"Number of steps for graph 0: {len(intermediate_steps[0])}")




if __name__ == '__main__':
    main()
