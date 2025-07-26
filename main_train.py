# main.py  ── versión con temporizador
from torch_geometric.loader import DataLoader
from diffusion_model import (LightweightPetriDiffusion, train_model,
                             compute_edge_weights, compute_marginal_probs)
from petri_dataset import PetriNetDataset
import argparse, time, os, torch


def main(net_type='machine'):           # 'machine' | 'assembly' | 'disassembly'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_root = f'petri_{net_type}_dataset'
    os.makedirs(os.path.join(dataset_root, 'raw'),       exist_ok=True)
    os.makedirs(os.path.join(dataset_root, 'processed'), exist_ok=True)

    dataset = PetriNetDataset(root=dataset_root, split='train')

    edge_weight             = compute_edge_weights(dataset, device)
    node_marg, edge_marg    = compute_marginal_probs(dataset, device)

    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    model      = LightweightPetriDiffusion(T=100, hidden_dim=12,
                                           time_embed_dim=16).to(device)
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



# ── main.py (solo la parte final) ─────────────────────────────
if __name__ == '__main__':
    import argparse, time
    parser = argparse.ArgumentParser()
    parser.add_argument('--net_type',
                        type=str,
                        default='machine',
                        choices=['machine', 'assembly', 'disassembly', 'all'],
                        help="Tipo de Petri-net a entrenar")
    args = parser.parse_args()

    # ⚠️  NO llames a main() antes de leer args
    if args.net_type == 'all':
        for t in ['machine', 'assembly', 'disassembly']:
            main(net_type=t)
    else:
        main(net_type=args.net_type)
