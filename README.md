# Hierarchical Graph Diffusion for Industrial Systems

> **Master Thesis – Thomas von Plessing**  
> Two‑level discrete diffusion to generate (1) plant‑level industrial graphs and (2) Petri‑net subgraphs per complex node, then stitch everything into one executable Petri net.

---

## 1. Quick Overview
- **Level 1 – Plant graph:** Directed graph with node types `MACHINE, BUFFER, ASSEMBLY, DISASSEMBLY` under hard structural constraints.  
- **Level 2 – Petri subgraphs:** For each complex node, a small Petri net (Places/Transitions, no self‑loops, no same‑type edges).  
- **Pipeline:** Sample plant graph → sample each node’s Petri net → **stitch** into one global Petri net.

---

## 2. Repository Structure
```
├─ diffusion_model.py        # Diffusion for Petri nets (nodes/edges), masks & losses
├─ industrial_diffusion.py   # Diffusion for plant graphs, constraints & projector
├─ integrated_diffusion.py   # Full pipeline: both levels + stitch()
├─ industrial_dataset.py     # PyG dataset for plant graphs (graphs_data.pt)
├─ petri_dataset.py          # PyG dataset for Petri nets
├─ train_industrial.py       # Train script for plant-level model
├─ main_train.py             # CLI training for Petri-net models (per node type)
├─ main.py                   # Simple train/sample runner for Petri nets
├─ experiments.py            # E1/E2/E3 runs, eval (valid/unique/novel, FID/MMD) & save
├─ ablation_study.py         # Ablations over variants → metrics + CSV
├─ industrial_visualize.py   # Render plant graphs to PNG
├─ plant_level.py            # Quick inspector for graphs_data_int.pt
└─ stich_visualize*          # (Optional) stitched Petri-net visualizer
```

---

## 3. Requirements & Environment (DiGress base)
Tested with **Python 3.9**, **PyTorch 2.0.1 (CUDA 11.8)**, **Torch Geometric 2.3.1**.  
Environment follows **DiGress** (Vignac et al. 2022). You can skip RDKit/graph-tool/ORCA if you don’t need their metrics.

```bash
# Miniconda/Anaconda

conda create -c conda-forge -n digress rdkit=2023.03.2 python=3.9
conda activate digress

python -c "from rdkit import Chem"

conda install -c conda-forge graph-tool=2.45
python -c "import graph_tool as gt"

conda install -c "nvidia/label/cuda-11.8.0" cuda
pip install torch==2.0.1 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
pip install -e .

# ORCA (only if you need DiGress graphlet metrics)
cd src/analysis/orca
g++ -O2 -std=c++11 -o orca orca.cpp
```

> **Optional:** If you only run the industrial/Petri diffusion (no graphlet/FID code), you can skip RDKit, graph-tool and ORCA.

---

## 4. Data Format
- **Plant graphs (`graphs_data.pt`)**: `{ adjacency_matrices, node_types, label2id }`.  
- **Petri nets (`petri_*_dataset/...`)**: `adjacency_matrix`, `node_types` (“Place”/“Transition”), `nodes_order`, etc.

---

## 5. Training
### Petri‑net models
```bash
python main_train.py --net_type machine
python main_train.py --net_type assembly
python main_train.py --net_type disassembly
# or
python main_train.py --net_type all
```
Outputs: `petri_{type}_model.pth`

### Plant‑level model
```bash
python train_industrial.py --epochs 30 --batch 4 --lr 1e-3
```
Output: `industrial_model.pth`

---

## 6. Generation / Full Pipeline
```python
from integrated_diffusion import IntegratedDiffusionPipeline
pipeline = IntegratedDiffusionPipeline(plant_model, petri_models, device)

g_nodes, g_edges = pipeline.generate_global_graph(n_nodes=15)
petri_nodes, petri_edges = pipeline.generate_petri_subgraph(node_type=0, n_nodes_petri=8)
integ = pipeline.generate_full_integrated_graph(n_nodes_global=15, n_nodes_petri=8)
pipeline.stitch("path_in.pt", save_path="stitched_graph.pt")
```
Guided options:
```python
pipeline.generate_global_graph_all_pinned(...)
pipeline.generate_global_graph_partial_pinned(...)
```

---

## 7. Experiments & Metrics
```bash
python experiments.py                # E1 (free)
python experiments.py --mode allpin  # E2, etc.
```
Metrics: validity, uniqueness, novelty (WL hash). Optional: FID & MMD with a light GIN encoder.  
Outputs under `exp_outputs/`.

---

## 8. Ablation
```bash
python ablation_study.py
```
Runs predefined variants and aggregates summaries into `ablation_summary.csv`.

---

## 9. Visualization
```bash
python industrial_visualize.py
```
Generates PNGs for plant graphs. Use `stich_visualize` for stitched Petri nets.

---


## 10. License
MIT — see [LICENSE](LICENSE).


