import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import os
from datetime import datetime
from typing import Dict, Any, List

# --- Data Loading ---
def load_data(pt_path):
    try:
        data = torch.load(pt_path, map_location="cpu", weights_only=False)
        return data
    except FileNotFoundError:
        print(f"Error: {pt_path} not found.")
        return None

# --- Constants ---
# Node type mappings
label2id = {'ASSEMBLY': 0, 'BUFFER': 1, 'DISASSEMBLY': 2, 'MACHINE': 3}
id2label = {v: k for k, v in label2id.items()}

# save path of the scatter plots
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
SAVE_PATH = f'Figure_Simulation/{timestamp}'
os.makedirs(SAVE_PATH, exist_ok=True)

# Simulation parameters
SIMULATION_TIME_STEPS = 1000
DEFAULT_BUFFER_CAPACITY = 20

# Mean processing time for exponential distribution
PROCESSING_TIMES_EXP_MEAN = {
    'MACHINE': 120,      # ~2 minutes
    'ASSEMBLY': 90,      # ~1.5 minutes
    'DISASSEMBLY': 105,  # ~1.75 minutes
    'BUFFER': 0          # Buffers don't "process"
}

# Energy consumption rates
ENERGY_CONSUMPTION_ACTIVE = {
    'MACHINE': 0.50,      # typical machine operation
    'ASSEMBLY': 0.20,     # assembly station
    'DISASSEMBLY': 0.20,  # disassembly station
    'BUFFER': 0       # passive buffer energy use
}

ENERGY_CONSUMPTION_IDLE = {
    'MACHINE': 0.1,      # standby
    'ASSEMBLY': 0.02,     # idle
    'DISASSEMBLY': 0.02,  # idle
    'BUFFER': 0        # constant low power
}

# --- Helper Functions ---
def generate_exp_processing_time(mean_time):
    """Generate processing time from exponential distribution with given mean"""
    if mean_time <= 0:
        return 0
    return int(max(1, random.expovariate(1/mean_time)))

# --- Node Class ---
class Node:
    def __init__(self, node_idx, node_type_id, global_id2label):
        self.id = node_idx
        self.type_id = node_type_id
        self.type_label = global_id2label.get(int(node_type_id), f"UNKNOWN_{node_type_id}")
        
        self.state = 'IDLE'  # IDLE, PROCESSING, BLOCKED_INPUT, BLOCKED_OUTPUT
        self.current_process_time_remaining = 0
        self.processed_item_count = 0
        self.total_energy_consumed = 0.0

        self.input_connections = []
        self.output_connections = []

        if self.type_label == 'BUFFER':
            self.capacity = DEFAULT_BUFFER_CAPACITY
            self.current_items = 0
        else:
            self.capacity = 0
            self.current_items = 0
            self.process_time_mean = PROCESSING_TIMES_EXP_MEAN.get(self.type_label, 1)
    
    def get_new_process_time(self):
        return generate_exp_processing_time(self.process_time_mean)

    def __repr__(self):
        base_repr = f"Node({self.id}, {self.type_label}, St: {self.state}"
        if self.type_label == 'BUFFER':
            base_repr += f", Items: {self.current_items}/{self.capacity}"
        else:
            base_repr += f", ProcTimeRem: {self.current_process_time_remaining}, Processed: {self.processed_item_count}"
        base_repr += f", Energy: {self.total_energy_consumed:.1f})"
        return base_repr

# --- Plant Simulator Class ---
class PlantSimulator:
    def __init__(self, design_idx, adj_matrix, node_type_ids_for_design, global_id2label):
        self.design_idx = design_idx
        self.adj_matrix = adj_matrix
        self.num_nodes = adj_matrix.shape[0]
        
        # Process node type IDs to ensure they are numeric
        actual_node_numeric_ids = []
        for identifier_element in node_type_ids_for_design:
            if isinstance(identifier_element, str):
                numeric_id = label2id.get(identifier_element, -1)
                actual_node_numeric_ids.append(numeric_id)
            elif isinstance(identifier_element, (int, np.integer)):
                actual_node_numeric_ids.append(int(identifier_element))
            else:
                actual_node_numeric_ids.append(-1)
        
        self.nodes = [Node(i, actual_node_numeric_ids[i], global_id2label) for i in range(self.num_nodes)]
        
        self._establish_connections()
        self._identify_entry_exit_nodes()

        self.cumulative_throughput = 0
        self.cumulative_energy_consumed = 0.0

    def _establish_connections(self):
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if self.adj_matrix[i, j] == 1:  # Connection from node i to node j
                    self.nodes[i].output_connections.append(j)
                    self.nodes[j].input_connections.append(i)

    def _identify_entry_exit_nodes(self):
        self.entry_point_node_indices = []
        self.exit_point_node_indices = []

        for i, node in enumerate(self.nodes):
            if node.type_label == 'BUFFER':
                continue  # Buffers are not entry/exit points

            # Check if it's an entry point (only connected to buffers on input side)
            is_entry = all(self.nodes[src_idx].type_label == 'BUFFER'   #tbd, here all the machines/assembly/disassembly are is_entry since there input are all buffers
                          for src_idx in node.input_connections) or not node.input_connections
            if is_entry and node.output_connections:
                self.entry_point_node_indices.append(i)

            # Check if it's an exit point (only connected to buffers on output side)
            is_exit = all(self.nodes[tgt_idx].type_label == 'BUFFER'    #tbd, same problem here, see before
                         for tgt_idx in node.output_connections) or not node.output_connections
            if is_exit and node.input_connections:
                self.exit_point_node_indices.append(i)

    def step_simulation(self):
        # Phase 1: Processing nodes attempt to output finished items
        for i in range(self.num_nodes):
            node = self.nodes[i]
            if node.state == 'PROCESSING' and node.current_process_time_remaining <= 0:
                node.state = 'IDLE'  # Temporarily set to IDLE
                
                if node.type_label == 'DISASSEMBLY':
                    # Disassembly pushes one item to EACH output buffer
                    output_buffers = [self.nodes[idx] for idx in node.output_connections 
                                     if self.nodes[idx].type_label == 'BUFFER']
                    
                    if output_buffers and all(buf.current_items < buf.capacity for buf in output_buffers):
                        for buf in output_buffers:
                            buf.current_items += 1
                        node.processed_item_count += 1
                        if i in self.exit_point_node_indices:
                            self.cumulative_throughput += 1
                    elif output_buffers:  # Has buffers but at least one is full
                        node.state = 'BLOCKED_OUTPUT'
                
                elif node.type_label in ('MACHINE', 'ASSEMBLY'):
                    # Machine/Assembly pushes one item to ONE available output buffer
                    pushed = False
                    for tgt_idx in node.output_connections:
                        tgt_node = self.nodes[tgt_idx]
                        if tgt_node.type_label == 'BUFFER' and tgt_node.current_items < tgt_node.capacity:
                            tgt_node.current_items += 1
                            node.processed_item_count += 1
                            if i in self.exit_point_node_indices:
                                self.cumulative_throughput += 1
                            pushed = True
                            break
                    
                    if not pushed and node.output_connections:
                        node.state = 'BLOCKED_OUTPUT'
            
        # Phase 2: IDLE or BLOCKED nodes attempt to start/resume processing
        for i in range(self.num_nodes):
            node = self.nodes[i]
            if node.type_label == 'BUFFER' or node.state == 'PROCESSING':
                continue

            # Handle entry points with infinite supply
            if i in self.entry_point_node_indices and node.state != 'BLOCKED_OUTPUT':
                node.state = 'PROCESSING'
                node.current_process_time_remaining = node.get_new_process_time()
                continue

            # Regular processing nodes need inputs
            if node.state in ('IDLE', 'BLOCKED_INPUT', 'BLOCKED_OUTPUT'):
                if node.state == 'BLOCKED_OUTPUT':
                    continue  # Can't start if output is blocked
                
                can_process = False
                
                if node.type_label in ('MACHINE', 'DISASSEMBLY'):
                    # Need one item from any input buffer
                    for src_idx in node.input_connections:
                        src_node = self.nodes[src_idx]
                        if src_node.type_label == 'BUFFER' and src_node.current_items > 0:
                            src_node.current_items -= 1
                            can_process = True
                            break
                
                elif node.type_label == 'ASSEMBLY':
                    # Need one item from EACH input buffer
                    input_buffers = [self.nodes[idx] for idx in node.input_connections 
                                    if self.nodes[idx].type_label == 'BUFFER']
                    
                    if input_buffers and all(buf.current_items > 0 for buf in input_buffers):
                        for buf in input_buffers:
                            buf.current_items -= 1
                        can_process = True
                
                if can_process:
                    node.state = 'PROCESSING'
                    node.current_process_time_remaining = node.get_new_process_time()
                else:
                    node.state = 'BLOCKED_INPUT' #tbd is it needed? maybe an idle state is enough

        # Phase 3: Update processing times and energy consumption
        for node in self.nodes:
            if node.state == 'PROCESSING':
                node.current_process_time_remaining -= 1
                energy = ENERGY_CONSUMPTION_ACTIVE.get(node.type_label, 0)
            else:  # IDLE, BLOCKED_INPUT, BLOCKED_OUTPUT
                energy = ENERGY_CONSUMPTION_IDLE.get(node.type_label, 0)
            
            node.total_energy_consumed += energy
            self.cumulative_energy_consumed += energy
            
    def run_full_simulation(self, time_steps):
        # Initialize some buffers to kickstart the system
        for node in self.nodes:
            if node.type_label == "BUFFER":
                # Check if it's an input buffer to an entry point
                is_input_to_entry = any(tgt in self.entry_point_node_indices 
                                      for tgt in node.output_connections)
                if is_input_to_entry:
                    node.current_items = int(node.capacity * 0.5)

        # Use seed for reproducibility
        random.seed(self.design_idx + 42)

        for _ in range(time_steps):
            self.step_simulation()

        return self.cumulative_throughput, self.cumulative_energy_consumed

def run_simulation_for_dataset(dataset_path):
    data = load_data(dataset_path)
    if data is None:
        return []
    
    adjacency_list = data.get("adjacency_matrices")
    node_id_list = data.get("node_types")
    
    if adjacency_list is None or node_id_list is None:
        print(f"Error: Required data not found in {dataset_path}.")
        return []
    
    num_designs = len(adjacency_list)
    print(f"Simulating {num_designs} designs from {dataset_path}...")
    
    simulation_results = []
    
    for i in range(num_designs):
        adj_matrix = adjacency_list[i]
        node_ids = node_id_list[i]
        
        simulator = PlantSimulator(
            design_idx=i, 
            adj_matrix=adj_matrix, 
            node_type_ids_for_design=node_ids, 
            global_id2label=id2label
        )
        
        throughput, total_energy = simulator.run_full_simulation(SIMULATION_TIME_STEPS)
        energy_efficiency = throughput / total_energy if total_energy > 0 else 0
        
        # Count non-buffer nodes
        non_buffer_count = sum(1 for node_type in node_ids if 
                              (isinstance(node_type, str) and node_type != 'BUFFER') or
                              (isinstance(node_type, (int, np.integer)) and int(node_type) != label2id['BUFFER']))
        
        simulation_results.append({
            "design_id": i,
            "dataset": dataset_path,
            "num_nodes": adj_matrix.shape[0],
            "num_non_buffer_nodes": non_buffer_count,
            "num_edges": int(np.sum(adj_matrix)),
            "throughput": throughput,
            "total_energy": total_energy,
            "energy_efficiency": energy_efficiency
        })
        
    return simulation_results


def save_labels_for_dataset(dataset_path: str, results: List[Dict[str, Any]]):
    """
    save a csv file with the metric labels
    """
    if not results:
        print(f"No results to save for {dataset_path}")
        return

    base_name = os.path.splitext(os.path.basename(dataset_path))[0]
    csv_path = os.path.join(SAVE_PATH, f"{base_name}_labels.csv")

    df = pd.DataFrame(results)

    df = df.sort_values(by=["design_id"]).reset_index(drop=True)
    df.to_csv(csv_path, index=False)
    print(f"Saved labels CSV: {csv_path}")


def create_plots(dataset_results, datasets_with_colors):
    # Create normalized plot (by non-buffer nodes)
    plt.figure(figsize=(12, 8))
    
    for dataset_path, color in datasets_with_colors.items():
        results = dataset_results.get(dataset_path, [])
        if not results:
            continue
            
        rgb_color = tuple(c/255 for c in color)
        
        # Normalized values by non-buffer nodes
        throughputs = [r["throughput"] / r["num_non_buffer_nodes"] for r in results]
        energies = [r["total_energy"] / r["num_non_buffer_nodes"] for r in results]
        
        plt.scatter(throughputs, energies, c=[rgb_color], alpha=0.7, 
                    label=f"{dataset_path} ({len(results)} designs)")
    
    plt.title("Normalized Throughput vs Energy Consumption (10,000 Time Step Simulation)")
    plt.xlabel("Throughput (items per node)")
    plt.ylabel("Energy Consumption (units per node)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f"{SAVE_PATH}/normalized_throughput_energy_scatter.png", dpi=500, bbox_inches="tight")
    
    # Create original plot
    plt.figure(figsize=(12, 8))
    
    for dataset_path, color in datasets_with_colors.items():
        results = dataset_results.get(dataset_path, [])
        if not results:
            continue
            
        rgb_color = tuple(c/255 for c in color)
        
        # Raw values
        throughputs = [r["throughput"] for r in results]
        energies = [r["total_energy"] for r in results]
        
        plt.scatter(throughputs, energies, c=[rgb_color], alpha=0.7, 
                    label=f"{dataset_path} ({len(results)} designs)")
    
    plt.title("Throughput vs Energy Consumption by Dataset")
    plt.xlabel("Throughput (items)")
    plt.ylabel("Energy Consumption (units)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(f"{SAVE_PATH}/throughput_energy_scatter.png", dpi=300, bbox_inches="tight")

def main():
    # Define datasets and their corresponding colors
    # datasets = {
    #     "Experiment/graphs_data/training.pt": (230, 109, 80),  # rgb(230,109,80)
    #     "Experiment/graphs_data/E1.pt": (39, 71, 83),          # rgb(39,71,83)
    #     "Experiment/graphs_data/E2.pt": (41, 157, 143),        # rgb(41,157,143)
    #     "Experiment/graphs_data/E3.pt": (231, 198, 107)        # rgb(231,198,107)
    # }
    datasets = {
        "Experiment/graphs_data/training.pt": (230, 109, 80),  # rgb(230,109,80)
        "exp_outputs/300_samples/E1.pt": (39, 71, 83),          # rgb(39,71,83)
        "exp_outputs/300_samples/E2.pt": (41, 157, 143),        # rgb(41,157,143)
        "exp_outputs/300_samples/E3.pt": (231, 198, 107)        # rgb(231,198,107)
    }

    print(f"Running simulations for {len(datasets)} datasets...")
    print(f"Each simulation will run for {SIMULATION_TIME_STEPS} time steps.\n")
    
    all_results = []
    dataset_results = {}
    
    for dataset_path, color in datasets.items():
        print(f"\n--- Processing dataset: {dataset_path} ---")
        results = run_simulation_for_dataset(dataset_path)
        all_results.extend(results)
        dataset_results[dataset_path] = results
        save_labels_for_dataset(dataset_path, results)

        if results:
            avg_throughput = sum(r["throughput"] for r in results) / len(results)
            avg_energy = sum(r["total_energy"] for r in results) / len(results)
            avg_efficiency = sum(r["energy_efficiency"] for r in results) / len(results)
            
            print(f"Dataset {dataset_path} summary:")
            print(f"  Designs: {len(results)}")
            print(f"  Avg Throughput: {avg_throughput:.2f}")
            print(f"  Avg Energy: {avg_energy:.2f}")
            print(f"  Avg Efficiency: {avg_efficiency:.4f}")
    
    # Create both plots
    create_plots(dataset_results, datasets)
    print("\nPlots saved:")
    print(f"- {SAVE_PATH}/normalized_throughput_energy_scatter.png")
    print(f"- {SAVE_PATH}/throughput_energy_scatter.png")
    
    # Show summary statistics
    if all_results:
        print("\n--- Overall Statistics ---")
        print(f"Total designs: {len(all_results)}")
        
        # Find best designs across all datasets
        best_throughput = max(all_results, key=lambda r: r["throughput"])
        best_efficiency = max(all_results, key=lambda r: r["energy_efficiency"])
        lowest_energy = min(all_results, key=lambda r: r["total_energy"])
        
        print(f"Best throughput: {best_throughput['throughput']} (Design {best_throughput['design_id']} from {best_throughput['dataset']})")
        print(f"Best efficiency: {best_efficiency['energy_efficiency']:.4f} (Design {best_efficiency['design_id']} from {best_efficiency['dataset']})")
        print(f"Lowest energy: {lowest_energy['total_energy']:.2f} (Design {lowest_energy['design_id']} from {lowest_energy['dataset']})")
        
        # Create efficiency statistics table
        print("\n--- Efficiency Statistics ---")
        efficiency_stats = []
        for dataset_path in datasets.keys():
            results = dataset_results.get(dataset_path, [])
            if results:
                efficiencies = [r["energy_efficiency"] for r in results]
                efficiency_stats.append({
                    "Scenario": dataset_path,
                    "Mean Efficiency": np.mean(efficiencies),
                    "Median": np.median(efficiencies),
                    "25%": np.percentile(efficiencies, 25),
                    "75%": np.percentile(efficiencies, 75)
                })
        
        # Print the table
        if efficiency_stats:
            print(f"{'Scenario':<15} | {'Mean Efficiency':<15} | {'Median':<10} | {'25%':<10} | {'75%':<10}")
            print("-" * 65)
            for stat in efficiency_stats:
                print(f"{stat['Scenario']:<15} | {stat['Mean Efficiency']:.6f}      | {stat['Median']:.6f} | {stat['25%']:.6f} | {stat['75%']:.6f}")

if __name__ == "__main__":
    main()