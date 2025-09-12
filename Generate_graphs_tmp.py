from integrated_diffusion import IntegratedDiffusionPipeline
pipeline = IntegratedDiffusionPipeline(plant_model, petri_models, device)

g_nodes, g_edges = pipeline.generate_global_graph(n_nodes=15)
petri_nodes, petri_edges = pipeline.generate_petri_subgraph(node_type=0, n_nodes_petri=8)
integ = pipeline.generate_full_integrated_graph(n_nodes_global=15, n_nodes_petri=8)
pipeline.stitch("path_in.pt", save_path="stitched_graph.pt")
1