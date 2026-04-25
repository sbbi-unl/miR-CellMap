import torch
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import pandas as pd
import os

# Define the central directory
output_dir = "phase3_aligned_data"

print(f"--- Step 2: Constructing Tripartite Graph from {output_dir} ---")

# 1. Load ID Mappings (Established in Step 1)
genes = pd.read_csv(f"{output_dir}/gene_ids.txt", header=None)[0].astype(str).tolist()
mirnas = pd.read_csv(f"{output_dir}/mirna_ids.txt", header=None)[0].astype(str).tolist()
cells = pd.read_csv(f"{output_dir}/cell_ids.txt", header=None)[0].astype(str).tolist()

# Create lookup dictionaries for integer indexing
gene_map = {name: i for i, name in enumerate(genes)}
mirna_map = {name: i for i, name in enumerate(mirnas)}
cell_map = {name: i for i, name in enumerate(cells)}

data = HeteroData()

# 2. Initialize Node Counts
data['gene'].num_nodes = len(genes)
data['mirna'].num_nodes = len(mirnas)
data['cell'].num_nodes = len(cells)

# 3. Map Cell-Gene Edges (mRNA)
print("    - Mapping mRNA edges...")
mrna_triplet = pd.read_csv(f"{output_dir}/A549_mRNA_sparse_triplet.csv")
mrna_triplet['Gene'] = mrna_triplet['Gene'].astype(str)
mrna_triplet['Cell_ID'] = mrna_triplet['Cell_ID'].astype(str)

mrna_triplet = mrna_triplet[
    mrna_triplet['Gene'].isin(gene_map) &
    mrna_triplet['Cell_ID'].isin(cell_map)
].copy()

mrna_edge_index = torch.tensor([
    [gene_map[g] for g in mrna_triplet['Gene']],
    [cell_map[c] for c in mrna_triplet['Cell_ID']]
], dtype=torch.long)
data['gene', 'expressed_in', 'cell'].edge_index = mrna_edge_index
print(f"      mRNA edges kept: {mrna_edge_index.shape[1]}")

# 4. Map Cell-miRNA Edges
print("    - Mapping miRNA edges...")
mirna_triplet = pd.read_csv(f"{output_dir}/A549_miRNA_sparse_triplet.csv")
mirna_triplet['miRNA'] = mirna_triplet['miRNA'].astype(str)
mirna_triplet['Cell_ID'] = mirna_triplet['Cell_ID'].astype(str)

mirna_triplet = mirna_triplet[
    mirna_triplet['miRNA'].isin(mirna_map) &
    mirna_triplet['Cell_ID'].isin(cell_map)
].copy()

mirna_edge_index = torch.tensor([
    [mirna_map[m] for m in mirna_triplet['miRNA']],
    [cell_map[c] for c in mirna_triplet['Cell_ID']]
], dtype=torch.long)
data['mirna', 'expressed_in', 'cell'].edge_index = mirna_edge_index
print(f"      miRNA edges kept: {mirna_edge_index.shape[1]}")

# 5. Map miRNA-Gene Prior Edges
print("    - Mapping Prior evidence edges...")
prior_df = pd.read_csv(f"{output_dir}/mirna_gene_prior.csv")
prior_df['miRNA'] = prior_df['miRNA'].astype(str)
prior_df['Target Gene'] = prior_df['Target Gene'].astype(str)

prior_total = len(prior_df)

# Keep only prior edges that exist in the graph ID space
prior_df = prior_df[
    prior_df['miRNA'].isin(mirna_map) &
    prior_df['Target Gene'].isin(gene_map)
].drop_duplicates().copy()

prior_kept = len(prior_df)
prior_dropped = prior_total - prior_kept

print(f"      Prior edges total:   {prior_total}")
print(f"      Prior edges kept:     {prior_kept}")
print(f"      Prior edges dropped: {prior_dropped}")

if prior_kept == 0:
    raise ValueError("No prior edges remained after matching to mirna_ids.txt and gene_ids.txt.")

prior_edge_index = torch.tensor([
    [mirna_map[m] for m in prior_df['miRNA']],
    [gene_map[g] for g in prior_df['Target Gene']]
], dtype=torch.long)
data['mirna', 'targets', 'gene'].edge_index = prior_edge_index

# 6. Apply Bidirectional Logic 
data = T.ToUndirected()(data)

# 7. Save the Graph Object for HGT Training
output_file = f"{output_dir}/tripartite_graph_object.pt"
torch.save(data, output_file)

print(f"\nStep 2 Complete!")
print(f"    - Saved: {output_file}")
print(f"    - Nodes: Genes({len(genes)}), miRNAs({len(mirnas)}), Cells({len(cells)})")
print(f"    - Edge types constructed: {data.edge_types}")
