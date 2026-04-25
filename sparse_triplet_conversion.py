import pandas as pd 
import os

# Set paths
output_dir = "phase3_aligned_data"

print("--- Step 1: Converting Aligned Matrices to Sparse Triplets & ID Maps ---")

# 1. Load Aligned Data
mrna_df = pd.read_csv(f"{output_dir}/A549_aligned_mRNA.csv", index_col=0)
mirna_df = pd.read_csv(f"{output_dir}/A549_aligned_miRNA.csv", index_col=0)

def matrix_to_triplet(df, id_col_name):
    # Melt the matrix: Rows are Genes/miRNAs, Columns are Cell Barcodes
    triplet = df.reset_index().melt(id_vars=df.index.name, var_name='Cell_ID', value_name='Expression')
    # Filter for expressed items only to keep it sparse
    triplet = triplet[triplet['Expression'] > 0]
    return triplet

# 2. Generate Triplets
mrna_triplet = matrix_to_triplet(mrna_df, "Gene")
mirna_triplet = matrix_to_triplet(mirna_df, "miRNA")

#  rename mRNA columns to remove typos
mrna_triplet.columns = ['Gene', 'Cell_ID', 'Expression']
mirna_triplet.columns = ['miRNA', 'Cell_ID', 'Expression']

# Save Triplets
mrna_triplet.to_csv(f"{output_dir}/A549_mRNA_sparse_triplet.csv", index=False)
mirna_triplet.to_csv(f"{output_dir}/A549_miRNA_sparse_triplet.csv", index=False)

# 3. Save ID mappings for downstream HGT traceability
# These are essential for Step 3 (Joint Embedding) and Step 6 (Regulator Ranking)
print("Saving unique ID mappings...")

# Gene IDs (one per line, matches row indices)
pd.Series(mrna_df.index.unique()).to_csv(f"{output_dir}/gene_ids.txt", index=False, header=False)

# miRNA IDs (one per line, matches row indices)
pd.Series(mirna_df.index.unique()).to_csv(f"{output_dir}/mirna_ids.txt", index=False, header=False)

# Cell IDs (Combined from both to ensure a unified cell node index)
all_cells = pd.Series(pd.concat([mrna_triplet['Cell_ID'], mirna_triplet['Cell_ID']]).unique())
all_cells.to_csv(f"{output_dir}/cell_ids.txt", index=False, header=False)

# Optional Logging
print(f"   - mRNA edges: {len(mrna_triplet)}")
print(f"   - miRNA edges: {len(mirna_triplet)}")
print(f"   - Unique genes: {len(mrna_df.index.unique())}")
print(f"   - Unique miRNAs: {len(mirna_df.index.unique())}")
print(f"   - Unique cells: {len(all_cells)}")

print(f"\n Done! Prepared for Step 2:")
print(f"   - mRNA/miRNA Triplets: A549_mRNA_sparse_triplet.csv, A549_miRNA_sparse_triplet.csv")
print(f"   - ID Maps: gene_ids.txt, mirna_ids.txt, cell_ids.txt")
