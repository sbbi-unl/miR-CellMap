import pandas as pd
import os

# --- Step 0: Set up Directory ---
output_dir = "phase3_aligned_data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Step 1: Load Barcodes & Filter for A549 ---
def get_a549_barcodes(file_path):
    # Using \s+ to handle variable whitespace in GEO txt files
    df = pd.read_csv(file_path, sep='\s+', header=None)
    return df[df[1] == 'A549-MS'][0].tolist()

print("--- Step 1: Barcode Processing ---")
barcodes_f = "GSE226714_cells_chip1_barcodes.txt"
a549_barcodes = get_a549_barcodes(barcodes_f)

# Step 2: Load Expression Matrices ---
print("--- Step 2: Loading Matrices ---")
mrna = pd.read_csv("GSM7082513_cells_chip1_mRNA.txt", sep='\s+', index_col=0)
mirna = pd.read_csv("GSM7082512_cells_chip1_miRNA.txt", sep='\s+', index_col=0)

# Step 3: Three-Way Alignment ---
print("--- Step 3: Aligning Cells ---")
common_cells = list(set(mrna.columns) & set(mirna.columns) & set(a549_barcodes))

# Warning for low aligned cells
if len(common_cells) < 100:
    print(f"⚠Warning: Low number of aligned cells ({len(common_cells)}). Check barcode overlap.")

mrna_final = mrna[common_cells]
mirna_final = mirna[common_cells]

# Explicitly name indices 
mrna_final.index.name = "Gene"
mirna_final.index.name = "miRNA"

# Step 4: Biological Prior Filtering ---
print("--- Step 4: Prior Matrix Generation ---")
mirtar = pd.read_csv("miRTarBase_MTI.csv", low_memory=False)

# Filter for Human and Gold Standard validated assays
mirtar_hsa = mirtar[mirtar['Species (Target Gene)'] == 'hsa']
gold_assays = ['Luciferase reporter assay', 'Western blot', 'qRT-PCR']
mirtar_hsa = mirtar_hsa[mirtar_hsa['Experiments'].str.contains('|'.join(gold_assays), na=False)]

expressed_genes = set(mrna_final.index)
expressed_mirnas = set(mirna_final.index)

# Intersect with data
prior_edges = mirtar_hsa[
    (mirtar_hsa['Target Gene'].isin(expressed_genes)) & 
    (mirtar_hsa['miRNA'].isin(expressed_mirnas))
]

print(f"   - Expressed genes: {len(expressed_genes)}")
print(f"   - Expressed miRNAs: {len(expressed_mirnas)}")
print(f"   - Filtered prior edges: {len(prior_edges)}")

# Step 5: Save Results ---
prior_edges[['miRNA', 'Target Gene']].drop_duplicates().to_csv(f"{output_dir}/mirna_gene_prior.csv", index=False)
mrna_final.to_csv(f"{output_dir}/A549_aligned_mRNA.csv")
mirna_final.to_csv(f"{output_dir}/A549_aligned_miRNA.csv")

print(f"\n Done! Aligned {len(common_cells)} A549 cells and saved outputs to: {output_dir}")
