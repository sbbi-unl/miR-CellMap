import pandas as pd
import os
import re
from io import StringIO


ALIGNED_DIR = "phase3_aligned_data"
RAW_MIRTARBASE = "miRTarBase_MTI.csv"
RAW_ENCORI = "ENCORI_hg38_miRNA_mRNA_all.txt"
OUTPUT_FILE = f"{ALIGNED_DIR}/mirna_gene_prior.csv"
METADATA_FILE = f"{ALIGNED_DIR}/mirna_gene_prior_metadata.csv"

def normalize_gene(x):
    return str(x).strip().upper()

def normalize_mirna(x):
    x = str(x).strip()
    x = x.replace("_", "-")
    x = re.sub(r"\s+", "", x)
    x = x.replace("MicroRNA", "miR")
    x = x.replace("MICRORNA", "MIR")
    x = x.replace("Mirna", "Mir")
    x = x.replace("MIRNA", "MIR")
    x = re.sub(r"-+", "-", x)
    return x.upper()

def load_encori_table(filepath):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"ENCORI file not found: {filepath}")

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    header_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if (
            ("mirnaid" in lower or "mirnaname" in lower or "\tmirna\t" in f"\t{lower}\t")
            and ("genename" in lower or "geneid" in lower or "\tgene\t" in f"\t{lower}\t")
        ):
            header_idx = i
            break

    if header_idx is None:
        raise ValueError("Could not detect ENCORI header row in file.")

    table_text = "".join(lines[header_idx:])
    df = pd.read_csv(StringIO(table_text), sep="\t", dtype=str, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]
    return df

print("--- Step 1: Loading Aligned A549 Expression Data ---")

mrna_df = pd.read_csv(f"{ALIGNED_DIR}/A549_aligned_mRNA.csv", index_col=0)
mirna_df = pd.read_csv(f"{ALIGNED_DIR}/A549_aligned_miRNA.csv", index_col=0)

gene_lookup = {normalize_gene(g): g for g in mrna_df.index}
mirna_lookup = {normalize_mirna(m): m for m in mirna_df.index}

expressed_genes_norm = set(gene_lookup.keys())
expressed_mirnas_norm = set(mirna_lookup.keys())

print(f"    - Genes expressed in A549: {len(expressed_genes_norm)}")
print(f"    - miRNAs expressed in A549: {len(expressed_mirnas_norm)}")
print(f"    - Example genes: {list(mrna_df.index[:10])}")
print(f"    - Example miRNAs: {list(mirna_df.index[:10])}")

print("--- Step 2A: Processing miRTarBase 10.0 ---")
cols = ['miRNA', 'Target Gene', 'Species (Target Gene)', 'Experiments', 'Support Type']
df_mirtar = pd.read_csv(RAW_MIRTARBASE, usecols=cols, low_memory=False)

df_mirtar_filtered = df_mirtar[
    (df_mirtar['Species (Target Gene)'] == 'hsa') &
    (df_mirtar['Support Type'] == 'Functional MTI')
].copy()

gold_assays = ['Luciferase reporter assay', 'Western blot', 'qRT-PCR']
df_mirtar_filtered = df_mirtar_filtered[
    df_mirtar_filtered['Experiments'].str.contains('|'.join(gold_assays), na=False)
].copy()

df_mirtar_filtered['miRNA_norm'] = df_mirtar_filtered['miRNA'].apply(normalize_mirna)
df_mirtar_filtered['Target_Gene_norm'] = df_mirtar_filtered['Target Gene'].apply(normalize_gene)
df_mirtar_filtered['Source'] = "miRTarBase"
df_mirtar_filtered['Is_Database_Prior'] = 1
df_mirtar_filtered['Is_A549_Context'] = 0

print(f"    - miRTarBase filtered interactions: {len(df_mirtar_filtered)}")

print("--- Step 2B: Processing ENCORI ---")
df_encori = load_encori_table(RAW_ENCORI)

mir_col = "miRNAname" if "miRNAname" in df_encori.columns else ("miRNA" if "miRNA" in df_encori.columns else "miRNAid")
gene_col = "geneName" if "geneName" in df_encori.columns else ("GeneName" if "GeneName" in df_encori.columns else "geneSymbol")
clip_col = "clipExpNum" if "clipExpNum" in df_encori.columns else ("AgoExpNum" if "AgoExpNum" in df_encori.columns else None)

possible_cell_cols = [
    "cellline/tissue", "cell line/tissue", "cell_line/tissue",
    "cellline", "cell line", "Cellline/Tissue", "CellLine", "Cell line"
]
cell_col = None
for c in possible_cell_cols:
    if c in df_encori.columns:
        cell_col = c
        break

if mir_col not in df_encori.columns or gene_col not in df_encori.columns:
    raise ValueError(
        f"Could not find miRNA/gene columns in ENCORI file. Columns found: {list(df_encori.columns)}"
    )

df_encori['miRNA_norm'] = df_encori[mir_col].apply(normalize_mirna)
df_encori['Target_Gene_norm'] = df_encori[gene_col].apply(normalize_gene)

if cell_col is not None:
    df_encori['Is_A549_Context'] = df_encori[cell_col].astype(str).str.contains("A549", case=False, na=False).astype(int)
else:
    df_encori['Is_A549_Context'] = 0

if clip_col is not None:
    df_encori[clip_col] = pd.to_numeric(df_encori[clip_col], errors="coerce").fillna(0)
    df_encori = df_encori[
        (df_encori[clip_col] >= 1) | (df_encori['Is_A549_Context'] == 1)
    ].copy()
else:
    df_encori = df_encori[df_encori['Is_A549_Context'] == 1].copy()

df_encori = df_encori[['miRNA_norm', 'Target_Gene_norm', 'Is_A549_Context']].drop_duplicates().copy()
df_encori['Source'] = "ENCORI"
df_encori['Is_Database_Prior'] = 1

print(f"    - ENCORI experimentally supported or A549-forced interactions: {len(df_encori)}")

print("--- Step 3: Mapping Priors to A549 Expression ---")

df_mirtar_final = df_mirtar_filtered[
    (df_mirtar_filtered['miRNA_norm'].isin(expressed_mirnas_norm)) &
    (df_mirtar_filtered['Target_Gene_norm'].isin(expressed_genes_norm))
][['miRNA_norm', 'Target_Gene_norm', 'Source', 'Is_Database_Prior', 'Is_A549_Context']].drop_duplicates().copy()

df_encori_final = df_encori[
    (df_encori['miRNA_norm'].isin(expressed_mirnas_norm)) &
    (df_encori['Target_Gene_norm'].isin(expressed_genes_norm))
][['miRNA_norm', 'Target_Gene_norm', 'Source', 'Is_Database_Prior', 'Is_A549_Context']].drop_duplicates().copy()

print(f"    - miRTarBase edges mapped to A549 expression: {len(df_mirtar_final)}")
print(f"    - ENCORI edges mapped to A549 expression: {len(df_encori_final)}")

df_final = pd.concat([df_mirtar_final, df_encori_final], ignore_index=True).drop_duplicates(
    subset=['miRNA_norm', 'Target_Gene_norm']
).copy()

df_final['miRNA'] = df_final['miRNA_norm'].map(mirna_lookup)
df_final['Target Gene'] = df_final['Target_Gene_norm'].map(gene_lookup)

df_final = df_final[
    df_final['miRNA'].notna() &
    df_final['Target Gene'].notna()
].copy()

print(f"    - Combined prior edges after A549 mapping: {len(df_final)}")

print("--- Step 4: Exporting Edge List for Graph ---")

final_edges = df_final[['miRNA', 'Target Gene']].drop_duplicates()
final_edges.to_csv(OUTPUT_FILE, index=False)

metadata_edges = df_final[['miRNA', 'Target Gene', 'Source', 'Is_Database_Prior', 'Is_A549_Context']].drop_duplicates()
metadata_edges.to_csv(METADATA_FILE, index=False)

print(f"Total High-Confidence Prior Edges: {len(final_edges)}")
print(f"Final file saved as: {OUTPUT_FILE}")
print(f"Metadata file saved as: {METADATA_FILE}")
