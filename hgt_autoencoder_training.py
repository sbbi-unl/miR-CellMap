import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch_geometric.nn import HGTConv, Linear
import torch.nn.functional as F
import torch.serialization
from torch_geometric.data.storage import NodeStorage, EdgeStorage, BaseStorage

torch.serialization.add_safe_globals([NodeStorage, EdgeStorage, BaseStorage])

output_dir = "phase3_aligned_data"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"--- Step 3: HGT Autoencoder Training from {output_dir} ---")
print(f"Using device: {device}")


# 1 Load ID Mappings

gene_list = pd.read_csv(f"{output_dir}/gene_ids.txt", header=None)[0].astype(str).tolist()
mirna_list = pd.read_csv(f"{output_dir}/mirna_ids.txt", header=None)[0].astype(str).tolist()
cell_list = pd.read_csv(f"{output_dir}/cell_ids.txt", header=None)[0].astype(str).tolist()

gene_map = {name: i for i, name in enumerate(gene_list)}
mirna_map = {name: i for i, name in enumerate(mirna_list)}
cell_map = {name: i for i, name in enumerate(cell_list)}

print(f"   - Genes: {len(gene_list)}")
print(f"   - miRNAs: {len(mirna_list)}")
print(f"   - Cells: {len(cell_list)}")


# 2 Load Graph Object

data = torch.load(f"{output_dir}/tripartite_graph_object.pt", weights_only=False).to(device)

#3 Load Sparse Triplets
mrna_triplet = pd.read_csv(f"{output_dir}/A549_mRNA_sparse_triplet.csv")
mirna_triplet = pd.read_csv(f"{output_dir}/A549_miRNA_sparse_triplet.csv")

mrna_triplet["Gene"] = mrna_triplet["Gene"].astype(str)
mrna_triplet["Cell_ID"] = mrna_triplet["Cell_ID"].astype(str)
mirna_triplet["miRNA"] = mirna_triplet["miRNA"].astype(str)
mirna_triplet["Cell_ID"] = mirna_triplet["Cell_ID"].astype(str)

# Filter to valid IDs
mrna_before = len(mrna_triplet)
mirna_before = len(mirna_triplet)

mrna_triplet = mrna_triplet[
    mrna_triplet["Gene"].isin(gene_map) &
    mrna_triplet["Cell_ID"].isin(cell_map)
].copy()

mirna_triplet = mirna_triplet[
    mirna_triplet["miRNA"].isin(mirna_map) &
    mirna_triplet["Cell_ID"].isin(cell_map)
].copy()

print(f"   - mRNA triplets kept: {len(mrna_triplet)} / {mrna_before}")
print(f"   - miRNA triplets kept: {len(mirna_triplet)} / {mirna_before}")


# 4 Initialize Features

mrna_matrix = pd.read_csv(f"{output_dir}/A549_aligned_mRNA.csv", index_col=0)
mirna_matrix = pd.read_csv(f"{output_dir}/A549_aligned_miRNA.csv", index_col=0)

# Ensure row order matches ID lists
mrna_matrix.index = mrna_matrix.index.astype(str)
mirna_matrix.index = mirna_matrix.index.astype(str)

missing_genes = [g for g in gene_list if g not in mrna_matrix.index]
missing_mirnas = [m for m in mirna_list if m not in mirna_matrix.index]

if len(missing_genes) > 0:
    raise ValueError(f"{len(missing_genes)} genes from gene_ids.txt are missing in A549_aligned_mRNA.csv")
if len(missing_mirnas) > 0:
    raise ValueError(f"{len(missing_mirnas)} miRNAs from mirna_ids.txt are missing in A549_aligned_miRNA.csv")

mrna_matrix = mrna_matrix.loc[gene_list]
mirna_matrix = mirna_matrix.loc[mirna_list]

print(f"   - mRNA matrix min/max: {mrna_matrix.values.min():.4f} / {mrna_matrix.values.max():.4f}")
print(f"   - miRNA matrix min/max: {mirna_matrix.values.min():.4f} / {mirna_matrix.values.max():.4f}")

data['gene'].x = torch.tensor(mrna_matrix.values, dtype=torch.float).to(device)
data['mirna'].x = torch.tensor(mirna_matrix.values, dtype=torch.float).to(device)
data['cell'].x = torch.eye(data['cell'].num_nodes, dtype=torch.float).to(device)

print(f"   - gene feature matrix: {data['gene'].x.shape}")
print(f"   - miRNA feature matrix: {data['mirna'].x.shape}")
print(f"   - cell feature matrix: {data['cell'].x.shape}")

# 5 Define Model

class HGTAutoencoder(torch.nn.Module):
    def __init__(self, hidden_channels, num_heads, num_layers, metadata):
        super().__init__()
        self.node_lins = torch.nn.ModuleDict()
        for node_type in ['gene', 'mirna', 'cell']:
            self.node_lins[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HGTConv(hidden_channels, hidden_channels, metadata, num_heads))

        self.decoder = nn.Sequential(
            nn.Linear(hidden_channels * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def encode(self, x_dict, edge_index_dict):
        x_dict = {k: v for k, v in x_dict.items()}
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.node_lins[node_type](x).relu()
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
        return x_dict

model = HGTAutoencoder(
    hidden_channels=128,
    num_heads=4,
    num_layers=2,
    metadata=data.metadata()
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# 6 Prepare Supervision Targets

mrna_labels = torch.tensor(mrna_triplet['Expression'].values, dtype=torch.float).to(device)
mirna_labels = torch.tensor(mirna_triplet['Expression'].values, dtype=torch.float).to(device)

mrna_pos = torch.tensor([
    [gene_map[g] for g in mrna_triplet['Gene']],
    [cell_map[c] for c in mrna_triplet['Cell_ID']]
], dtype=torch.long).to(device)

mirna_pos = torch.tensor([
    [mirna_map[m] for m in mirna_triplet['miRNA']],
    [cell_map[c] for c in mirna_triplet['Cell_ID']]
], dtype=torch.long).to(device)

print("--- Step 3: Training HGT Autoencoder ---")

def get_scores(src_emb, tgt_emb, edges, decoder):
    input_pairs = torch.cat([src_emb[edges[0]], tgt_emb[edges[1]]], dim=1)
    return decoder(input_pairs).squeeze()


# 7 Train

model.train()
for epoch in range(1, 101):
    optimizer.zero_grad()

    z_dict = model.encode(data.x_dict, data.edge_index_dict)

    mrna_scores = get_scores(z_dict['gene'], z_dict['cell'], mrna_pos, model.decoder)
    mirna_scores = get_scores(z_dict['mirna'], z_dict['cell'], mirna_pos, model.decoder)

    loss = F.mse_loss(mrna_scores, mrna_labels) + F.mse_loss(mirna_scores, mirna_labels)

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d}, Loss: {loss.item():.4f}")


# 8 Final Export

model.eval()
with torch.no_grad():
    z_final = model.encode(data.x_dict, data.edge_index_dict)

    # Save embeddings 
    pd.DataFrame(z_final['gene'].cpu().numpy()).to_csv(f"{output_dir}/gene_embeddings.csv", index=False)
    pd.DataFrame(z_final['mirna'].cpu().numpy()).to_csv(f"{output_dir}/mirna_embeddings.csv", index=False)
    pd.DataFrame(z_final['cell'].cpu().numpy()).to_csv(f"{output_dir}/cell_embeddings.csv", index=False)

    torch.save(model.state_dict(), f"{output_dir}/hgt_trained_model.pt")

    # Save attention-like interaction scores for miRNA-gene prior edges
    edge_index = data['mirna', 'targets', 'gene'].edge_index
    attention_scores = torch.sigmoid(
        torch.sum(
            z_final['mirna'][edge_index[0]] * z_final['gene'][edge_index[1]],
            dim=1
        )
    ).cpu().numpy()

    pd.DataFrame({
        'miRNA': [mirna_list[i] for i in edge_index[0].cpu().numpy()],
        'Target Gene': [gene_list[i] for i in edge_index[1].cpu().numpy()],
        'Attention_Weight': attention_scores
    }).to_csv(f"{output_dir}/hgt_attention_edges.csv", index=False)

print(f"\nStep 3 Complete!")
print(f"   - Saved: {output_dir}/gene_embeddings.csv")
print(f"   - Saved: {output_dir}/mirna_embeddings.csv")
print(f"   - Saved: {output_dir}/cell_embeddings.csv")
print(f"   - Saved: {output_dir}/hgt_trained_model.pt")
print(f"   - Saved: {output_dir}/hgt_attention_edges.csv")
