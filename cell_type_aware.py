import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")

output_dir = "phase3_aligned_data"

final_net_file = f"{output_dir}/final_network_inference.csv"
program_summary_file = f"{output_dir}/Figure3_program_distance_summary.csv"
cell_umap_file = f"{output_dir}/cell_umap_coords.csv"
mrna_expr_file = f"{output_dir}/A549_aligned_mRNA.csv"

top_marker_genes_per_cluster = 300
top_mirnas_rank = 5
top_mirnas_network = 4
top_targets_per_mirna = 4
selected_cluster = None
correlation_mode = "absolute"
random_seed = 42

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.titlesize": 15,
    "pdf.fonttype": 42
})

print("--- Building cleaner manuscript Figure 4 ---")


def require_file(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")


def normalize_name(x):
    return str(x).strip()


def pick_cluster(program_df, user_cluster=None):
    if user_cluster is not None:
        if user_cluster not in set(program_df["best_cluster"].astype(int)):
            raise ValueError(f"selected_cluster={user_cluster} not found.")
        return int(user_cluster)

    tmp = (
        program_df.groupby("best_cluster")["shift_random_minus_observed"]
        .sum()
        .sort_values(ascending=False)
    )
    return int(tmp.index[0])


def make_cluster_marker_dict(mrna_expr, cell_clusters, top_n=300):
    common_cells = [c for c in mrna_expr.columns if c in cell_clusters.index]
    if len(common_cells) == 0:
        raise ValueError("No overlapping cells between expression matrix and cluster file.")

    mrna_expr = mrna_expr[common_cells]
    cluster_labels = cell_clusters.loc[common_cells].values

    marker_dict = {}
    for c in sorted(np.unique(cluster_labels)):
        in_cluster = cluster_labels == c
        out_cluster = cluster_labels != c

        cluster_mean = mrna_expr.iloc[:, in_cluster].mean(axis=1)
        other_mean = mrna_expr.iloc[:, out_cluster].mean(axis=1)
        marker_score = (cluster_mean - other_mean).sort_values(ascending=False)

        marker_dict[int(c)] = marker_score.head(top_n).index.astype(str).tolist()

    return marker_dict


def edge_subset_for_cluster(final_net, cluster_mirnas, cluster_genes):
    return final_net[
        final_net["miRNA"].isin(cluster_mirnas) &
        final_net["Target Gene"].isin(cluster_genes)
    ].copy()


def build_hgt_network_df(cluster_df, top_mirnas=4, top_targets=4):
    if cluster_df.empty:
        raise ValueError("No HGT edges available after cluster-specific filtering.")

    mir_rank = (
        cluster_df.groupby("miRNA")["Final_miR_CellMap_Score"]
        .sum()
        .sort_values(ascending=False)
    )

    chosen_mirs = mir_rank.head(top_mirnas).index.tolist()

    hgt_df = (
        cluster_df[cluster_df["miRNA"].isin(chosen_mirs)]
        .sort_values(["miRNA", "Final_miR_CellMap_Score"], ascending=[True, False])
        .groupby("miRNA")
        .head(top_targets)
        .copy()
    )

    return hgt_df, mir_rank


def build_corr_network_df(cluster_df, chosen_mirs, n_edges):
    corr_df = cluster_df[cluster_df["miRNA"].isin(chosen_mirs)].copy()

    if correlation_mode == "absolute":
        corr_df["corr_rank_value"] = corr_df["CC"].abs()
    else:
        corr_df["corr_rank_value"] = corr_df["CC"]

    corr_df = corr_df.sort_values("corr_rank_value", ascending=False).head(n_edges).copy()
    return corr_df


def draw_bipartite_network(ax, df_edges, title, edge_weight_col,
                           mir_color="salmon", gene_color="skyblue",
                           layout_seed=42, label_font=10):
    G = nx.Graph()

    for _, row in df_edges.iterrows():
        G.add_edge(
            row["miRNA"],
            row["Target Gene"],
            weight=float(row[edge_weight_col])
        )

    if len(G.nodes()) == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No edges", ha="center", va="center")
        ax.axis("off")
        return

    mirnas = [
        n for n in G.nodes()
        if str(n).startswith("hsa-") or str(n).startswith("miR") or str(n).startswith("mir")
    ]
    genes = [n for n in G.nodes() if n not in mirnas]

    pos = nx.spring_layout(G, seed=layout_seed, k=0.95)

    nx.draw_networkx_nodes(G, pos, nodelist=mirnas, node_color=mir_color, node_size=1800, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=genes, node_color=gene_color, node_size=850, ax=ax)

    edges_sorted = sorted(G.edges(data=True), key=lambda x: x[2]["weight"], reverse=True)
    widths = [max(0.8, float(attr["weight"]) * 1.8) for _, _, attr in edges_sorted]

    nx.draw_networkx_edges(
        G,
        pos,
        edgelist=[(u, v) for u, v, _ in edges_sorted],
        width=widths,
        edge_color="gray",
        alpha=0.45,
        ax=ax
    )
    nx.draw_networkx_labels(G, pos, font_size=label_font, font_weight="bold", ax=ax)

    ax.set_title(title)
    ax.axis("off")


for f in [final_net_file, program_summary_file, cell_umap_file, mrna_expr_file]:
    require_file(f)

final_net = pd.read_csv(final_net_file)
program_df = pd.read_csv(program_summary_file)
cell_umap = pd.read_csv(cell_umap_file)
mrna_expr = pd.read_csv(mrna_expr_file, index_col=0)

final_net.columns = [c.strip() for c in final_net.columns]
program_df.columns = [c.strip() for c in program_df.columns]
cell_umap.columns = [c.strip() for c in cell_umap.columns]
mrna_expr.index = mrna_expr.index.astype(str)
mrna_expr.columns = mrna_expr.columns.astype(str)

required_final_cols = {"miRNA", "Target Gene", "Final_miR_CellMap_Score", "CC"}
missing_final = required_final_cols - set(final_net.columns)
if missing_final:
    raise ValueError(f"Missing in final_network_inference.csv: {missing_final}")

required_program_cols = {"miRNA", "best_cluster", "shift_random_minus_observed"}
missing_program = required_program_cols - set(program_df.columns)
if missing_program:
    raise ValueError(f"Missing in Figure3_program_distance_summary.csv: {missing_program}")

required_umap_cols = {"CellID", "Cluster"}
missing_umap = required_umap_cols - set(cell_umap.columns)
if missing_umap:
    raise ValueError(f"Missing in cell_umap_coords.csv: {missing_umap}")

final_net["miRNA"] = final_net["miRNA"].map(normalize_name)
final_net["Target Gene"] = final_net["Target Gene"].map(normalize_name)
program_df["miRNA"] = program_df["miRNA"].map(normalize_name)
cell_umap["CellID"] = cell_umap["CellID"].astype(str)
cell_umap["Cluster"] = cell_umap["Cluster"].astype(int)

chosen_cluster = pick_cluster(program_df, user_cluster=selected_cluster)
print(f"Selected representative cluster: C{chosen_cluster + 1}")

cell_clusters = pd.Series(cell_umap["Cluster"].values, index=cell_umap["CellID"].values)
cluster_marker_dict = make_cluster_marker_dict(
    mrna_expr=mrna_expr,
    cell_clusters=cell_clusters,
    top_n=top_marker_genes_per_cluster
)
cluster_genes = set(cluster_marker_dict[chosen_cluster])

cluster_mirnas = set(
    program_df.loc[program_df["best_cluster"].astype(int) == chosen_cluster, "miRNA"].tolist()
)

cluster_df = edge_subset_for_cluster(final_net, cluster_mirnas, cluster_genes)

if cluster_df.empty:
    raise ValueError(f"No cluster-specific edges found for cluster {chosen_cluster}")

hgt_network_df, mir_rank_full = build_hgt_network_df(
    cluster_df=cluster_df,
    top_mirnas=top_mirnas_network,
    top_targets=top_targets_per_mirna
)

rank_df = (
    mir_rank_full.reset_index()
    .rename(columns={"index": "miRNA", "Final_miR_CellMap_Score": "Cluster_miR_CellMap_Score"})
    .head(top_mirnas_rank)
)

chosen_mirs = hgt_network_df["miRNA"].unique().tolist()
corr_network_df = build_corr_network_df(
    cluster_df=cluster_df,
    chosen_mirs=chosen_mirs,
    n_edges=len(hgt_network_df)
)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='miRNA', markerfacecolor='salmon', markersize=14),
    Line2D([0], [0], marker='o', color='w', label='Target Gene', markerfacecolor='skyblue', markersize=10)
]

fig_a, ax_a = plt.subplots(figsize=(8, 6))
draw_bipartite_network(
    ax=ax_a,
    df_edges=hgt_network_df,
    title=f"A. Cluster-Specific HGT Network (C{chosen_cluster + 1})",
    edge_weight_col="Final_miR_CellMap_Score",
    layout_seed=random_seed,
    label_font=9
)
ax_a.legend(handles=legend_elements, loc='upper right', frameon=True)
plt.tight_layout()
plt.savefig(f"{output_dir}/Figure4_A_Cluster_Specific_HGT_Network.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{output_dir}/Figure4_A_Cluster_Specific_HGT_Network.pdf", bbox_inches="tight")
plt.close(fig_a)

fig_b, ax_b = plt.subplots(figsize=(7, 6))
sns.barplot(
    data=rank_df,
    x="Cluster_miR_CellMap_Score",
    y="miRNA",
    hue="miRNA",
    palette="viridis",
    legend=False,
    ax=ax_b
)
ax_b.set_title(f"B. Regulator Ranking in C{chosen_cluster + 1}")
ax_b.set_xlabel("Cluster-Specific miR-CellMap Score")
ax_b.set_ylabel("miRNA")
plt.tight_layout()
plt.savefig(f"{output_dir}/Figure4_B_Regulator_Ranking.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{output_dir}/Figure4_B_Regulator_Ranking.pdf", bbox_inches="tight")
plt.close(fig_b)

fig_c, ax_c = plt.subplots(figsize=(8, 6))
draw_bipartite_network(
    ax=ax_c,
    df_edges=corr_network_df,
    title=f"C. Correlation Network (C{chosen_cluster + 1})",
    edge_weight_col="corr_rank_value",
    layout_seed=random_seed,
    label_font=9
)
plt.tight_layout()
plt.savefig(f"{output_dir}/Figure4_C_Correlation_Network.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{output_dir}/Figure4_C_Correlation_Network.pdf", bbox_inches="tight")
plt.close(fig_c)

hgt_network_df.to_csv(f"{output_dir}/Figure4_cluster_specific_hgt_network.csv", index=False)
rank_df.to_csv(f"{output_dir}/Figure4_cluster_specific_rankings.csv", index=False)
corr_network_df.to_csv(f"{output_dir}/Figure4_cluster_specific_correlation_network.csv", index=False)

print(f"Saved separate figure panels to: {output_dir}")
print("Saved supporting CSV files.")
