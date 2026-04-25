import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import umap
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import os

# Configuration
output_dir = "phase3_aligned_data"
random_seed = 42
n_clusters = 3
top_panel_b_candidates = 15
top_mirnas_panel_c_scan = 15
top_positive_mirnas_for_plot = 3
min_targets_total = 10
min_targets_in_cluster_program = 5
top_marker_genes_per_cluster = 300
k_nearest_targets = 10
n_random_sets = 100
distance_metric = "cosine"   # "cosine" or "euclidean"

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.titlesize": 15
})

print("--- Generating Final Polished Figure 3: Joint miRNA–Gene Embedding ---")

# Load data
cell_emb = pd.read_csv(f"{output_dir}/cell_embeddings.csv")
mirna_emb = pd.read_csv(f"{output_dir}/mirna_embeddings.csv")
gene_emb = pd.read_csv(f"{output_dir}/gene_embeddings.csv")
mirna_expr = pd.read_csv(f"{output_dir}/A549_aligned_miRNA.csv", index_col=0)
priors = pd.read_csv(f"{output_dir}/mirna_gene_prior.csv")

gene_ids = pd.read_csv(f"{output_dir}/gene_ids.txt", header=None)[0].astype(str).tolist()
mirna_ids = pd.read_csv(f"{output_dir}/mirna_ids.txt", header=None)[0].astype(str).tolist()
cell_ids = pd.read_csv(f"{output_dir}/cell_ids.txt", header=None)[0].astype(str).tolist()

mirna_emb.index = mirna_ids
gene_emb.index = gene_ids

mrna_expr_path = f"{output_dir}/A549_aligned_mRNA.csv"
if not os.path.exists(mrna_expr_path):
    raise FileNotFoundError(f"Required file not found: {mrna_expr_path}")
mrna_expr = pd.read_csv(mrna_expr_path, index_col=0)

if cell_emb.shape[0] != mirna_expr.shape[1]:
    raise ValueError(
        f"Mismatch: cell_embeddings has {cell_emb.shape[0]} rows, "
        f"but A549_aligned_miRNA has {mirna_expr.shape[1]} columns."
    )

if cell_emb.shape[0] != mrna_expr.shape[1]:
    raise ValueError(
        f"Mismatch: cell_embeddings has {cell_emb.shape[0]} rows, "
        f"but A549_aligned_mRNA has {mrna_expr.shape[1]} columns."
    )

if len(cell_ids) != cell_emb.shape[0]:
    raise ValueError(
        f"Mismatch: cell_ids has {len(cell_ids)} IDs, "
        f"but cell_embeddings has {cell_emb.shape[0]} rows."
    )

mirna_expr_cols = [str(x) for x in mirna_expr.columns]
mrna_expr_cols = [str(x) for x in mrna_expr.columns]

if mirna_expr_cols != cell_ids:
    if set(mirna_expr_cols) == set(cell_ids):
        mirna_expr = mirna_expr[cell_ids]
    else:
        raise ValueError("A549_aligned_miRNA columns do not match cell_ids.txt")

if mrna_expr_cols != cell_ids:
    if set(mrna_expr_cols) == set(cell_ids):
        mrna_expr = mrna_expr[cell_ids]
    else:
        raise ValueError("A549_aligned_mRNA columns do not match cell_ids.txt")

# functions
def safe_zscore(values):
    values = pd.Series(values)
    std = values.std()
    if std == 0 or pd.isna(std):
        return np.zeros(len(values))
    return ((values - values.mean()) / std).values

def compute_cluster_specificity(expr_vector, labels):
    expr_vector = np.asarray(expr_vector, dtype=float)
    labels = np.asarray(labels)
    overall_mean = expr_vector.mean()

    between = 0.0
    within = 0.0
    for c in np.unique(labels):
        vals = expr_vector[labels == c]
        if len(vals) == 0:
            continue
        c_mean = vals.mean()
        between += len(vals) * (c_mean - overall_mean) ** 2
        within += np.sum((vals - c_mean) ** 2)

    if within == 0:
        return np.inf
    return between / within

def robust_program_distance(mir_vec, gene_matrix, k=10, metric="cosine"):
    if gene_matrix.shape[0] == 0:
        return np.nan
    dists = cdist(mir_vec.reshape(1, -1), gene_matrix.values, metric=metric)[0]
    k_eff = min(k, len(dists))
    return float(np.median(np.sort(dists)[:k_eff]))

def choose_panel_b_mirna(candidate_df, mirna_expr, cluster_labels, top_n=15):
    top_candidates = candidate_df.head(top_n)["miRNA"].tolist()
    best_mir = None
    best_score = -np.inf

    for mir in top_candidates:
        vals = mirna_expr.loc[mir].values.astype(float)
        cluster_means = np.array([vals[cluster_labels == c].mean() for c in np.unique(cluster_labels)])
        visual_score = cluster_means.max() - cluster_means.min()
        if visual_score > best_score:
            best_score = visual_score
            best_mir = mir
    return best_mir

# Step 1: Cell UMAP and clustering
reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=random_seed)
embedding_2d = reducer.fit_transform(cell_emb)

kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed, n_init=10)
cluster_labels = kmeans.fit_predict(cell_emb)

umap_df = pd.DataFrame(embedding_2d, columns=["UMAP_1", "UMAP_2"])
umap_df["Cluster"] = cluster_labels
umap_df["CellID"] = cell_ids
umap_df.to_csv(f"{output_dir}/cell_umap_coords.csv", index=False)

# Step 2: Panel B miRNA selection
candidate_scores = []
for mir in mirna_expr.index.astype(str):
    vals = mirna_expr.loc[mir].values.astype(float)
    if np.all(vals == vals[0]):
        continue
    score = compute_cluster_specificity(vals, cluster_labels)
    variance = np.var(vals)
    candidate_scores.append((mir, score, variance))

if len(candidate_scores) == 0:
    raise ValueError("No valid miRNAs found for Panel B selection.")

candidate_df = pd.DataFrame(
    candidate_scores,
    columns=["miRNA", "cluster_specificity", "variance"]
).sort_values(
    by=["cluster_specificity", "variance"],
    ascending=[False, False]
).reset_index(drop=True)

selected_mirna = choose_panel_b_mirna(
    candidate_df=candidate_df,
    mirna_expr=mirna_expr,
    cluster_labels=cluster_labels,
    top_n=top_panel_b_candidates
)

selected_vals = mirna_expr.loc[selected_mirna].values.astype(float)
z_vals = safe_zscore(selected_vals)

candidate_df.to_csv(f"{output_dir}/Figure3_miRNA_selection_scores.csv", index=False)

# Step 3: Define cluster-specific gene programs
mrna_expr.index = mrna_expr.index.astype(str)
valid_genes_for_programs = set(mrna_expr.index).intersection(set(gene_emb.index))

cluster_marker_dict = {}
for c in sorted(np.unique(cluster_labels)):
    in_cluster = cluster_labels == c
    out_cluster = cluster_labels != c

    cluster_mean = mrna_expr.iloc[:, in_cluster].mean(axis=1)
    other_mean = mrna_expr.iloc[:, out_cluster].mean(axis=1)

    marker_score = (cluster_mean - other_mean)
    marker_score = marker_score.loc[marker_score.index.isin(valid_genes_for_programs)]
    marker_score = marker_score.sort_values(ascending=False)

    cluster_marker_dict[c] = marker_score.head(top_marker_genes_per_cluster).index.tolist()

# Step 4: Scan informative miRNAs and keep the top positive ones
priors_valid = priors.copy()
priors_valid["miRNA"] = priors_valid["miRNA"].astype(str)
priors_valid["Target Gene"] = priors_valid["Target Gene"].astype(str)

priors_valid = priors_valid[
    priors_valid["miRNA"].isin(mirna_emb.index) &
    priors_valid["Target Gene"].isin(gene_emb.index)
].drop_duplicates().copy()

valid_expr_mirnas = set(mirna_expr.index.astype(str)).intersection(set(mirna_emb.index))
priors_valid = priors_valid[priors_valid["miRNA"].isin(valid_expr_mirnas)].copy()

informative_mirnas = candidate_df["miRNA"].tolist()
mirnas_to_scan = []

for mir in informative_mirnas:
    n_tg = priors_valid.loc[priors_valid["miRNA"] == mir, "Target Gene"].nunique()
    if n_tg >= min_targets_total:
        mirnas_to_scan.append(mir)
    if len(mirnas_to_scan) >= top_mirnas_panel_c_scan:
        break

if len(mirnas_to_scan) == 0:
    raise ValueError("No informative miRNAs passed the minimum target threshold for Panel C.")

rng = np.random.default_rng(random_seed)
scan_rows = []

for mir in mirnas_to_scan:
    mir_targets = set(priors_valid.loc[priors_valid["miRNA"] == mir, "Target Gene"].unique().tolist())
    mir_vec = mirna_emb.loc[mir].values

    best_cluster = None
    best_observed_dist = None
    best_overlap = None
    best_random_median = None

    for c in sorted(cluster_marker_dict.keys()):
        cluster_program = set(cluster_marker_dict[c])
        overlap_targets = sorted(mir_targets.intersection(cluster_program))
        overlap_targets = [g for g in overlap_targets if g in gene_emb.index]

        if len(overlap_targets) < min_targets_in_cluster_program:
            continue

        observed_dist = robust_program_distance(
            mir_vec=mir_vec,
            gene_matrix=gene_emb.loc[overlap_targets],
            k=k_nearest_targets,
            metric=distance_metric
        )

        cluster_gene_pool = np.array([g for g in cluster_marker_dict[c] if g in gene_emb.index])
        if len(cluster_gene_pool) == 0:
            continue

        rand_dists = []
        sample_size = len(overlap_targets)
        for _ in range(n_random_sets):
            replace_flag = sample_size > len(cluster_gene_pool)
            rand_genes = rng.choice(cluster_gene_pool, size=sample_size, replace=replace_flag)
            rand_dist = robust_program_distance(
                mir_vec=mir_vec,
                gene_matrix=gene_emb.loc[rand_genes],
                k=k_nearest_targets,
                metric=distance_metric
            )
            rand_dists.append(rand_dist)

        rand_median = float(np.median(rand_dists))
        shift = rand_median - observed_dist

        if (best_observed_dist is None) or (shift > (best_random_median - best_observed_dist)):
            best_cluster = c
            best_observed_dist = float(observed_dist)
            best_overlap = overlap_targets
            best_random_median = rand_median

    if best_cluster is not None:
        scan_rows.append({
            "miRNA": mir,
            "best_cluster": int(best_cluster),
            "n_total_targets": int(len(mir_targets)),
            "n_targets_in_cluster_program": int(len(best_overlap)),
            "observed_program_distance": float(best_observed_dist),
            "median_random_program_distance": float(best_random_median),
            "shift_random_minus_observed": float(best_random_median - best_observed_dist)
        })

scan_df = pd.DataFrame(scan_rows).sort_values(
    by="shift_random_minus_observed",
    ascending=False
)

if scan_df.empty:
    raise ValueError("No miRNAs produced valid cluster-aware program results.")

scan_df.to_csv(f"{output_dir}/Figure3_program_distance_summary.csv", index=False)

top_positive_df = scan_df[scan_df["shift_random_minus_observed"] > 0].head(top_positive_mirnas_for_plot).copy()
if top_positive_df.empty:
    raise ValueError("No positively shifted miRNAs were found for manuscript-ready Panel C.")

# Step 5: Build Panel C only from top positive miRNAs
observed_program_distances = []
random_program_distances = []

for _, row in top_positive_df.iterrows():
    mir = row["miRNA"]
    best_cluster = int(row["best_cluster"])

    mir_targets = set(priors_valid.loc[priors_valid["miRNA"] == mir, "Target Gene"].unique().tolist())
    overlap_targets = sorted(mir_targets.intersection(set(cluster_marker_dict[best_cluster])))
    overlap_targets = [g for g in overlap_targets if g in gene_emb.index]

    mir_vec = mirna_emb.loc[mir].values
    obs_dist = robust_program_distance(
        mir_vec=mir_vec,
        gene_matrix=gene_emb.loc[overlap_targets],
        k=k_nearest_targets,
        metric=distance_metric
    )
    observed_program_distances.append(obs_dist)

    cluster_gene_pool = np.array([g for g in cluster_marker_dict[best_cluster] if g in gene_emb.index])
    sample_size = len(overlap_targets)

    for _ in range(n_random_sets):
        replace_flag = sample_size > len(cluster_gene_pool)
        rand_genes = rng.choice(cluster_gene_pool, size=sample_size, replace=replace_flag)
        rand_dist = robust_program_distance(
            mir_vec=mir_vec,
            gene_matrix=gene_emb.loc[rand_genes],
            k=k_nearest_targets,
            metric=distance_metric
        )
        random_program_distances.append(rand_dist)

observed_program_distances = np.array(observed_program_distances)
random_program_distances = np.array(random_program_distances)

# Step 6: Plot figures separately

# Panel A
plt.figure(figsize=(7.5, 6.5))
plt.scatter(
    embedding_2d[:, 0],
    embedding_2d[:, 1],
    s=22,
    alpha=0.78,
    c=cluster_labels,
    cmap="tab10"
)
plt.title("A. UMAP of the Joint latent space showing cluster structure.")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")

for i in range(n_clusters):
    pts = embedding_2d[cluster_labels == i]
    if len(pts) == 0:
        continue
    centroid = np.median(pts, axis=0)
    plt.text(
        centroid[0], centroid[1], f"C{i+1}",
        weight="bold",
        fontsize=14,
        bbox=dict(facecolor="white", alpha=0.65, edgecolor="none", pad=2)
    )

plt.tight_layout()
plt.savefig(f"{output_dir}/Figure3_A_UMAP.png", dpi=300)
plt.savefig(f"{output_dir}/Figure3_A_UMAP.pdf")
plt.close()

# Panel B
plt.figure(figsize=(7.5, 6.5))
scatter_b = plt.scatter(
    embedding_2d[:, 0],
    embedding_2d[:, 1],
    s=22,
    c=z_vals,
    cmap="viridis"
)
plt.colorbar(scatter_b, label="Z-scored Expression")
plt.title("B. UMAP colored by selected top-ranked miRNAs")
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")

plt.tight_layout()
plt.savefig(f"{output_dir}/Figure3_B_miRNA_Activity.png", dpi=300)
plt.savefig(f"{output_dir}/Figure3_B_miRNA_Activity.pdf")
plt.close()

# Panel C
plt.figure(figsize=(7.5, 6.5))
sns.kdeplot(
    observed_program_distances,
    fill=True,
    color="#8E44AD",
    lw=2,
    label=f"Observed Top miRNA Programs (n={len(observed_program_distances)})"
)
sns.kdeplot(
    random_program_distances,
    fill=True,
    color="#BDC3C7",
    lw=2,
    label=f"Random Matched Programs (n={len(random_program_distances)})"
)

m_obs = float(np.median(observed_program_distances))
m_rand = float(np.median(random_program_distances))

plt.axvline(m_obs, color="#8E44AD", linestyle="--", lw=1.5)
plt.axvline(m_rand, color="#7F8C8D", linestyle="--", lw=1.5)

metric_label = "Cosine Distance" if distance_metric == "cosine" else "Euclidean Distance"
top_names = ", ".join(top_positive_df["miRNA"].tolist())
plt.title("C. Proximity of top-ranked miRNAs to gene programs")
plt.xlabel(f"Median {metric_label} to {k_nearest_targets} Nearest Target Genes")
plt.ylabel("Density")
plt.legend(loc="upper right")

plt.tight_layout()
plt.savefig(f"{output_dir}/Figure3_C_Program_Distance.png", dpi=300)
plt.savefig(f"{output_dir}/Figure3_C_Program_Distance.pdf")
plt.close()
