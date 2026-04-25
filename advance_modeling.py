import os
import warnings
import itertools
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots

# 1. Main evaluation and local PDCD4 figures

plt.style.use(['science', 'no-latex'])
plt.rcParams['text.usetex'] = False
sns.set_palette("viridis")
warnings.filterwarnings("ignore", category=FutureWarning)

# 2. Path configuration
output_dir = "phase3_aligned_data"

inference_file = f"{output_dir}/final_network_inference.csv"
prior_file = f"{output_dir}/mirna_gene_prior.csv"
a549_verified_file = f"{output_dir}/ENCORI_A549_verified_interactions.csv"

global_encori_file = "ENCORI_hg38_miRNA_mRNA_all.txt"
pdcd4_clash_file = "ENCORI_hg38_CLIP-seq_miRNA-target_all_PDCD4.xls"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'pdf.fonttype': 42,
    'font.size': 7,
    'axes.labelweight': 'bold',
    'axes.titlesize': 8,
    'axes.titleweight': 'bold',
    'axes.linewidth': 0.6,
    'xtick.labelsize': 6,
    'ytick.labelsize': 6,
    'legend.fontsize': 6,
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'axes.facecolor': 'white',
    'figure.facecolor': 'white'
})

COLOR_BASE = '#7f8c8d'
COLOR_REF = '#2c3e50'
COLOR_POINT = '#2980b9'
COLOR_OTHER = '#d9d9d9'
COLOR_TOP1 = '#4c72b0'
COLOR_TOP2 = '#2a9d8f'
COLOR_TOP3 = '#6abf69'

print("Step 5: Main evaluation and local PDCD4 figures")

# 3. Helper functions
def normalize_name(x):
    return str(x).strip().upper()

def normalize_gene(x):
    return normalize_name(x)

def normalize_mirna_exact(x):
    x = normalize_name(x)
    if not x.startswith("HSA-") and "MIR" in x:
        x = "HSA-" + x
    return x

def normalize_mirna_stem(x):
    x = normalize_mirna_exact(x)
    x = x.replace("-5P", "").replace("-3P", "")
    return x

def robust_read_pdcd4_table(path):
    header_line = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            if "miRNAid" in line or "miRNAname" in line:
                header_line = i
                break

    df = pd.read_table(path, skiprows=header_line, sep="\t")
    df.columns = [str(c).strip() for c in df.columns]
    return df

def robust_read_global_encori(path):
    df = pd.read_csv(
        path,
        sep=r"\s+",
        skiprows=3,
        engine="python"
    )
    df.columns = [str(c).strip() for c in df.columns]
    return df

def finalize(filename):
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.6)
    ax.spines['bottom'].set_linewidth(0.6)
    ax.grid(True, axis='y', alpha=0.25)
    plt.tight_layout(pad=0.5)
    plt.savefig(f"{output_dir}/{filename}.pdf", bbox_inches='tight')
    plt.savefig(f"{output_dir}/{filename}.png", dpi=600, bbox_inches='tight')
    plt.close()

def compute_binary_metrics(tp, fp, fn, tn):
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1 = (2 * precision * sensitivity / (precision + sensitivity)) if (precision + sensitivity) > 0 else 0.0

    denom = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0

    return {
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Accuracy": accuracy,
        "Precision": precision,
        "F1": f1,
        "MCC": mcc
    }

def pdcd4_overlap_precision(model_df, pdcd4_truth_df, target="PDCD4"):
    subset = model_df[model_df["Gene_norm"] == target].copy()
    predicted_edges = set(zip(subset["miRNA_exact"], subset["Gene_norm"]))
    truth_edges = set(zip(pdcd4_truth_df["miRNA_exact"], pdcd4_truth_df["Gene_norm"]))
    overlap = predicted_edges.intersection(truth_edges)
    return (len(overlap) / len(predicted_edges) * 100) if len(predicted_edges) > 0 else 0.0

# 4. Check required files
for fp in [inference_file, prior_file, a549_verified_file, global_encori_file, pdcd4_clash_file]:
    if not os.path.exists(fp):
        raise FileNotFoundError(f"Missing required file: {fp}")

# 5. Load predicted BN graph
df = pd.read_csv(inference_file)

df["miRNA_exact"] = df["miRNA"].map(normalize_mirna_exact)
df["miRNA_stem"] = df["miRNA"].map(normalize_mirna_stem)
df["Gene_norm"] = df["Target Gene"].map(normalize_gene)

pred_edges_exact = set(zip(df["miRNA_exact"], df["Gene_norm"]))
pred_edges_stem = set(zip(df["miRNA_stem"], df["Gene_norm"]))

bn_mirnas_exact = set(df["miRNA_exact"])
bn_mirnas_stem = set(df["miRNA_stem"])
bn_genes = set(df["Gene_norm"])

# 6. Load prior scaffold
prior_df = pd.read_csv(prior_file)
prior_df["miRNA_exact"] = prior_df["miRNA"].map(normalize_mirna_exact)
prior_df["miRNA_stem"] = prior_df["miRNA"].map(normalize_mirna_stem)
prior_df["Gene_norm"] = prior_df["Target Gene"].map(normalize_gene)

prior_edges_exact = set(zip(prior_df["miRNA_exact"], prior_df["Gene_norm"]))
prior_edges_stem = set(zip(prior_df["miRNA_stem"], prior_df["Gene_norm"]))

# 7. Load A549-specific verified set
a549_df = pd.read_csv(a549_verified_file)
a549_df.columns = [str(c).strip() for c in a549_df.columns]

if "miRNAname" not in a549_df.columns or "geneName" not in a549_df.columns:
    raise ValueError("A549 verified file must contain 'miRNAname' and 'geneName' columns.")

a549_df["miRNA_exact"] = a549_df["miRNAname"].map(normalize_mirna_exact)
a549_df["miRNA_stem"] = a549_df["miRNAname"].map(normalize_mirna_stem)
a549_df["Gene_norm"] = a549_df["geneName"].map(normalize_gene)

a549_eval_df = a549_df[
    a549_df["Gene_norm"].isin(bn_genes) &
    a549_df["miRNA_exact"].isin(bn_mirnas_exact)
].drop_duplicates(subset=["miRNA_exact", "Gene_norm"]).copy()

positive_edges_exact = set(zip(a549_eval_df["miRNA_exact"], a549_eval_df["Gene_norm"]))
positive_edges_stem = set(zip(a549_eval_df["miRNA_stem"], a549_eval_df["Gene_norm"]))

a549_genes = sorted(set(a549_eval_df["Gene_norm"]))
candidate_mirnas_exact = sorted(bn_mirnas_exact)
candidate_mirnas_stem = sorted(bn_mirnas_stem)

print(f"A549 verified interactions loaded: {len(a549_df)}")
print(f"A549 verified interactions in BN universe: {len(a549_eval_df)}")
print(f"A549 genes in evaluation set: {len(a549_genes)}")
print(f"BN miRNAs available for evaluation: {len(candidate_mirnas_exact)}")

# 8. Load global ENCORI for higher-confidence negative filtering
global_encori = robust_read_global_encori(global_encori_file)

required_global_cols = {"miRNAname", "geneName"}
if not required_global_cols.issubset(set(global_encori.columns)):
    raise ValueError(f"Global ENCORI file must contain columns: {required_global_cols}")

global_encori["miRNA_exact"] = global_encori["miRNAname"].map(normalize_mirna_exact)
global_encori["miRNA_stem"] = global_encori["miRNAname"].map(normalize_mirna_stem)
global_encori["Gene_norm"] = global_encori["geneName"].map(normalize_gene)

global_edges_exact = set(zip(global_encori["miRNA_exact"], global_encori["Gene_norm"]))
global_edges_stem = set(zip(global_encori["miRNA_stem"], global_encori["Gene_norm"]))

# 9. Build negative sets
all_candidate_pairs_exact = set(itertools.product(candidate_mirnas_exact, a549_genes))
all_candidate_pairs_stem = set(itertools.product(candidate_mirnas_stem, a549_genes))

negative_low_exact = all_candidate_pairs_exact - positive_edges_exact
negative_low_stem = all_candidate_pairs_stem - positive_edges_stem

negative_high_exact = {
    edge for edge in negative_low_exact
    if edge not in global_edges_exact and edge not in prior_edges_exact
}
negative_high_stem = {
    edge for edge in negative_low_stem
    if edge not in global_edges_stem and edge not in prior_edges_stem
}

# 10. Binary evaluation
def evaluate_binary(pred_exact, pred_stem, pos_exact, pos_stem, neg_exact, neg_stem, label):
    pred_eval_exact = pred_exact.intersection(pos_exact.union(neg_exact))
    pred_eval_stem = pred_stem.intersection(pos_stem.union(neg_stem))

    tp_exact = len(pred_eval_exact.intersection(pos_exact))
    fn_exact = len(pos_exact - pred_eval_exact)
    fp_exact = len(pred_eval_exact.intersection(neg_exact))
    tn_exact = len(neg_exact - pred_eval_exact)

    tp_stem = len(pred_eval_stem.intersection(pos_stem))
    fn_stem = len(pos_stem - pred_eval_stem)
    fp_stem = len(pred_eval_stem.intersection(neg_stem))
    tn_stem = len(neg_stem - pred_eval_stem)

    m_exact = compute_binary_metrics(tp_exact, fp_exact, fn_exact, tn_exact)
    m_stem = compute_binary_metrics(tp_stem, fp_stem, fn_stem, tn_stem)

    exact_row = {
        "Evaluation_Set": label,
        "Match_Type": "Exact",
        "TP": tp_exact,
        "FN": fn_exact,
        "FP": fp_exact,
        "TN": tn_exact,
        "Positive_Set_Size": len(pos_exact),
        "Negative_Set_Size": len(neg_exact),
        **m_exact
    }

    stem_row = {
        "Evaluation_Set": label,
        "Match_Type": "Stem",
        "TP": tp_stem,
        "FN": fn_stem,
        "FP": fp_stem,
        "TN": tn_stem,
        "Positive_Set_Size": len(pos_stem),
        "Negative_Set_Size": len(neg_stem),
        **m_stem
    }

    return exact_row, stem_row

low_exact_row, low_stem_row = evaluate_binary(
    pred_edges_exact, pred_edges_stem,
    positive_edges_exact, positive_edges_stem,
    negative_low_exact, negative_low_stem,
    "Low-confidence negatives"
)

high_exact_row, high_stem_row = evaluate_binary(
    pred_edges_exact, pred_edges_stem,
    positive_edges_exact, positive_edges_stem,
    negative_high_exact, negative_high_stem,
    "High-confidence negatives"
)

eval_table = pd.DataFrame([
    low_exact_row,
    low_stem_row,
    high_exact_row,
    high_stem_row
])

eval_table.to_csv(f"{output_dir}/A549_binary_evaluation_metrics.csv", index=False)

pd.DataFrame(sorted(list(positive_edges_exact)), columns=["miRNA_exact", "Gene_norm"]).to_csv(
    f"{output_dir}/A549_positive_set_exact.csv", index=False
)
pd.DataFrame(sorted(list(negative_low_exact)), columns=["miRNA_exact", "Gene_norm"]).to_csv(
    f"{output_dir}/A549_negative_set_low_confidence_exact.csv", index=False
)
pd.DataFrame(sorted(list(negative_high_exact)), columns=["miRNA_exact", "Gene_norm"]).to_csv(
    f"{output_dir}/A549_negative_set_high_confidence_exact.csv", index=False
)

# 11. Local PDCD4 analysis
group_sum = df.groupby("Gene_norm")["Final_miR_CellMap_Score"].transform("sum")
df["Dominance_Share"] = df["Final_miR_CellMap_Score"] / (group_sum + 1e-9)

df["Rank_per_Gene"] = df.groupby("Gene_norm")["Final_miR_CellMap_Score"].rank(
    ascending=False, method="first"
)

df_refined = df[df["Rank_per_Gene"] <= 3].copy()
df_refined.to_csv(f"{output_dir}/refined_competition_network.csv", index=False)

mir21_targets = set(df_refined[df_refined["miRNA_exact"] == "HSA-MIR-21-5P"]["Gene_norm"])
mir125_targets = set(df_refined[df_refined["miRNA_exact"] == "HSA-MIR-125B-5P"]["Gene_norm"])
shared_targets = sorted(list(mir21_targets.intersection(mir125_targets)))

# 12. Load PDCD4 truth for local figure benchmark
pdcd4_clip = robust_read_pdcd4_table(pdcd4_clash_file)

pdcd4_mir_col = "miRNAname" if "miRNAname" in pdcd4_clip.columns else "miRNAid"
pdcd4_gene_col = "geneName" if "geneName" in pdcd4_clip.columns else "geneSymbol"

pdcd4_clip["miRNA_exact"] = pdcd4_clip[pdcd4_mir_col].map(normalize_mirna_exact)
pdcd4_clip["Gene_norm"] = pdcd4_clip[pdcd4_gene_col].map(normalize_gene)
pdcd4_clip = pdcd4_clip[["miRNA_exact", "Gene_norm"]].drop_duplicates()

rate_base = pdcd4_overlap_precision(df, pdcd4_clip, target="PDCD4")
rate_ref = pdcd4_overlap_precision(df_refined, pdcd4_clip, target="PDCD4")
precision_gain_pdcd4 = rate_ref - rate_base

# 13. Save text summary
summary_lines = [
    "A549 main evaluation",
    f"A549 verified interactions loaded: {len(a549_df)}",
    f"A549 verified interactions in BN universe: {len(a549_eval_df)}",
    f"A549 genes in evaluation set: {len(a549_genes)}",
    f"BN miRNAs available for evaluation: {len(candidate_mirnas_exact)}",
    f"Low-confidence negatives (exact): {len(negative_low_exact)}",
    f"High-confidence negatives (exact): {len(negative_high_exact)}",
    "",
    "Metrics saved in: A549_binary_evaluation_metrics.csv",
    "",
    "Local PDCD4 evaluation",
    f"PDCD4 overlap precision-like baseline: {rate_base:.2f}%",
    f"PDCD4 overlap precision-like refined: {rate_ref:.2f}%",
    f"PDCD4 precision gain: {precision_gain_pdcd4:.2f}%",
    f"Shared targets (miR-21-5p & miR-125b-5p): {len(shared_targets)}"
]

with open(f"{output_dir}/network_summary_stats.txt", "w") as f:
    f.write("\n".join(summary_lines))

# 14. Figure 5A: PDCD4 dominance profile
example_gene = "PDCD4"
example_gene_norm = example_gene.strip().upper()

plot_data_full = df[df["Gene_norm"] == example_gene_norm].copy().sort_values(
    "Dominance_Share", ascending=False
).reset_index(drop=True)

if plot_data_full.empty:
    raise ValueError(f"No interactions found for {example_gene_norm}")

plot_data_full = plot_data_full.head(10).copy().reset_index(drop=True)

plot_data_full["Color"] = COLOR_OTHER
if len(plot_data_full) >= 1:
    plot_data_full.loc[0, "Color"] = COLOR_TOP1
if len(plot_data_full) >= 2:
    plot_data_full.loc[1, "Color"] = COLOR_TOP2
if len(plot_data_full) >= 3:
    plot_data_full.loc[2, "Color"] = COLOR_TOP3

fig_width = 8.8
fig_height = max(6.5, 0.34 * len(plot_data_full))
plt.figure(figsize=(fig_width, fig_height))

ax = plt.gca()
bars = ax.barh(
    y=np.arange(len(plot_data_full)),
    width=plot_data_full["Dominance_Share"].values,
    color=plot_data_full["Color"].values,
    edgecolor="black",
    linewidth=0.4
)

ax.set_yticks(np.arange(len(plot_data_full)))
ax.set_yticklabels(plot_data_full["miRNA"].tolist(), fontsize=7)
ax.invert_yaxis()
ax.set_xlabel("Dominance Share", fontsize=10)
ax.set_ylabel("Top 10 PDCD4 Regulators", fontsize=10)
ax.set_title("Competitive Dominance Profile: PDCD4", fontsize=11, pad=8)

max_val = plot_data_full["Dominance_Share"].max()
ax.set_xlim(0, max_val * 1.30)

for i, (_, row) in enumerate(plot_data_full.iterrows()):
    width = row["Dominance_Share"]
    label_text = f"{width * 100:.2f}%"

    if i < 3:
        font_wt = "bold"
        font_sz = 7.5
    else:
        font_wt = "normal"
        font_sz = 6.8

    ax.text(
        width + max_val * 0.015,
        i,
        label_text,
        va="center",
        ha="left",
        fontsize=font_sz,
        fontweight=font_wt
    )

legend_text = (
    "Highlighted regulators:\n"
    f"1. {plot_data_full.loc[0, 'miRNA']}" +
    (f"\n2. {plot_data_full.loc[1, 'miRNA']}" if len(plot_data_full) > 1 else "") +
    (f"\n3. {plot_data_full.loc[2, 'miRNA']}" if len(plot_data_full) > 2 else "")
)

ax.text(
    0.985, 0.08,
    legend_text,
    transform=ax.transAxes,
    ha="right", va="bottom",
    fontsize=7,
    bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="gray", alpha=0.95)
)

plt.tight_layout()
plt.savefig(f"{output_dir}/Figure5A_PDCD4_Dominance_FullProfile.png", dpi=300, bbox_inches="tight")
plt.savefig(f"{output_dir}/Figure5A_PDCD4_Dominance_FullProfile.pdf", bbox_inches="tight")
plt.close()

# 15. Figure 5B: Detailed cooperative module
plt.figure(figsize=(5.4, 5.8))
ax = plt.gca()
ax.axis("off")

x_mir1 = 0.12
x_mir2 = 0.88
y_mir = 0.84

x_center = 0.50
y_gene_start = 0.54
y_step = 0.11

ax.set_xlim(0.02, 0.98)
ax.set_ylim(-0.12, 1.02)

ax.scatter(x_mir1, y_mir, s=420, color=COLOR_BASE, edgecolor='black', linewidth=0.7, zorder=3)
ax.text(
    x_mir1, y_mir + 0.075,
    'miR-21-5p',
    ha='center',
    va='bottom',
    fontsize=8,
    fontweight='bold'
)

ax.scatter(x_mir2, y_mir, s=420, color=COLOR_POINT, edgecolor='black', linewidth=0.7, zorder=3)
ax.text(
    x_mir2, y_mir + 0.075,
    'miR-125b-5p',
    ha='center',
    va='bottom',
    fontsize=8,
    fontweight='bold'
)

if len(shared_targets) > 0:
    ax.text(
        0.50, 0.69,
        f"Shared targets: {len(shared_targets)}",
        ha='center',
        fontsize=9,
        fontweight='bold',
        color=COLOR_REF
    )

    for i, gene_label in enumerate(shared_targets):
        y_gene = y_gene_start - i * y_step

        ax.scatter(
            x_center, y_gene,
            s=220,
            color='#2c3e50',
            edgecolor='black',
            linewidth=0.6,
            zorder=3
        )

        ax.text(
            x_center,
            y_gene - 0.050,
            gene_label,
            ha='center',
            va='top',
            fontsize=7,
            fontweight='bold'
        )

        ax.plot(
            [x_mir1, x_center],
            [y_mir - 0.02, y_gene + 0.02],
            color='#34495e',
            linewidth=1.0,
            alpha=0.85
        )
        ax.plot(
            [x_mir2, x_center],
            [y_mir - 0.02, y_gene + 0.02],
            color='#34495e',
            linewidth=1.0,
            alpha=0.85
        )

else:
    ax.text(
        0.5, 0.5,
        "No Cooperative Targets Detected",
        ha='center', va='center',
        fontsize=8, fontweight='bold'
    )

plt.suptitle("Cooperative Regulatory Module", y=0.98, fontsize=10, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
finalize("Figure5B_Cooperativity_Detailed")

# 16. Figure 5C: PDCD4 local overlap benchmark
val_df = pd.DataFrame({
    "Method": ["Baseline", "Refined"],
    "Overlap": [rate_base, rate_ref]
})

plt.figure(figsize=(2.8, 3.0))
sns.barplot(
    data=val_df,
    x="Method",
    y="Overlap",
    hue="Method",
    palette=[COLOR_BASE, COLOR_REF],
    legend=False,
    edgecolor="black",
    linewidth=0.5
)

plt.plot([0, 0, 1, 1],
         [rate_ref + 2, rate_ref + 4, rate_ref + 4, rate_ref + 2],
         lw=0.8, c='#34495e')

plt.text(
    0.5,
    rate_ref + 6,
    f"+{precision_gain_pdcd4:.2f}%",
    ha='center',
    color=COLOR_REF,
    fontweight='bold',
    fontsize=7
)

plt.ylabel("PDCD4 CLIP Overlap (%)")
plt.ylim(0, 100)
plt.title("PDCD4 Local Precision Benchmark", loc='left', fontweight='bold', fontsize=8)
finalize("Figure5C_PDCD4_LocalBenchmark")

# 17. Print summary
print("Run complete. Evaluation metrics and PDCD4 figures saved in phase3_aligned_data/")
