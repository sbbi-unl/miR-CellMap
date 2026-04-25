import os
import warnings
import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# 1. Optional Bayesian structure learning dependency
try:
    from pgmpy.estimators import HillClimbSearch

    try:
        from pgmpy.estimators import ExpertKnowledge
        HAS_EXPERT_KNOWLEDGE = True
    except ImportError:
        HAS_EXPERT_KNOWLEDGE = False

    try:
        # newer pgmpy
        from pgmpy.estimators import BDeu
        def make_bdeu_score(dataframe, equivalent_sample_size):
            return BDeu(dataframe, equivalent_sample_size=equivalent_sample_size)
    except ImportError:
        # older pgmpy
        from pgmpy.estimators import BDeuScore
        def make_bdeu_score(dataframe, equivalent_sample_size):
            return BDeuScore(dataframe, equivalent_sample_size=equivalent_sample_size)

    try:
        from pgmpy.base import DAG
    except ImportError:
        from pgmpy.models import BayesianNetwork as DAG

    PGMPY_AVAILABLE = True
except ImportError:
    PGMPY_AVAILABLE = False

# 2. Configuration
output_dir = "phase3_aligned_data"

print("Step 4: Network inference and topology optimization")

# Hybrid score weights
alpha, beta, gamma = 0.4, 0.3, 0.3

# Bayesian topology optimization parameters
TABU_LENGTH = 10

BDEU_EQUIVALENT_SAMPLE_SIZE = 1

DISCRETIZATION_BINS = 3
MIN_RETAINED_EDGES_PER_GENE = 1

# 3. Utility functions
def safe_quantile_discretize(series: pd.Series, bins: int = 3) -> pd.Series:
    x = pd.Series(series).astype(float)

    if x.nunique(dropna=True) < 2:
        return pd.Series([np.nan] * len(x), index=x.index)

    try:
        disc = pd.qcut(
            x.rank(method="average"),
            q=bins,
            labels=False,
            duplicates="drop"
        )
        return disc.astype("Int64")
    except Exception:
        return pd.Series([np.nan] * len(x), index=x.index)


def assign_prior_strength(row):
    if row.get("Is_A549_Context", 0) == 1:
        return 1.0
    elif row.get("Is_Database_Prior", 0) == 1:
        return 0.8
    else:
        return 0.5


def score_interactions(df, mirna_matrix, mrna_matrix):
    results = []

    print(f"Scoring {len(df)} regulatory interactions...")

    for _, row in df.iterrows():
        m = str(row["miRNA"])
        g = str(row["Target Gene"])

        if m not in mirna_matrix.index or g not in mrna_matrix.index:
            continue

        mir_vec = mirna_matrix.loc[m].values.astype(float)
        gene_vec = mrna_matrix.loc[g].values.astype(float)

        # Pearson correlation
        if np.std(mir_vec) == 0 or np.std(gene_vec) == 0:
            cc = 0.0
        else:
            cc, _ = pearsonr(mir_vec, gene_vec)
            if np.isnan(cc):
                cc = 0.0

        # Expression compatibility
        norm_m = np.linalg.norm(mir_vec)
        norm_g = np.linalg.norm(gene_vec)
        ec = np.dot(mir_vec, gene_vec) / (norm_m * norm_g) if (norm_m * norm_g) > 0 else 0.0

        results.append({
            "miRNA": m,
            "Target Gene": g,
            "Attention": float(row["Attention_Weight"]),
            "EC": float(ec),
            "CC": float(cc),
            "Prior": float(row["Prior_Strength"]),
            "Is_A549_Context": int(row.get("Is_A549_Context", 0)),
            "Is_Database_Prior": int(row.get("Is_Database_Prior", 0))
        })

    scored_df = pd.DataFrame(results)

    if scored_df.empty:
        raise ValueError("No interactions were successfully scored. Check matrix indices and prior/attention overlap.")

    # Hybrid score
    scored_df["Hybrid_Score"] = (
        alpha * scored_df["Attention"] +
        beta * scored_df["EC"] +
        gamma * scored_df["Prior"]
    )

    # Pre-BN competition
    scored_df["PreBN_Competition_Weight"] = scored_df.groupby("Target Gene")["Hybrid_Score"].transform(
        lambda x: x / (x.sum() + 1e-9)
    )

    scored_df["PreBN_Final_Score"] = (
        scored_df["Hybrid_Score"] * scored_df["PreBN_Competition_Weight"]
    )

    return scored_df


def build_start_dag(selected_mirnas):
    start_dag = DAG()
    start_dag.add_node("TARGET_GENE")

    for mir_id in selected_mirnas:
        start_dag.add_node(mir_id)
        start_dag.add_edge(mir_id, "TARGET_GENE")

    return start_dag


def bn_optimize_gene_subgraph(gene_df, mirna_matrix, mrna_matrix):
    gene_df = gene_df.sort_values("PreBN_Final_Score", ascending=False).copy()
    target_gene = str(gene_df["Target Gene"].iloc[0])

    candidate_df = gene_df.copy()

    if not PGMPY_AVAILABLE:
        warnings.warn(
            f"pgmpy is not installed. Using the top retained edge for gene {target_gene}.",
            RuntimeWarning
        )
        retained = candidate_df.head(MIN_RETAINED_EDGES_PER_GENE).copy()
        retained["BN_Selected"] = 1
        retained["BN_Method"] = "fallback_top_hgt"
        return retained

    if target_gene not in mrna_matrix.index:
        retained = candidate_df.head(MIN_RETAINED_EDGES_PER_GENE).copy()
        retained["BN_Selected"] = 1
        retained["BN_Method"] = "fallback_missing_gene"
        return retained

    # Build BN input table from cell-resolved expression
    bn_table = pd.DataFrame(index=mrna_matrix.columns)
    bn_table["TARGET_GENE"] = mrna_matrix.loc[target_gene].astype(float).values

    mirna_id_to_name = {}
    selected_mirnas = []

    for i, mirna_name in enumerate(candidate_df["miRNA"].astype(str).tolist()):
        if mirna_name not in mirna_matrix.index:
            continue
        node_name = f"MIR_{i:03d}"
        bn_table[node_name] = mirna_matrix.loc[mirna_name].astype(float).values
        mirna_id_to_name[node_name] = mirna_name
        selected_mirnas.append(node_name)

    if len(selected_mirnas) == 0:
        retained = candidate_df.head(MIN_RETAINED_EDGES_PER_GENE).copy()
        retained["BN_Selected"] = 1
        retained["BN_Method"] = "fallback_no_mirna_nodes"
        return retained

    # Discretize
    disc_table = pd.DataFrame(index=bn_table.index)
    for col in bn_table.columns:
        disc_table[col] = safe_quantile_discretize(bn_table[col], bins=DISCRETIZATION_BINS)

    usable_cols = [c for c in disc_table.columns if disc_table[c].notna().sum() == len(disc_table)]
    disc_table = disc_table[usable_cols].copy()

    if "TARGET_GENE" not in disc_table.columns:
        retained = candidate_df.head(MIN_RETAINED_EDGES_PER_GENE).copy()
        retained["BN_Selected"] = 1
        retained["BN_Method"] = "fallback_target_constant"
        return retained

    selected_mirnas = [c for c in selected_mirnas if c in disc_table.columns]

    if len(selected_mirnas) == 0:
        retained = candidate_df.head(MIN_RETAINED_EDGES_PER_GENE).copy()
        retained["BN_Selected"] = 1
        retained["BN_Method"] = "fallback_all_mirnas_constant"
        return retained

    # Only allow miRNA -> TARGET_GENE
    all_nodes = list(disc_table.columns)
    forbidden_edges = []

    for src in all_nodes:
        for dst in all_nodes:
            if src == dst:
                continue
            if src in selected_mirnas and dst == "TARGET_GENE":
                continue
            forbidden_edges.append((src, dst))

    score_method = make_bdeu_score(
        disc_table,
        equivalent_sample_size=BDEU_EQUIVALENT_SAMPLE_SIZE
    )
    searcher = HillClimbSearch(disc_table)

    start_dag = build_start_dag(selected_mirnas)

    try:
        if HAS_EXPERT_KNOWLEDGE:
            expert_knowledge = ExpertKnowledge(forbidden_edges=forbidden_edges)
            learned_model = searcher.estimate(
                scoring_method=score_method,
                start_dag=start_dag,
                expert_knowledge=expert_knowledge,
                tabu_length=TABU_LENGTH,
                show_progress=False
            )
        else:
            learned_model = searcher.estimate(
                scoring_method=score_method,
                start_dag=start_dag,
                black_list=forbidden_edges,
                tabu_length=TABU_LENGTH,
                show_progress=False
            )

        retained_mirna_ids = sorted(
            [src for src, dst in learned_model.edges() if dst == "TARGET_GENE"]
        )

        if len(retained_mirna_ids) == 0:
            retained = candidate_df.head(MIN_RETAINED_EDGES_PER_GENE).copy()
            retained["BN_Selected"] = 1
            retained["BN_Method"] = "fallback_no_bn_edge"
            return retained

        retained_mirnas = [mirna_id_to_name[mid] for mid in retained_mirna_ids]
        retained = candidate_df[candidate_df["miRNA"].isin(retained_mirnas)].copy()
        retained["BN_Selected"] = 1
        retained["BN_Method"] = "bayesian_structure_learning"
        return retained

    except TypeError:
        learned_model = searcher.estimate(
            scoring_method=score_method,
            start_dag=start_dag,
            black_list=forbidden_edges,
            tabu_length=TABU_LENGTH,
            show_progress=False
        )

        retained_mirna_ids = sorted(
            [src for src, dst in learned_model.edges() if dst == "TARGET_GENE"]
        )

        if len(retained_mirna_ids) == 0:
            retained = candidate_df.head(MIN_RETAINED_EDGES_PER_GENE).copy()
            retained["BN_Selected"] = 1
            retained["BN_Method"] = "fallback_no_bn_edge"
            return retained

        retained_mirnas = [mirna_id_to_name[mid] for mid in retained_mirna_ids]
        retained = candidate_df[candidate_df["miRNA"].isin(retained_mirnas)].copy()
        retained["BN_Selected"] = 1
        retained["BN_Method"] = "bayesian_structure_learning"
        return retained

    except Exception as e:
        warnings.warn(
            f"Bayesian optimization failed for gene {target_gene} with error: {e}. "
            f"Using the top retained edge.",
            RuntimeWarning
        )
        retained = candidate_df.head(MIN_RETAINED_EDGES_PER_GENE).copy()
        retained["BN_Selected"] = 1
        retained["BN_Method"] = "fallback_bn_error"
        return retained


# 4. Load inputs
attention_df = pd.read_csv(f"{output_dir}/hgt_attention_edges.csv")
mrna_matrix = pd.read_csv(f"{output_dir}/A549_aligned_mRNA.csv", index_col=0)
mirna_matrix = pd.read_csv(f"{output_dir}/A549_aligned_miRNA.csv", index_col=0)
priors = pd.read_csv(f"{output_dir}/mirna_gene_prior.csv")

metadata_file = f"{output_dir}/mirna_gene_prior_metadata.csv"
if os.path.exists(metadata_file):
    prior_meta = pd.read_csv(metadata_file)
    print(f"Loaded prior metadata: {metadata_file}")
else:
    prior_meta = priors.copy()
    print("Prior metadata file not found; using the main prior file.")

mrna_matrix.index = mrna_matrix.index.astype(str)
mirna_matrix.index = mirna_matrix.index.astype(str)

# 5. Merge attention with prior evidence
df = pd.merge(attention_df, prior_meta, on=["miRNA", "Target Gene"], how="left")

for col in ["Is_A549_Context", "Is_Database_Prior"]:
    if col not in df.columns:
        df[col] = 0
    df[col] = df[col].fillna(0).astype(int)

df["Prior_Strength"] = df.apply(assign_prior_strength, axis=1)

# 6. Score all prior-supported interactions
scored_df = score_interactions(df, mirna_matrix, mrna_matrix)

full_ranked_network = scored_df.sort_values(
    by="PreBN_Final_Score",
    ascending=False
).reset_index(drop=True)

full_ranked_network.to_csv(f"{output_dir}/final_network_inference_pre_bn.csv", index=False)

# 7. Bayesian topology optimization on miRNA-gene edges
print("\nApplying Bayesian structure learning on the miRNA-gene graph")

optimized_parts = []
gene_summaries = []

for gene_name, gene_df in scored_df.groupby("Target Gene"):
    retained_df = bn_optimize_gene_subgraph(gene_df, mirna_matrix, mrna_matrix)
    optimized_parts.append(retained_df)

    gene_summaries.append({
        "Target Gene": gene_name,
        "Candidates_Before_BN": int(len(gene_df)),
        "Retained_After_BN": int(len(retained_df)),
        "BN_Method": str(retained_df["BN_Method"].iloc[0]) if not retained_df.empty else "none"
    })

optimized_df = pd.concat(optimized_parts, ignore_index=True)

if optimized_df.empty:
    raise ValueError("Bayesian topology optimization produced an empty network.")

print(f"Full weighted prior network: {len(scored_df)} edges")
print(f"Bayesian-optimized network: {len(optimized_df)} edges")

# 8. Recompute competition after BN topology editing
optimized_df["PostBN_Competition_Weight"] = optimized_df.groupby("Target Gene")["Hybrid_Score"].transform(
    lambda x: x / (x.sum() + 1e-9)
)

optimized_df["Final_miR_CellMap_Score"] = (
    optimized_df["Hybrid_Score"] * optimized_df["PostBN_Competition_Weight"]
)

optimized_network = optimized_df.sort_values(
    by="Final_miR_CellMap_Score",
    ascending=False
).reset_index(drop=True)

# 9. Export outputs
optimized_network.to_csv(f"{output_dir}/final_network_inference_bn_optimized.csv", index=False)
optimized_network.to_csv(f"{output_dir}/final_network_inference.csv", index=False)

master_regulators = optimized_network.groupby("miRNA")["Final_miR_CellMap_Score"].sum().sort_values(ascending=False)
master_regulators.to_csv(f"{output_dir}/top_master_regulators.csv")

per_gene_counts = optimized_network.groupby("Target Gene").size().reset_index(name="Retained_Regulators")
per_gene_counts.to_csv(f"{output_dir}/bn_retained_counts_per_gene.csv", index=False)

gene_summary_df = pd.DataFrame(gene_summaries)
gene_summary_df.to_csv(f"{output_dir}/bn_gene_optimization_summary.csv", index=False)

summary = pd.DataFrame([{
    "full_ranked_edges_before_bn": len(full_ranked_network),
    "retained_edges_after_bn": len(optimized_network),
    "unique_genes_before": full_ranked_network["Target Gene"].nunique(),
    "unique_genes_after": optimized_network["Target Gene"].nunique(),
    "unique_miRNAs_before": full_ranked_network["miRNA"].nunique(),
    "unique_miRNAs_after": optimized_network["miRNA"].nunique(),
    "tabu_length": TABU_LENGTH,
    "bdeu_equivalent_sample_size": BDEU_EQUIVALENT_SAMPLE_SIZE,
    "discretization_bins": DISCRETIZATION_BINS,
    "pgmpy_available": PGMPY_AVAILABLE,
    "manual_candidate_cap_removed": True,
    "manual_parent_cap_removed": True
}])
summary.to_csv(f"{output_dir}/bn_topology_optimization_summary.csv", index=False)

# 10. Print summary
print("\nRun complete.")
print(f"Full weighted prior network: {len(full_ranked_network)} edges")
print(f"Bayesian-optimized network: {len(optimized_network)} edges")

print("\nTop 10 master regulators:")
print(master_regulators.head(10))
