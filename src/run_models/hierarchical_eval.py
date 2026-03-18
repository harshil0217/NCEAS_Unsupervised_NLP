# ========================
# Environment
# ========================
import os
import warnings
import numpy as np
import pandas as pd
import ast

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

print("Starting hierarchical evaluation...")
print("Current working directory:", os.getcwd())

# ========================
# Paths (SAFE)
# ========================
BASE_DIR = os.getcwd()

eval_dir = os.path.join(BASE_DIR, "evaluation_results")
results_dir = os.path.join(BASE_DIR, "hierarchical_results")

os.makedirs(results_dir, exist_ok=True)

print("Evaluation dir:", eval_dir)
print("Results dir:", results_dir)

# ========================
# Datasets
# ========================
datasets = ["arxiv"]

# ========================
# Dendrogram Purity
# ========================
def dendrogram_purity(true_labels, pred_labels):

    # remove HDBSCAN noise (-1)
    valid_mask = pred_labels != -1
    true_labels = true_labels[valid_mask]
    pred_labels = pred_labels[valid_mask]

    if len(true_labels) == 0:
        return np.nan

    total = len(true_labels)
    purity = 0

    for cluster in np.unique(pred_labels):
        mask = pred_labels == cluster
        labels = true_labels[mask]

        if len(labels) == 0:
            continue

        counts = pd.Series(labels).value_counts()
        purity += counts.iloc[0]

    return purity / total


# ========================
# Run Evaluation
# ========================
for dataset in datasets:

    print("\n========================")
    print(f"Running dataset: {dataset}")
    print("========================")

    eval_file = os.path.join(eval_dir, f"{dataset}_FULL.csv")
    print("Looking for file:", eval_file)

    # 🚨 check file exists
    if not os.path.exists(eval_file):
        print(f"❌ Missing file: {eval_file}")
        continue

    print("✅ File found, loading...")

    df = pd.read_csv(eval_file)
    print("Rows in file:", len(df))

    results = []

    for i, row in df.iterrows():

        try:
            # ========================
            # Safe parsing
            # ========================
            true_labels = np.array(ast.literal_eval(str(row["true_labels"])))
            pred_labels = np.array(ast.literal_eval(str(row["pred_labels"])))

            # ========================
            # Length check
            # ========================
            if len(true_labels) != len(pred_labels):
                print(f"⚠️ Row {i}: length mismatch → skipping")
                continue

            # ========================
            # Compute purity
            # ========================
            purity = dendrogram_purity(true_labels, pred_labels)

            results.append({
                "dataset": dataset,
                "embedding": row.get("embedding", "unknown"),
                "reduction": row.get("reduction", "unknown"),
                "cluster_method": row.get("cluster_method", "unknown"),
                "dendrogram_purity": purity,
                "lca_f1": np.nan
            })

            # print progress every few rows
            if i % 5 == 0:
                print(
                    f"[{i}] {row.get('embedding')} + {row.get('reduction')} + {row.get('cluster_method')} "
                    f"→ Purity: {purity:.4f}"
                )

        except Exception as e:
            print(f"❌ Error at row {i}:", e)

    # ========================
    # Save results
    # ========================
    if len(results) > 0:

        results_df = pd.DataFrame(results)

        # raw results
        output_file = os.path.join(results_dir, f"{dataset}_hierarchical.csv")
        results_df.to_csv(output_file, index=False)

        print("✅ Saved raw results:", output_file)

        # ========================
        # SUMMARY (VERY IMPORTANT)
        # ========================
        summary_df = results_df.groupby(
            ["embedding", "reduction", "cluster_method"]
        )["dendrogram_purity"].mean().reset_index()

        summary_file = os.path.join(results_dir, f"{dataset}_hierarchical_summary.csv")
        summary_df.to_csv(summary_file, index=False)

        print("✅ Saved summary:", summary_file)

    else:
        print("❌ No valid results computed.")

print("\n🎉 All datasets finished.")