# ========================
# Phase 3: HDBSCAN (FINAL CORRECT)
# ========================

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import adjusted_mutual_info_score

import hdbscan

warnings.filterwarnings("ignore")

print("======================================")
print("Starting Phase 3 (HDBSCAN CORRECT)")
print("======================================")

# ========================
# CPU CONTROL
# ========================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

# ========================
# PATHS
# ========================
BASE_DIR = os.getcwd()
gt_path = os.path.join(BASE_DIR, "hercules_outputs", "cluster_assignments_arxiv_direct.csv")

# ========================
# LOAD GROUND TRUTH
# ========================
gt_df = pd.read_csv(gt_path)

LABELS_DICT = {
    "L1": LabelEncoder().fit_transform(gt_df["L1_cluster_id"].values),
    "L2": LabelEncoder().fit_transform(gt_df["L2_cluster_id"].values)
}

# ========================
# CLUSTER PURITY (with noise removed)
# ========================
def cluster_purity(true_labels, pred_labels):
    mask = pred_labels != -1   # 🔥 remove noise

    true_labels = true_labels[mask]
    pred_labels = pred_labels[mask]

    if len(true_labels) == 0:
        return np.nan

    total = len(true_labels)
    purity = 0

    for c in np.unique(pred_labels):
        idx = pred_labels == c
        counts = np.bincount(true_labels[idx])
        purity += np.max(counts)

    return purity / total


# ========================
# MODEL PATHS
# ========================
models = {
    "MiniLM": "all-MiniLM-L6-v2_reduced_embeddings",
    "Qwen": "Qwen3-Embedding-0"
}

reductions = ["PCA", "PHATE", "UMAP"]

results = []

# ========================
# MAIN LOOP
# ========================
for level_name, TRUE_LABELS in LABELS_DICT.items():

    print(f"\nLEVEL: {level_name}")

    for model_name, emb_folder in models.items():

        for reduction in reductions:

            print(f"\nProcessing: {model_name} | {reduction}")

            emb_path = os.path.join(BASE_DIR, emb_folder, f"{reduction}_arxiv_embed.npy")

            if not os.path.exists(emb_path):
                print("Missing:", emb_path)
                continue

            embeddings = np.load(emb_path)
            embeddings = StandardScaler().fit_transform(embeddings)

            true_labels = TRUE_LABELS

            if len(true_labels) != len(embeddings):
                print("SIZE MISMATCH")
                continue

            # ========================
            # RUN HDBSCAN
            # ========================
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=20,
                core_dist_n_jobs=4
            )

            pred_labels = clusterer.fit_predict(embeddings)

            # ========================
            # REMOVE NOISE (-1)
            # ========================
            mask = pred_labels != -1

            filtered_true = true_labels[mask]
            filtered_pred = pred_labels[mask]

            if len(filtered_true) == 0:
                print("All points labeled as noise")
                continue

            # ========================
            # METRICS (ONLY THESE)
            # ========================
            ami = adjusted_mutual_info_score(filtered_true, filtered_pred)
            purity = cluster_purity(true_labels, pred_labels)

            print(f"AMI: {ami:.6f} | Purity: {purity:.4f}")

            results.append({
                "level": level_name,
                "model": model_name,
                "method": "HDBSCAN",
                "reduction": reduction,
                "AMI": ami,
                "Purity": purity,
                "LCA_F1": None,   # ✅ REQUIRED
                "Tree_Edit_Distance": None  # ✅ REQUIRED
            })

# ========================
# SAVE RESULTS
# ========================
df = pd.DataFrame(results)

output_file = os.path.join(BASE_DIR, "phase3_HDBSCAN_results.csv")
df.to_csv(output_file, index=False)

print("\n======================================")
print("DONE → HDBSCAN CORRECT")
print("Saved to:", output_file)
print("======================================")