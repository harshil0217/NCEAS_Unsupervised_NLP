import os
import warnings
import numpy as np
import pandas as pd

from collections import defaultdict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import adjusted_mutual_info_score

from evaluations.arxiv.diffusion_condensation import DiffusionCondensation
from phase4_DC_tree import build_dc_tree, lca_f1_dc, tree_edit_distance_dc

warnings.filterwarnings("ignore")

print("Starting Phase 3 (DC FINAL + SAMPLING)")

# ========================
# CPU CONTROL
# ========================
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# ========================
# LOAD GROUND TRUTH
# ========================
gt_df = pd.read_csv("hercules_outputs/cluster_assignments_arxiv_direct.csv")

LABELS_DICT = {
    "L1": LabelEncoder().fit_transform(gt_df["L1_cluster_id"].values),
    "L2": LabelEncoder().fit_transform(gt_df["L2_cluster_id"].values)
}

# ========================
# CLUSTER PURITY
# ========================
def cluster_purity(true_labels, pred_labels):
    mask = pred_labels != -1
    true_labels = true_labels[mask]
    pred_labels = pred_labels[mask]

    total = len(true_labels)
    purity = 0

    for c in np.unique(pred_labels):
        idx = pred_labels == c
        counts = np.bincount(true_labels[idx])
        purity += np.max(counts)

    return purity / total

# ========================
# DENDROGRAM PURITY
# ========================
def dendrogram_purity_dc(parent, true_labels):
    children = defaultdict(list)

    for child, par in parent.items():
        children[par].append(child)

    def get_leaves(node):
        if node not in children:
            return [node]

        leaves = []
        for c in children[node]:
            leaves += get_leaves(c)

        return leaves

    def compute(node):
        if node not in children:
            return 0, 0

        leaves = get_leaves(node)

        # 🔥 FIX
        leaves = [l for l in leaves if l < len(true_labels)]

        if len(leaves) == 0:
            return 0, 0

        labels = true_labels[leaves]

        counts = np.bincount(labels)
        purity = np.max(counts)

        total_score = purity
        total_count = len(leaves)

        for c in children[node]:
            s, t = compute(c)
            total_score += s
            total_count += t

        return total_score, total_count

    root = max(parent.values())
    score, total = compute(root)

    return score / total if total > 0 else 0

# ========================
# SETTINGS
# ========================
SAMPLE_SIZE = 10000   # 🔥 SAME IDEA AS AGGLO

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

            path = f"{emb_folder}/{reduction}_arxiv_embed.npy"

            if not os.path.exists(path):
                print("Missing:", path)
                continue

            embeddings = np.load(path)
            embeddings = StandardScaler().fit_transform(embeddings)

            true_labels = TRUE_LABELS

            # ========================
            # 🔥 SAMPLING (CONSISTENT)
            # ========================
            rng = np.random.default_rng(42)
            idx = rng.choice(len(embeddings), SAMPLE_SIZE, replace=False)

            embeddings = embeddings[idx]
            true_labels = true_labels[idx]

            n_clusters = len(np.unique(true_labels))

            model = DiffusionCondensation(
                k=5,                 # 🔥 smaller graph
                min_clusters=n_clusters,
                max_iterations=10
            )

            model.fit(embeddings)

            pred_labels = model.labels_
            history = model.history_

            parent = build_dc_tree(history)

            # ========================
            # METRICS
            # ========================
            ami = adjusted_mutual_info_score(true_labels, pred_labels)
            purity = cluster_purity(true_labels, pred_labels)
            tree_purity = dendrogram_purity_dc(parent, true_labels)
            lca = lca_f1_dc(parent, true_labels, max_pairs=50000)
            ted = tree_edit_distance_dc(parent, true_labels, max_pairs=50000)

            print(
                f"AMI: {ami:.4f} | Purity: {purity:.4f} | "
                f"DendroPurity: {tree_purity:.4f} | LCA: {lca:.4f} | TED: {ted:.4f}"
            )

            results.append({
                "level": level_name,
                "model": model_name,
                "method": "DC",
                "reduction": reduction,
                "AMI": ami,
                "Purity": purity,
                "Dendrogram_Purity": tree_purity,
                "LCA_F1": lca,
                "Tree_Edit_Distance": ted,
                "Num_Clusters": n_clusters
            })

# ========================
# SAVE
# ========================
df = pd.DataFrame(results)
df.to_csv("phase3_DC_results.csv", index=False)

print("DONE DC")