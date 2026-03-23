# ========================
# Environment
# ========================
import os
import warnings
import numpy as np
import pandas as pd

from scipy.cluster.hierarchy import linkage, to_tree, fcluster
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import adjusted_mutual_info_score

warnings.filterwarnings("ignore")

print("======================================")
print("Starting Phase 3 (Agglomerative FINAL + ALL METRICS)")
print("======================================")

# ========================
# PATHS
# ========================
BASE_DIR = os.getcwd()

gt_path = os.path.join(BASE_DIR, "hercules_outputs", "cluster_assignments_arxiv_direct.csv")

# ========================
# LOAD GROUND TRUTH
# ========================
print("\nLoading ground truth...")

gt_df = pd.read_csv(gt_path)

LABELS_DICT = {
    "L1": LabelEncoder().fit_transform(gt_df["L1_cluster_id"].values),
    "L2": LabelEncoder().fit_transform(gt_df["L2_cluster_id"].values)
}

print("L1 clusters:", len(np.unique(LABELS_DICT["L1"])))
print("L2 clusters:", len(np.unique(LABELS_DICT["L2"])))

# ========================
# DENDROGRAM PURITY
# ========================
def dendrogram_purity(Z, true_labels):
    tree, _ = to_tree(Z, rd=True)

    def get_leaves(node):
        if node.is_leaf():
            return [node.id]
        return get_leaves(node.left) + get_leaves(node.right)

    def compute(node):
        if node.is_leaf():
            return 0, 0

        leaves = get_leaves(node)
        labels = true_labels[leaves]

        counts = np.bincount(labels)
        purity = np.max(counts)

        l_score, l_total = compute(node.left)
        r_score, r_total = compute(node.right)

        return purity + l_score + r_score, len(leaves) + l_total + r_total

    score, total = compute(tree)
    return score / total

# ========================
# TREE STRUCTURE HELPERS
# ========================
def build_parent_map(Z, n_samples):
    parent = {}
    current_id = n_samples

    for left, right, _, _ in Z:
        parent[int(left)] = current_id
        parent[int(right)] = current_id
        current_id += 1

    return parent


def get_ancestors(node, parent_map):
    ancestors = set()
    while node in parent_map:
        node = parent_map[node]
        ancestors.add(node)
    return ancestors


def compute_lca(node1, node2, parent_map):
    ancestors1 = get_ancestors(node1, parent_map)
    while node2 in parent_map:
        node2 = parent_map[node2]
        if node2 in ancestors1:
            return node2
    return None

# ========================
# LCA-F1 (50k sampling)
# ========================
def lca_f1_score_sampled(Z, true_labels, max_pairs=50000):
    n = len(true_labels)
    parent_map = build_parent_map(Z, n)

    rng = np.random.default_rng(42)

    pairs = 0
    correct = 0

    for _ in range(max_pairs):
        i, j = rng.choice(n, size=2, replace=False)

        same_true = (true_labels[i] == true_labels[j])
        lca = compute_lca(i, j, parent_map)
        same_pred = lca is not None

        if same_true == same_pred:
            correct += 1

        pairs += 1

    return correct / pairs

# ========================
# TREE EDIT DISTANCE (APPROX)
# ========================
def tree_edit_distance_sampled(Z, true_labels, max_pairs=50000):
    n = len(true_labels)
    parent_map = build_parent_map(Z, n)

    rng = np.random.default_rng(42)

    mismatches = 0
    pairs = 0

    for _ in range(max_pairs):
        i, j = rng.choice(n, size=2, replace=False)

        same_true = (true_labels[i] == true_labels[j])
        lca = compute_lca(i, j, parent_map)
        same_pred = lca is not None

        if same_true != same_pred:
            mismatches += 1

        pairs += 1

    return mismatches / pairs  # lower = better

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

    print(f"\n==============================")
    print(f"LEVEL: {level_name}")
    print(f"==============================")

    for model_name, emb_folder in models.items():

        for reduction in reductions:

            print(f"\nProcessing: {model_name} | {reduction} | {level_name}")

            emb_path = os.path.join(BASE_DIR, emb_folder, f"{reduction}_arxiv_embed.npy")

            if not os.path.exists(emb_path):
                print(f"❌ Missing file: {emb_path}")
                continue

            print("Loading:", emb_path)

            embeddings = np.load(emb_path)
            print("Embeddings shape:", embeddings.shape)

            # Scale
            embeddings = StandardScaler().fit_transform(embeddings)

            true_labels = TRUE_LABELS

            if len(true_labels) != len(embeddings):
                print("❌ SIZE MISMATCH — STOP")
                continue

            n_clusters = len(np.unique(true_labels))
            print("Ground truth clusters:", n_clusters)

            # ------------------------
            # Build tree
            # ------------------------
            print("Building linkage tree...")
            Z = linkage(embeddings, method="ward")

            # ------------------------
            # Purity
            # ------------------------
            print("Computing dendrogram purity...")
            purity = dendrogram_purity(Z, true_labels)
            print(f"Purity: {purity:.4f}")

            # ------------------------
            # AMI
            # ------------------------
            print("Computing AMI...")
            pred_labels = fcluster(Z, t=n_clusters, criterion="maxclust")
            ami = adjusted_mutual_info_score(true_labels, pred_labels)
            print(f"AMI: {ami:.6f}")

            # ------------------------
            # LCA-F1
            # ------------------------
            print("Computing LCA-F1 (50k)...")
            lca_f1 = lca_f1_score_sampled(Z, true_labels, max_pairs=50000)
            print(f"LCA-F1: {lca_f1:.4f}")

            # ------------------------
            # Tree Edit Distance
            # ------------------------
            print("Computing Tree Edit Distance...")
            ted = tree_edit_distance_sampled(Z, true_labels, max_pairs=50000)
            print(f"Tree Edit Distance: {ted:.4f}")

            # ------------------------
            # Save
            # ------------------------
            results.append({
                "level": level_name,
                "model": model_name,
                "method": "Agglomerative",
                "reduction": reduction,
                "Dendrogram_Purity": purity,
                "AMI": ami,
                "LCA_F1": lca_f1,
                "Tree_Edit_Distance": ted,
                "Num_Clusters": n_clusters
            })

# ========================
# SAVE RESULTS
# ========================
print("\nSaving results...")

df = pd.DataFrame(results)
output_file = os.path.join(BASE_DIR, "phase3_agglomerative_ALL_METRICS.csv")

df.to_csv(output_file, index=False)

print("======================================")
print("✅ DONE — Results saved to:")
print(output_file)
print("======================================")