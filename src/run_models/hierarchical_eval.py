# ========================
# Environment
# ========================g
import os
import warnings
import numpy as np
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# ========================
# Hierarchical tools
# ========================
from scipy.cluster.hierarchy import linkage, fcluster

# ========================
# Paths
# ========================
mini_dir = "all-MiniLM-L6-v2_reduced_embeddings"
qwen_dir = "Qwen3-Embedding-0.6B_reduced_embeddings"

results_dir = "hierarchical_results"
os.makedirs(results_dir, exist_ok=True)

# ========================
# Datasets
# ========================
datasets = {
    "arxiv": "arxiv_30k_clean.csv",
    "rcv1": "rcv1_clean.csv",
    "amazon": "amazon_clean.csv",
    "wos": "wos_clean.csv"
}

# ========================
# Embeddings
# ========================
embeddings = {
    "MiniLM": mini_dir,
    "Qwen": qwen_dir
}

# ========================
# Reductions
# ========================
reductions = ["PCA", "PHATE", "UMAP"]

# ========================
# Load embedding
# ========================
def load_embedding(folder, reduction, dataset):
    path = f"{folder}/{reduction}_{dataset}_embed.npy"

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing embedding file: {path}")

    return np.load(path)


# ========================
# Dendrogram Purity (simple version)
# ========================
def dendrogram_purity(Z, labels_true, k):
    """
    Approximate dendrogram purity by cutting tree at k clusters
    """
    pred = fcluster(Z, t=k, criterion='maxclust')

    purity = 0
    total = len(labels_true)

    for cluster_id in np.unique(pred):
        mask = pred == cluster_id
        true_labels = labels_true[mask]

        if len(true_labels) == 0:
            continue

        counts = np.bincount(pd.factorize(true_labels)[0])
        purity += np.max(counts)

    return purity / total


# ========================
# Placeholder LCA-F1
# ========================
def lca_f1_placeholder(Z, labels_true):
    """
    TEMP placeholder — replace with HERCULES implementation later
    """
    return np.nan


# ========================
# Run Experiments
# ========================
for dataset_name, dataset_file in datasets.items():

    print("\n========================")
    print(f"Running dataset: {dataset_name}")
    print("========================")

    if not os.path.exists(dataset_file):
        print(f"Dataset not found: {dataset_file}, skipping...")
        continue

    df = pd.read_csv(dataset_file)

    if "label" not in df.columns:
        print("No label column found, skipping...")
        continue

    labels_true = df["label"].fillna("None").values
    true_cluster_count = len(np.unique(labels_true))

    results = []

    for embed_name, folder in embeddings.items():

        for reduction in reductions:

            print(f"\nLoading {embed_name} + {reduction}")

            try:
                embed = load_embedding(folder, reduction, dataset_name)
                print("Embedding shape:", embed.shape)

            except Exception as e:
                print("Missing embedding:", e)
                continue

            try:
                # ========================
                # Build hierarchical tree
                # ========================
                Z = linkage(embed, method="ward")

                # ========================
                # Metrics
                # ========================
                purity = dendrogram_purity(Z, labels_true, true_cluster_count)
                lca_f1 = lca_f1_placeholder(Z, labels_true)

                results.append({
                    "dataset": dataset_name,
                    "embedding": embed_name,
                    "reduction": reduction,

                    "dendrogram_purity": purity,
                    "lca_f1": lca_f1
                })

                print(f"Purity: {purity:.4f}")

            except Exception as e:
                print("Error:", e)

    # ========================
    # Save results
    # ========================
    if len(results) > 0:

        results_df = pd.DataFrame(results)

        output_file = f"{results_dir}/{dataset_name}_hierarchical.csv"
        results_df.to_csv(output_file, index=False)

        print("\nSaved results to:", output_file)

    else:
        print("No results produced for this dataset.")

print("\nAll datasets finished.")
