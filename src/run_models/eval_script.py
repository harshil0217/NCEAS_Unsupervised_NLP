# ========================
# Environment
# ========================
import os
import warnings
import numpy as np
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")

# ========================
# Clustering + Metrics
# ========================
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    adjusted_rand_score,
    rand_score,
    adjusted_mutual_info_score
)
from hdbscan import HDBSCAN

# ========================
# Custom clustering
# ========================
from diffusion_condensation import DiffusionCondensation as dc
from fowlkes_mallows import FowlkesMallows

# ========================
# Paths
# ========================
mini_dir = "all-MiniLM-L6-v2_reduced_embeddings"
qwen_dir = "Qwen3-Embedding-0.6B_reduced_embeddings"

results_dir = "evaluation_results"
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
# Embedding folders
# ========================
embeddings = {
    "MiniLM": mini_dir,
    "Qwen": qwen_dir
}

# ========================
# Dimensionality reductions
# ========================
reductions = ["PCA", "PHATE", "UMAP"]

# ========================
# Clustering methods
# ========================
cluster_methods = [
    "Agglomerative",
    "HDBSCAN",
    "DiffusionCondensation"
]

# ========================
# Load embedding
# ========================
def load_embedding(folder, reduction, dataset):

    path = f"{folder}/{reduction}_{dataset}_embed.npy"

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing embedding file: {path}")

    return np.load(path)


# ========================
# Clustering
# ========================
def run_clustering(embed, cluster_name, true_cluster_count):

    if cluster_name == "Agglomerative":

        model = AgglomerativeClustering(
            n_clusters=true_cluster_count,
            linkage="ward"
        )
        labels_pred = model.fit_predict(embed)

    elif cluster_name == "HDBSCAN":

        model = HDBSCAN(
            min_cluster_size=15,
            cluster_selection_method="eom"
        )
        labels_pred = model.fit_predict(embed)

    elif cluster_name == "DiffusionCondensation":

        model = dc()
        model.fit(embed)

        if hasattr(model, "labels_"):
            labels_pred = model.labels_
        elif hasattr(model, "labels"):
            labels_pred = model.labels
        elif hasattr(model, "cluster_labels"):
            labels_pred = model.cluster_labels
        else:
            raise RuntimeError("DiffusionCondensation does not expose cluster labels")

    else:
        raise ValueError("Unknown cluster method")

    return labels_pred


# ========================
# Evaluation (FIXED)
# ========================
def evaluate(labels_true, labels_pred):

    labels_true = np.array(labels_true)
    labels_pred = np.array(labels_pred)

    # 🔥 HDBSCAN fix: remove noise (-1)
    valid_mask = labels_pred != -1

    if valid_mask.sum() == 0:
        return np.nan, np.nan, np.nan, np.nan

    labels_true = labels_true[valid_mask]
    labels_pred = labels_pred[valid_mask]

    true_cluster_count = len(np.unique(labels_true))

    fm = FowlkesMallows.Bk(
        {true_cluster_count: labels_true},
        {true_cluster_count: labels_pred}
    )

    rand = rand_score(labels_true, labels_pred)
    ari = adjusted_rand_score(labels_true, labels_pred)
    ami = adjusted_mutual_info_score(labels_true, labels_pred)  # ✅ NEW

    return fm[true_cluster_count]["FM"], rand, ari, ami


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

    labels_true = df["label"].values
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

            for cluster in cluster_methods:

                print(f"Running {cluster}")

                try:
                    labels_pred = run_clustering(
                        embed,
                        cluster,
                        true_cluster_count
                    )

                    fm, rand, ari, ami = evaluate(labels_true, labels_pred)

                    results.append({
                        "dataset": dataset_name,
                        "embedding": embed_name,
                        "reduction": reduction,
                        "cluster_method": cluster,
                        "FM": fm,
                        "Rand": rand,
                        "ARI": ari,
                        "AMI": ami   # ✅ NEW
                    })

                except Exception as e:
                    print("Error:", e)

    # ========================
    # Save results
    # ========================
    if len(results) > 0:

        results_df = pd.DataFrame(results)

        output_file = f"{results_dir}/{dataset_name}_comparison.csv"

        results_df.to_csv(output_file, index=False)

        print("\nSaved results to:", output_file)

    else:
        print("No results produced for this dataset.")

print("\nAll datasets finished.")