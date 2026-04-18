# arxiv_minilm_benchmark.py
# HPCC-ready (flat directory version)

import os
import numpy as np
import pandas as pd
import umap
import phate

from collections import defaultdict
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, rand_score
from scipy.cluster.hierarchy import fcluster
from hdbscan import HDBSCAN

# Local imports (files must be in SAME directory)
from diffusion_condensation import DiffusionCondensation as dc
from fowlkes_mallows import FowlkesMallows


# Configuration

dataset_name = "arxiv"
embedding_model = "all-MiniLM-L6-v2"

data_path = "arxiv_30k_clean.csv"
embedding_path = "arxiv_minilm_embeddings.npy"


# Safety Checks

if not os.path.exists(data_path):
    raise FileNotFoundError(f"{data_path} not found")

if not os.path.exists(embedding_path):
    raise FileNotFoundError(f"{embedding_path} not found")


# Create output folders

os.makedirs(f"{embedding_model}_reduced_embeddings", exist_ok=True)
os.makedirs(f"{embedding_model}_results", exist_ok=True)


# Load dataset

print("Loading dataset...")
df = pd.read_csv(data_path)

df_new = pd.DataFrame()
df_new["topic"] = df["text"]
df_new["category_1"] = df["label"]
df_new["category_0"] = df["label"].apply(lambda x: x.split(".")[0])

df_new = df_new.dropna().reset_index(drop=True)

print("Dataset shape:", df_new.shape)

# Load embeddings

print("Loading embeddings...")
embedding_list = np.load(embedding_path)

print("Embedding shape:", embedding_list.shape)

if len(df_new) != len(embedding_list):
    raise ValueError("Mismatch between dataset and embeddings length")


# Shuffle consistently


print("Shuffling data...")
shuffle_idx = np.random.RandomState(seed=42).permutation(len(df_new))

topic_data = df_new.iloc[shuffle_idx].reset_index(drop=True)
data = embedding_list[shuffle_idx]


# Build topic dictionary

topic_dict = {}
for col in topic_data.columns:
    if col.startswith("category_"):
        unique_count = len(topic_data[col].unique())
        topic_dict[unique_count] = np.array(topic_data[col])

# Cluster levels

depth = 2
cluster_levels = [
    len(topic_data[f"category_{i}"].unique())
    for i in reversed(range(depth))
]

print("Cluster levels:", cluster_levels)


# Dimensionality Reduction

embeddings = np.array(data)
embedding_methods = {}

# PHATE 
print("\nRunning PHATE...")
phate_model = phate.PHATE(
    n_jobs=-1,
    random_state=42,
    n_components=50
)

embed_phate = phate_model.fit_transform(embeddings)
embedding_methods["PHATE"] = embed_phate

np.save(
    f"{embedding_model}_reduced_embeddings/PHATE_{dataset_name}_embed.npy",
    embed_phate
)

# PCA 
print("Running PCA...")
pca = PCA(n_components=50, random_state=42)
embed_pca = pca.fit_transform(embeddings)
embedding_methods["PCA"] = embed_pca

np.save(
    f"{embedding_model}_reduced_embeddings/PCA_{dataset_name}_embed.npy",
    embed_pca
)

# UMAP 
print("Running UMAP...")
umap_model = umap.UMAP(
    n_components=50,
    random_state=42,
    min_dist=0.05,
    n_neighbors=10
)

embed_umap = umap_model.fit_transform(embeddings)
embedding_methods["UMAP"] = embed_umap

np.save(
    f"{embedding_model}_reduced_embeddings/UMAP_{dataset_name}_embed.npy",
    embed_umap
)


# Clustering + Metrics

print("\nRunning clustering...")

scores_all = defaultdict(lambda: defaultdict(list))

for embed_name, embed_data in tqdm(embedding_methods.items()):

    print(f"\nProcessing {embed_name}")

    for cluster_method in ["Agglomerative", "HDBSCAN", "DC"]:

        print(f"  {cluster_method}")

        for level in cluster_levels:

            # Clustering 
            if cluster_method == "Agglomerative":
                model = AgglomerativeClustering(n_clusters=level)
                labels = model.fit_predict(embed_data)

            elif cluster_method == "HDBSCAN":
                model = HDBSCAN(min_cluster_size=level)
                model.fit(embed_data)
                Z = model.single_linkage_tree_.to_numpy()
                labels = fcluster(Z, level, criterion="maxclust")
                labels[labels == -1] = labels.max() + 1

            elif cluster_method == "DC":
                model = dc(min_clusters=level, max_iterations=5000)
                model.fit(embed_data)
                labels = model.labels_

            # Align labels safely 
            available_levels = np.array(sorted(topic_dict.keys()))
            closest_level = min(available_levels, key=lambda k: abs(k - level))

            target = topic_dict[closest_level]

            valid_idx = ~pd.isna(target)
            target = target[valid_idx]
            labels = labels[valid_idx]

            # Metrics 
            try:
                fm_score = FowlkesMallows.Bk(
                    {level: target},
                    {level: labels}
                )[level]["FM"]
            except Exception:
                fm_score = np.nan

            scores_all[(embed_name, cluster_method)]["FM"].append(fm_score)
            scores_all[(embed_name, cluster_method)]["Rand"].append(
                rand_score(target, labels)
            )
            scores_all[(embed_name, cluster_method)]["ARI"].append(
                adjusted_rand_score(target, labels)
            )

# Save Results

print("\nSaving results...")

rows = []

for (embed_name, cluster_method), score_dict in scores_all.items():
    for i, level in enumerate(cluster_levels):
        rows.append({
            "reduction_method": embed_name,
            "cluster_method": cluster_method,
            "level": level,
            "FM": score_dict["FM"][i],
            "Rand": score_dict["Rand"][i],
            "ARI": score_dict["ARI"][i],
        })

scores_df = pd.DataFrame(rows)

output_path = f"{embedding_model}_results/other_{dataset_name}_results.csv"

scores_df.to_csv(output_path, index=False)

print("\n====================================")
print("MiniLM Benchmark complete.")
print("Results saved to:", output_path)
print("====================================")