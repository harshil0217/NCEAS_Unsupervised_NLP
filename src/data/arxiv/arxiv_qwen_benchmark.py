import os
import re
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
import phate
import umap

from tqdm import tqdm
from scipy.cluster.hierarchy import fcluster
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, rand_score
from hdbscan import HDBSCAN

from diffusion_condensation import DiffusionCondensation as dc
from fowlkes_mallows import FowlkesMallows


warnings.filterwarnings("ignore")
np.random.seed(42)

dataset_name = "arxiv"
embedding_model = "Qwen3-Embedding-0.6B"

os.makedirs(f"{embedding_model}_reduced_embeddings", exist_ok=True)
os.makedirs(f"{embedding_model}_results", exist_ok=True)


# Load Dataset
print("Loading dataset...")
df = pd.read_csv("arxiv_30k_clean.csv")

df_new = pd.DataFrame()
df_new["topic"] = df["text"]
df_new["category_1"] = df["label"]
df_new["category_0"] = df["label"].apply(lambda x: x.split(".")[0])

df_new = df_new.dropna().reset_index(drop=True)
df_new = df_new[
    df_new["topic"].apply(lambda x: isinstance(x, str) and x.strip() != "")
].reset_index(drop=True)


# Load Embeddings
print("Loading embeddings...")
embedding_list = np.load("arxiv_qwen_embeddings.npy")
print("Embedding shape:", embedding_list.shape)



# Shuffle

shuffle_idx = np.random.RandomState(seed=42).permutation(len(df_new))
topic_data = df_new.iloc[shuffle_idx].reset_index(drop=True)
data = embedding_list[shuffle_idx]

print("Shuffled data shape:", data.shape)


# Build Ground Truth Dictionary

topic_dict = {}

for col in topic_data.columns:
    if re.match(r'^category_\d+$', col):
        unique_count = len(topic_data[col].unique())
        topic_dict[unique_count] = np.array(topic_data[col])

print("Hierarchy levels:", sorted(topic_dict.keys()))



# PHATE

print("Running PHATE...")

reducer_model = phate.PHATE(
    n_jobs=-2,
    random_state=42,
    n_components=300,
    decay=20,
    t="auto",
    n_pca=None
)

embed_phate = reducer_model.fit_transform(data)

np.save(
    f"{embedding_model}_reduced_embeddings/PHATE_{dataset_name}_embed.npy",
    embed_phate
)



# Build Hierarchy Levels
depth = 2
cluster_levels = []

for i in reversed(range(depth)):
    cluster_levels.append(len(topic_data[f'category_{i}'].unique()))

print("Cluster levels used:", cluster_levels)



# Dimensionality Reduction

embedding_methods = {}
embedding_methods["PHATE"] = embed_phate

print("Running PCA...")
pca = PCA(n_components=50, random_state=42)
embed_pca = pca.fit_transform(data)

embedding_methods["PCA"] = embed_pca

np.save(
    f"{embedding_model}_reduced_embeddings/PCA_{dataset_name}_embed.npy",
    embed_pca
)


print("Running UMAP...")
umap_model = umap.UMAP(
    n_components=50,
    random_state=42,
    min_dist=0.05,
    n_neighbors=10
)

embed_umap = umap_model.fit_transform(data)

embedding_methods["UMAP"] = embed_umap

np.save(
    f"{embedding_model}_reduced_embeddings/UMAP_{dataset_name}_embed.npy",
    embed_umap
)



# Clustering + Metrics

print("Running clustering...")

scores_all = defaultdict(lambda: defaultdict(list))

for embed_name, embed_data in tqdm(embedding_methods.items()):
    print(f"\nProcessing {embed_name} embeddings...")

    for cluster_method in ["Agglomerative", "HDBSCAN", "DC"]:
        print(f"  Clustering method: {cluster_method}")

        for level in cluster_levels:

            
            # Clustering
            
            if cluster_method == "Agglomerative":
                model = AgglomerativeClustering(n_clusters=level)
                labels = model.fit_predict(embed_data)

            elif cluster_method == "HDBSCAN":
                model = HDBSCAN(min_cluster_size=level)
                model.fit(embed_data)

                Z = model.single_linkage_tree_.to_numpy()
                labels = fcluster(Z, level, criterion='maxclust')
                labels[labels == -1] = labels.max() + 1

            elif cluster_method == "DC":
                model = dc(
                    min_clusters=level,
                    max_iterations=5000,
                    k=10,
                    alpha=3
                )
                model.fit(embed_data)
                labels = model.labels_

            
            # Match Ground Truth
            available_levels = np.array(sorted(topic_dict.keys()))
            closest_level = min(available_levels, key=lambda k: abs(k - level))

            topic_series = topic_dict[closest_level]
            valid_idx = ~pd.isna(topic_series)

            target_lst = topic_series[valid_idx]
            label_lst = labels[valid_idx]
            # Metrics
            try:
                fm_score = FowlkesMallows.Bk(
                    {level: target_lst},
                    {level: label_lst}
                )[level]["FM"]
            except:
                fm_score = np.nan

            scores_all[(embed_name, cluster_method)]["FM"].append(fm_score)
            scores_all[(embed_name, cluster_method)]["Rand"].append(
                rand_score(target_lst, label_lst)
            )
            scores_all[(embed_name, cluster_method)]["ARI"].append(
                adjusted_rand_score(target_lst, label_lst)
            )



# Save Results
print("Saving results...")

rows = []

for (embed_name, cluster_method), score_dict in scores_all.items():
    for i in range(len(score_dict["FM"])):
        rows.append({
            "reduction_method": embed_name,
            "cluster_method": cluster_method,
            "level": cluster_levels[i],
            "FM": score_dict["FM"][i],
            "Rand": score_dict["Rand"][i],
            "ARI": score_dict["ARI"][i],
        })

scores_df = pd.DataFrame(rows)

scores_df = scores_df.sort_values(
    by=["reduction_method", "cluster_method", "level"]
).reset_index(drop=True)

output_file = f"{embedding_model}_results/other_{dataset_name}_results.csv"

scores_df.to_csv(output_file, index=False)

print("Done.")
print("Results saved to:", output_file)


