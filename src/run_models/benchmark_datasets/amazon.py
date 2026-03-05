import os
import sys
from dotenv import load_dotenv
import json
load_dotenv() 

# Set the target folder name you want to reach
target_folder = "src"

# Get the current working directory
current_dir = os.getcwd()

# Loop to move up the directory tree until we reach the target folder
while os.path.basename(current_dir) != target_folder:
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    if parent_dir == current_dir:
        # If we reach the root directory and haven't found the target, exit
        raise FileNotFoundError(f"{target_folder} not found in the directory tree.")
    current_dir = parent_dir

# Change the working directory to the folder where "phate-for-text" is found
os.chdir(current_dir)

# Add the "phate-for-text" directory to sys.path
sys.path.insert(0, current_dir)

# ===================
# Standard Libraries
# ===================
import importlib
import os
import re
from pathlib import Path
import warnings
from collections import defaultdict
import torch

# ===================
# Data Manipulation
# ===================
import numpy as np
import pandas as pd

# ====================
# Embeddings
# ====================
from sentence_transformers import SentenceTransformer
device = "cuda" if torch.cuda.is_available() else "cpu"



# ==========================
# Dimensionality Reduction
# ==========================
import phate
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pacmap
import trimap

# ========================
# Clustering
# ========================
from hdbscan import HDBSCAN
from scipy.cluster.hierarchy import fcluster
from sklearn.cluster import AgglomerativeClustering
from custom_packages.diffusion_condensation import DiffusionCondensation as dc
from custom_packages.hercules import Hercules


# ======================
# Evaluation Metrics
# ======================
from custom_packages.fowlkes_mallows import FowlkesMallows
from sklearn.metrics import adjusted_rand_score, rand_score, adjusted_mutual_info_score


from tqdm import tqdm
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib
# ==============
# Global Config
# ==============
np.random.seed(67)
warnings.filterwarnings("ignore")

# Reload modules if needed
importlib.reload(phate)




amz_40 = pd.read_csv("data/amazon/train_40k.csv")
amz_10 = pd.read_csv("data/amazon/val_10k.csv")

amz=pd.concat([amz_40,amz_10])
amz = amz.drop_duplicates(subset='Title', keep=False).reset_index(drop=True)
amz = amz.drop_duplicates(subset='productId', keep=False).reset_index(drop=True)

amz = amz.rename(columns={'Title': 'topic'})
amz = amz.rename(columns={'Cat1': 'category_0'})
amz = amz.rename(columns={'Cat2': 'category_1'})
amz = amz.rename(columns={'Cat3': 'category_2'})



amz= amz.dropna().reset_index(drop=True)  # remove NaNs
amz = amz[amz['topic'].apply(lambda x: isinstance(x, str) and x.strip() != '')].reset_index(drop=True) 

amz.to_csv("data/amazon/amz_data.csv")

amz.shape

def get_embeddings(texts, model):
    """
    Fetches embeddings using the specified backend: 'gpt' (OpenAI) or 'sentence-transformers'.
    
    Args:
        texts (list of str): List of text inputs.
        backend (str): 'gpt' or 'sentence-transformers'.
        model (str): Model name for the chosen backend.
        
    Returns:
        list: List of embeddings.
    """
    print("Using device:", device)
    
    model = SentenceTransformer(model, device=device)
    
    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    
    return embeddings


embedding_model_names = [
    "Qwen/Qwen3-Embedding-0.6B",
    "sentence-transformers/all-MiniLM-L6-v2",
]

embedding_models = {}  # Will store {model_name: {reduction_method: embeddings}}

# Prepare data once (same for all embedding models)
shuffle_idx = np.random.RandomState(seed=67).permutation(len(amz))
topic_data = amz.iloc[shuffle_idx].reset_index(drop=True)
reverse_idx = np.argsort(shuffle_idx)

# Build topic_dict from ground truth categories
topic_dict = {}
for col in topic_data.columns:
    if re.match(r'^category_\d+$', col):
        unique_count = len(topic_data[col].unique())
        topic_dict[unique_count] = np.array(topic_data[col])

# Determine cluster levels from hierarchy depth
depth = 3
print(f"Depth: {depth}")
print(f"Building cluster levels by counting unique categories at each level...\n")

cluster_levels = []
for i in reversed(range(0, depth)):
    unique_count = len(topic_data[f'category_{i}'].unique())
    print(f"Level {i} (category_{i}): {unique_count} unique categories")
    cluster_levels.append(unique_count)

print(f"\nFinal cluster_levels (from deepest to shallowest): {cluster_levels}\n")

# Now process each embedding model
for embedding_model in embedding_model_names:
    print(f"\n{'='*60}")
    print(f"Processing embedding model: {embedding_model}")
    print(f"{'='*60}\n")

    os.makedirs(f'{embedding_model}_results', exist_ok=True)
    os.makedirs(f"{embedding_model}_embeddings", exist_ok=True)
    embedding_list = get_embeddings(amz['topic'], model=embedding_model)

    np.save(f"{embedding_model}_embeddings/amz_embed.npy", embedding_list)

    embedding_list = np.load(f"{embedding_model}_embeddings/amz_embed.npy")

    os.makedirs(f'{embedding_model}_reduced_embeddings', exist_ok=True)

    # Shuffle embeddings with the same index as topic_data
    data = np.array(embedding_list)[shuffle_idx]

    reducer_model = phate.PHATE(n_jobs=-2, random_state=67, n_components=300, decay=20, t="auto", n_pca=None)
    embed_phate = reducer_model.fit_transform(data)
    np.save(f"{embedding_model}_reduced_embeddings/PHATE_amz_embed.npy", embed_phate)

    embed_phate = np.load(f"{embedding_model}_reduced_embeddings/PHATE_amz_embed.npy")

    import umap
    import matplotlib.pyplot as plt

    # Load your embeddings
    embeddings = np.array(data)
    embedding_methods_for_model = {}  # Local dict for this embedding model

    embedding_methods_for_model["PHATE"] = embed_phate

    # Apply other dimensionality reduction methods
    include_pca = True
    include_umap = True

    if include_pca:
        pca_model = PCA(n_components=300, random_state=67)
        embedding_methods_for_model["PCA"] = pca_model.fit_transform(embeddings)
        np.save(f"{embedding_model}_reduced_embeddings/PCA_amz_embed.npy", embedding_methods_for_model["PCA"])

    # # UMAP to 2D
    if include_umap:
        umap_model = umap.UMAP(n_components=300, random_state=67, min_dist=.05, n_neighbors=10)
        embedding_methods_for_model["UMAP"] = umap_model.fit_transform(embeddings)
        np.save(f"{embedding_model}_reduced_embeddings/UMAP_amz_embed_new.npy", embedding_methods_for_model["UMAP"])

    from sklearn.manifold import TSNE

    # # Fit t-SNE
    tsne_model = TSNE(n_components=3, random_state=67)
    embedding_methods_for_model["tSNE"] = tsne_model.fit_transform(embeddings)
    np.save(f"{embedding_model}_reduced_embeddings/tSNE_amz_embed.npy", embedding_methods_for_model["tSNE"])

    # # Fit to PaCMAP
    pac = pacmap.PaCMAP(n_components=300, random_state=67, n_neighbors=10, MN_ratio=0.5, FP_ratio=2.0)
    embedding_methods_for_model["PaCMAP"] = pac.fit_transform(embeddings)
    np.save(f"{embedding_model}_reduced_embeddings/PaCMAP_amz_embed.npy", embedding_methods_for_model["PaCMAP"])

    # # Fit to TriMAP
    tr = trimap.TRIMAP(n_dims=300, random_state=67, n_neighbors=10, min_dist=0.05)
    embedding_methods_for_model["TriMAP"] = tr.fit_transform(embeddings)
    np.save(f"{embedding_model}_reduced_embeddings/TriMAP_amz_embed.npy", embedding_methods_for_model["TriMAP"])

    # Store the embedding methods for this model
    embedding_models[embedding_model] = embedding_methods_for_model

    
scores_all = defaultdict(lambda: defaultdict(list))


def safe_run_combo(embedding_model, embed_name, cluster_method):
    embed_data = embedding_models[embedding_model][embed_name]
    combo_scores = {"FM": [], "Rand": [], "ARI": [], "AMI": []}
    try:
        print(f"\n{'='*60}")
        print(f"Processing Embedding Method: {embed_name}")
        print(f"Clustering Method: {cluster_method}")
        print(f"Embedding shape: {embed_data.shape}")
        print(f"{'='*60}")

        for level in cluster_levels:
            print(f"Testing cluster level: {level}")

            if cluster_method == "Agglomerative":
                model = AgglomerativeClustering(n_clusters=level)
                model.fit(embed_data)
                labels = model.labels_
                print(f"Agglomerative clustering complete. Unique labels: {len(np.unique(labels))}")

            elif cluster_method == "HDBSCAN":
                model = HDBSCAN(min_cluster_size=level)
                model.fit(embed_data)
                Z = model.single_linkage_tree_.to_numpy()
                labels = fcluster(Z, level, criterion='maxclust')
                labels[labels == -1] = None
                print(f"HDBSCAN clustering complete. Unique labels: {len(np.unique(labels))}")

            elif cluster_method == "DC":
                model = dc(min_clusters=level, max_iterations=5000, k=10, alpha=3)
                model.fit(embed_data)
                labels = model.labels_
                print(f"DC clustering complete. Unique labels: {len(np.unique(labels))}")

            
            available_levels = np.array(sorted(topic_dict.keys()))
            closest_level = min(available_levels, key=lambda k: abs(k - level))
            print(f"Ground truth: Using closest level {closest_level} (requested: {level})")

            topic_series = topic_dict[closest_level]
            valid_idx = ~pd.isna(topic_series)
            target_lst = topic_series[valid_idx]
            label_lst = labels[valid_idx]

            try:
                fm_score = FowlkesMallows.Bk({level: target_lst}, {level: label_lst})[level]['FM']
            except Exception:
                fm_score = np.nan
                print("WARNING: FM score computation failed!")

            rand = rand_score(target_lst, label_lst)
            ari = adjusted_rand_score(target_lst, label_lst)
            ami = adjusted_mutual_info_score(target_lst, label_lst)
            print(f"Scores - FM: {fm_score:.4f}, Rand: {rand:.4f}, ARI: {ari:.4f}")

            combo_scores["FM"].append(fm_score)
            combo_scores["Rand"].append(rand)
            combo_scores["ARI"].append(ari)
            combo_scores["AMI"].append(ami)

        return embedding_model, embed_name, cluster_method, combo_scores
    except Exception as e:
        print(f"Error in combo ({embedding_model}, {embed_name}, {cluster_method}): {e}")
        return embedding_model, embed_name, cluster_method, combo_scores

combo_params = [
    (embedding_model, embed_name, cluster_method)
    for embedding_model in embedding_models.keys()
    for embed_name in embedding_models[embedding_model].keys()
    for cluster_method in ["Agglomerative", "HDBSCAN", "DC"]
]

with tqdm_joblib(tqdm(desc="Processing embedding-clustering combos", total=len(combo_params))):
    with Parallel(n_jobs=-4, backend="threading") as parallel:
        combo_results = parallel(
            delayed(safe_run_combo)(embedding_model, embed_name, cluster_method)
            for embedding_model, embed_name, cluster_method in combo_params
        )

for embedding_model, embed_name, cluster_method, combo_scores in combo_results:
    scores_all[(embedding_model, embed_name, cluster_method)]["FM"] = combo_scores["FM"]
    scores_all[(embedding_model, embed_name, cluster_method)]["Rand"] = combo_scores["Rand"]
    scores_all[(embedding_model, embed_name, cluster_method)]["ARI"] = combo_scores["ARI"]
    scores_all[(embedding_model, embed_name, cluster_method)]["AMI"] = combo_scores["AMI"]

print(f"\n{'='*60}")
print("All clustering and evaluation complete!")
print(f"{'='*60}")



rows = []

for (embedding_model, embed_name, cluster_method), score_dict in scores_all.items():
    n_levels = len(score_dict["FM"])  # assuming all score lists have same length
    for i in range(n_levels):
            rows.append({
                "embedding_model": embedding_model,
                "reduction_method": embed_name,
                "cluster_method": cluster_method,
                "level": cluster_levels[i],  # assumes scores were appended in order
                "FM": score_dict["FM"][i],
                "Rand": score_dict["Rand"][i],
                "ARI": score_dict["ARI"][i],
                "AMI": score_dict["AMI"][i],
            })


# Create DataFrame
scores_df = pd.DataFrame(rows)


# Optional: sort for easier viewing
scores_df = scores_df.sort_values(by=["embedding_model", "reduction_method", "cluster_method", "level"]).reset_index(drop=True)
scores_df.to_csv(f"results/amazon_clustering_scores.csv", index=False)

combo_color_map = {
    ("PHATE", "Agglomerative"): "tab:blue",
    ("PHATE", "HDBSCAN"): "deepskyblue",
    ("PHATE", "DC"): "navy",
    ("PCA", "Agglomerative"): "tab:orange",
    ("PCA", "HDBSCAN"): "gold",
    ("PCA", "DC"): "chocolate",
    ("UMAP", "Agglomerative"): "tab:green",
    ("UMAP", "HDBSCAN"): "limegreen",
    ("UMAP", "DC"): "darkgreen",
    ("tSNE", "Agglomerative"): "tab:red",
    ("tSNE", "HDBSCAN"): "lightcoral",
    ("tSNE", "DC"): "darkred",
    ("PaCMAP", "Agglomerative"): "tab:purple",
    ("PaCMAP", "HDBSCAN"): "mediumorchid",
    ("PaCMAP", "DC"): "indigo",
    ("TriMAP", "Agglomerative"): "tab:brown",
    ("TriMAP", "HDBSCAN"): "sandybrown",
    ("TriMAP", "DC"): "maroon",
}

# Plot per embedding model
metrics = ['FM', 'Rand', 'ARI', 'AMI']

for embedding_model, df in scores_df.groupby("embedding_model"):
    os.makedirs(f"{embedding_model}_results", exist_ok=True)

    for metric in metrics:
        plt.figure(figsize=(10, 6))

        for (embed_name, cluster_method), group_df in df.groupby(["reduction_method", "cluster_method"]):
            plot_df = group_df.sort_values("level")
            method_label = "Diffusion Condensation" if cluster_method == "DC" else cluster_method

            plt.plot(
                plot_df["level"],
                plot_df[metric],
                marker='o',
                label=f"{embed_name} {method_label}",
                color=combo_color_map.get((embed_name, cluster_method), 'black')
            )

        plt.title(f"{metric} Score Across Cluster Levels ({embedding_model})")
        plt.xlabel("Cluster Level")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{embedding_model}_results/{metric}_scores_plot.png")
        plt.close()

