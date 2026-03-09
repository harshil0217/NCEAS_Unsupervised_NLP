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
import matplotlib.pyplot as plt

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
import pacmap
import trimap

# cuML GPU-accelerated dimensionality reduction
import cuml
from cuml.decomposition import PCA as cuPCA
from cuml.manifold import TSNE as cuTSNE
from cuml.manifold import UMAP as cuUMAP

# ========================
# Clustering
# ========================
from custom_packages.diffusion_condensation import DiffusionCondensation as dc

# cuML GPU-accelerated clustering
from cuml.cluster import AgglomerativeClustering as cuAgglomerativeClustering
from cuml.cluster import HDBSCAN as cuHDBSCAN


# ======================
# Evaluation Metrics
# ======================
from custom_packages.fowlkes_mallows import FowlkesMallows
from sklearn.metrics import adjusted_rand_score, rand_score, adjusted_mutual_info_score


from tqdm import tqdm
# ==============
# Global Config
# ==============
np.random.seed(67)
warnings.filterwarnings("ignore")

# Reload modules if needed
importlib.reload(phate)




rcv1 = pd.read_csv('data/rcv1/rcv1_data.csv')

rcv1 = rcv1.drop_duplicates(subset='topic', keep=False).reset_index(drop=True)
rcv1 = rcv1.drop_duplicates(subset='item_id', keep=False).reset_index(drop=True)

rcv1 = rcv1.rename(columns={'Title': 'topic'})
rcv1 = rcv1.rename(columns={'Cat1': 'category_0'})
rcv1 = rcv1.rename(columns={'Cat2': 'category_1'})
rcv1 = rcv1.rename(columns={'Cat3': 'category_2'})



rcv1= rcv1.dropna().reset_index(drop=True)  # remove NaNs
rcv1 = rcv1[rcv1['topic'].apply(lambda x: isinstance(x, str) and x.strip() != '')].reset_index(drop=True) 

rcv1.to_csv("data/rcv1/rcv1_data.csv")

rcv1.shape

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
    print(len(texts))
    
    model = SentenceTransformer(model, device=device)

    tok = model.tokenizer(texts.tolist(), truncation=False, padding=False)

    lens = [len(x) for x in tok['input_ids']]

    print(f"Total tokens: {sum(lens):,}")
    print(f"Avg tokens: {sum(lens)/len(lens):.1f}")
    print(f"Max tokens: {max(lens)}")
    
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
shuffle_idx = np.random.RandomState(seed=67).permutation(len(rcv1))
topic_data = rcv1.iloc[shuffle_idx].reset_index(drop=True)
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
    embedding_list = get_embeddings(rcv1['topic'], model=embedding_model)

    np.save(f"{embedding_model}_embeddings/rcv1_embed.npy", embedding_list)

    embedding_list = np.load(f"{embedding_model}_embeddings/rcv1_embed.npy")

    os.makedirs(f'{embedding_model}_reduced_embeddings', exist_ok=True)

    # Shuffle embeddings with the same index as topic_data
    data = np.array(embedding_list)[shuffle_idx]

    reducer_model = phate.PHATE(n_jobs=-2, random_state=67, n_components=300, decay=20, t="auto", n_pca=None)
    embed_phate = reducer_model.fit_transform(data)
    np.save(f"{embedding_model}_reduced_embeddings/PHATE_rcv1_embed.npy", embed_phate)

    embed_phate = np.load(f"{embedding_model}_reduced_embeddings/PHATE_rcv1_embed.npy")

    # Load your embeddings
    embeddings = np.array(data)
    embedding_methods_for_model = {}  # Local dict for this embedding model

    embedding_methods_for_model["PHATE"] = embed_phate

    # Apply other dimensionality reduction methods
    include_pca = True
    include_umap = True

    # PCA using cuML (GPU-accelerated)
    if include_pca:
        print("Running PCA with cuML (GPU)...")
        pca_model = cuPCA(n_components=300)
        pca_result = pca_model.fit_transform(embeddings)
        # Convert from GPU to numpy
        embedding_methods_for_model["PCA"] = pca_result.to_output('numpy') if hasattr(pca_result, 'to_output') else np.array(pca_result)
        np.save(f"{embedding_model}_reduced_embeddings/PCA_rcv1_embed.npy", embedding_methods_for_model["PCA"])

    # UMAP using cuML (GPU-accelerated)
    if include_umap:
        print("Running UMAP with cuML (GPU)...")
        umap_model = cuUMAP(n_components=300, min_dist=.05, n_neighbors=10)
        umap_result = umap_model.fit_transform(embeddings)
        # Convert from GPU to numpy
        embedding_methods_for_model["UMAP"] = umap_result.to_output('numpy') if hasattr(umap_result, 'to_output') else np.array(umap_result)
        np.save(f"{embedding_model}_reduced_embeddings/UMAP_rcv1_embed.npy", embedding_methods_for_model["UMAP"])

    # t-SNE using cuML (GPU-accelerated)
    print("Running t-SNE with cuML (GPU)...")
    tsne_model = cuTSNE(n_components=2)
    tsne_result = tsne_model.fit_transform(embeddings)
    # Convert from GPU to numpy
    embedding_methods_for_model["tSNE"] = tsne_result.to_output('numpy') if hasattr(tsne_result, 'to_output') else np.array(tsne_result)
    np.save(f"{embedding_model}_reduced_embeddings/tSNE_rcv1_embed.npy", embedding_methods_for_model["tSNE"])

    # # Fit to PaCMAP
    pac = pacmap.PaCMAP(n_components=300, random_state=67)
    embedding_methods_for_model["PaCMAP"] = pac.fit_transform(embeddings)
    np.save(f"{embedding_model}_reduced_embeddings/PaCMAP_rcv1_embed.npy", embedding_methods_for_model["PaCMAP"])

    # # Fit to TriMAP
    tr = trimap.TRIMAP(n_dims=300)
    embedding_methods_for_model["TriMAP"] = tr.fit_transform(embeddings)
    np.save(f"{embedding_model}_reduced_embeddings/TriMAP_rcv1_embed.npy", embedding_methods_for_model["TriMAP"])

    # Store the embedding methods for this model
    embedding_models[embedding_model] = embedding_methods_for_model

    
scores_all = defaultdict(lambda: defaultdict(list))


def make_noise_labels_unique(labels):
    labels = np.asarray(labels).copy()
    noise_mask = labels == -1
    if not np.any(noise_mask):
        return labels

    if labels.size == 0:
        return labels

    next_label = int(np.max(labels)) + 1
    if next_label <= -1:
        next_label = 0

    noise_indices = np.where(noise_mask)[0]
    labels[noise_indices] = np.arange(next_label, next_label + noise_indices.size)
    return labels


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
                # Use cuML GPU-accelerated Agglomerative Clustering
                print("Using cuML Agglomerative Clustering (GPU)...")
                model = cuAgglomerativeClustering(n_clusters=level)
                model.fit(embed_data)
                # Convert GPU result to numpy
                labels = model.labels_.to_output('numpy') if hasattr(model.labels_, 'to_output') else np.array(model.labels_)
                print(f"Agglomerative clustering complete. Unique labels: {len(np.unique(labels))}")

            elif cluster_method == "HDBSCAN":
                # Use cuML GPU-accelerated HDBSCAN
                print("Using cuML HDBSCAN (GPU)...")
                model = cuHDBSCAN(min_cluster_size=level, min_samples=1)
                model.fit(embed_data)
                # Convert GPU result to numpy
                labels = model.labels_.to_output('numpy') if hasattr(model.labels_, 'to_output') else np.array(model.labels_)

                # cuML HDBSCAN returns cluster labels directly, no need for single_linkage_tree
                # If all points are noise (-1), assign unique labels
                if np.all(labels == -1):
                    print("WARNING: All points labeled as noise. Assigning unique labels.")
                    labels = np.arange(len(labels))

                print(f"HDBSCAN clustering complete. Unique labels: {len(np.unique(labels))}")

            elif cluster_method == "DC":
                # Diffusion Condensation uses CPU implementation
                print("Using CPU Diffusion Condensation...")
                model = dc(min_clusters=level, max_iterations=5000, k=10, alpha=3)
                model.fit(embed_data)
                labels = model.labels_
                print(f"DC clustering complete. Unique labels: {len(np.unique(labels))}")


            available_levels = np.array(sorted(topic_dict.keys()))
            closest_level = min(available_levels, key=lambda k: abs(k - level))
            print(f"Ground truth: Using closest level {closest_level} (requested: {level})")

            topic_series = topic_dict[closest_level]
            valid_idx = (~pd.isna(topic_series))
            target_lst = topic_series[valid_idx]
            label_lst = labels[valid_idx]

            if cluster_method == "HDBSCAN":
                label_lst = make_noise_labels_unique(label_lst)

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

# Run each combo sequentially (no parallel processing)
combo_results = []
for embedding_model, embed_name, cluster_method in tqdm(combo_params, desc="Processing embedding-clustering combos"):
    result = safe_run_combo(embedding_model, embed_name, cluster_method)
    combo_results.append(result)

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
os.makedirs("results", exist_ok=True)
scores_df.to_csv(f"results/rcv1_clustering_scores.csv", index=False)

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

