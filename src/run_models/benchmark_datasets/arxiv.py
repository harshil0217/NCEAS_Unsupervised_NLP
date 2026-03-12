import os
import sys
from tkinter.ttk import Frame
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
torch.cuda.empty_cache()
import torch.nn.functional as F
from torch.nn import DataParallel
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

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

arx = pd.read_csv("src/data/arxiv/arxiv_raw_30k.csv")
arx["topic"] = arx["text"]
arx["category_0"] = arx["label"].apply(lambda x: x.split(".")[0])
arx["category_1"] = arx["label"].apply(lambda x: x.split(".")[1])

arx = arx.dropna().reset_index(drop=True)

def get_embeddings(texts, model_id):
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
    
    model = SentenceTransformer(
        "Qwen/Qwen3-Embedding-0.6B",
        model_kwargs={"attn_implementation": "sdpa", "device_map": "auto"},
        tokenizer_kwargs={"padding_side": "left"}
    )
    

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
shuffle_idx = np.random.RandomState(seed=67).permutation(len(arx))
topic_data = arx.iloc[shuffle_idx].reset_index(drop=True)
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
    
    reduction_dir = f"{embedding_model}_reduced_embeddings"
    os.makedirs(reduction_dir, exist_ok=True)


    embedding_list = []

    if os.path.exists(f"{embedding_model}_embeddings/db_embed.npy"):
        print(f"Loading existing embeddings")
        embedding_list = np.load(f"{embedding_model}_embeddings/db_embed.npy")
    else:
        embedding_list = get_embeddings(arx['topic'], model_id=embedding_model)
        np.save(f"{embedding_model}_embeddings/arx_embed.npy", embedding_list)


    os.makedirs(f'{embedding_model}_reduced_embeddings', exist_ok=True)

    # Shuffle embeddings with the same index as topic_data
    data = np.array(embedding_list)[shuffle_idx]

    #reducer_model = phate.PHATE(n_jobs=-2, random_state=67, n_components=300, decay=20, t="auto", n_pca=None)
    #embed_phate = reducer_model.fit_transform(data)
    #np.save(f"{embedding_model}_reduced_embeddings/PHATE_db_embed.npy", embed_phate)

    #embed_phate = np.load(f"{embedding_model}_reduced_embeddings/PHATE_db_embed.npy")

    # Load your embeddings
    embeddings = np.array(data)
    embedding_methods_for_model = {}  # Local dict for this embedding model

    reduction_tasks = {
    "PHATE": {
        "path": f"{reduction_dir}/PHATE_arx_embed.npy",
        "run": lambda: phate.PHATE(n_jobs=-2, random_state=67, n_components=300, decay=20, t="auto", n_pca=None).fit_transform(embeddings)
    },
    "PCA": {
        "path": f"{reduction_dir}/PCA_arx_embed.npy",
        "run": lambda: cuPCA(n_components=300).fit_transform(embeddings)
    },
    "UMAP": {
        "path": f"{reduction_dir}/UMAP_arx_embed_new.npy",
        "run": lambda: cuUMAP(n_components=300, min_dist=.05, n_neighbors=10).fit_transform(embeddings)
    },
    "tSNE": {
        "path": f"{reduction_dir}/tSNE_arx_embed.npy",
        "run": lambda: cuTSNE(n_components=2).fit_transform(embeddings)
    },
    "PaCMAP": {
        "path": f"{reduction_dir}/PaCMAP_arx_embed.npy",
        "run": lambda: pacmap.PaCMAP(n_components=300, random_state=67).fit_transform(embeddings)
    },
}

for method_name, task in reduction_tasks.items():
    if os.path.exists(task["path"]):
        print(f"Loading cached {method_name} from {task['path']}...")
        result = np.load(task["path"])
    else:
        print(f"Running {method_name}...")
        result = task["run"]()
        
        # Handle cuML GPU to CPU conversion if necessary
        if hasattr(result, 'to_output'):
            result = result.to_output('numpy')
        elif not isinstance(result, np.ndarray):
            result = np.array(result)
            
        np.save(task["path"], result)
        print(f"Saved {method_name} to {task['path']}")
    
    embedding_methods_for_model[method_name] = result

# Store the final dict for the global embedding_models
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
                # 1. Path setup: Organizing by Embedding Model and Reduction Method
                # Example: Qwen3-Embedding_results/labels/DC_PHATE_level_142.npy
                label_dir = f"{embedding_model}_results/arxiv/labels"
                os.makedirs(label_dir, exist_ok=True)
                label_path = f"{label_dir}/{embed_name}_level_{level}.npy"

                # 2. Check if this specific level has already been computed
                if os.path.exists(label_path):
                    print(f"Loading cached DC labels from {label_path}")
                    labels = np.load(label_path)
                else:
                    print(f"Running CPU Diffusion Condensation for {embed_name} (Target Level: {level})")
                    
                    # Initialize and fit the model
                    model = dc(min_clusters=level, max_iterations=5000, k=10, alpha=3)
                    model.fit(embed_data)
                    
                    # 3. Direct assignment (since DC outputs NumPy arrays)
                    labels = model.labels_
                    
                    # 4. Save to disk for future runs
                    np.save(label_path, labels)
                    print(f"DC complete. Saved {len(np.unique(labels))} unique labels to {label_path}")


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
scores_df.to_csv(f"results/arxiv_clustering_scores.csv", index=False)

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