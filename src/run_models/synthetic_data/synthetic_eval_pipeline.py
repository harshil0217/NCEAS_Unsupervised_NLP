# ========================
# Environment Configuration
# ========================
from dotenv import load_dotenv
load_dotenv() 
import os
import sys

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
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ========================
# Standard Libraries
# ========================
import importlib
import re
import time
import warnings

# ========================
# Data Manipulation
# ========================
import numpy as np
import pandas as pd

# ===============================
# Machine Learning & Clustering
# ===============================
from sklearn.metrics import adjusted_rand_score, rand_score, adjusted_mutual_info_score
from collections import defaultdict
import torch

# ===========================
# Dimensionality Reduction
# ===========================
import phate
import pacmap
#import trimap

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

# ========================
# NLP & Transformers
# ========================
from sentence_transformers import SentenceTransformer
device = "cuda" if torch.cuda.is_available() else "cpu"

# ========================
# Parallel Processing
# ========================
from tqdm import tqdm

# ========================
# Evaluation Metrics
# ========================
from custom_packages.fowlkes_mallows import FowlkesMallows


# ===================
# Global Config
# ===================
np.random.seed(67)
warnings.filterwarnings("ignore")
importlib.reload(phate)

# ===================
# Embedding Functions
# ===================
def get_embeddings(texts, model):
    """
    Fetches embeddings using sentence-transformers.

    Args:
        texts (list of str): List of text inputs.
        model (str): Model name for sentence-transformers.

    Returns:
        numpy.ndarray: Array of embeddings.
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

# ===================
# Utility Functions
# ===================
def make_noise_labels_unique(labels):
    """Convert HDBSCAN noise points (-1) to unique cluster labels."""
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


# ===================
# Clustering Runner
# ===================
def safe_run_combo(embedding_model, embed_name, cluster_method, embed_data, cluster_levels, topic_dict, theme, t, max_sub, depth, synonyms, branching, add_noise):
    """Run clustering on reduced embeddings and evaluate against ground truth."""
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
                # Create label path for caching
                if float(add_noise) > 0:
                    label_path = f"intermediate_data/{embedding_model}_labels/{theme}_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_noise{add_noise}_{branching}/{embed_name}/HDB_{level}_labels.npy"
                else:
                    label_path = f"intermediate_data/{embedding_model}_labels/{theme}_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_{branching}/{embed_name}/HDB_{level}_labels.npy"

                if os.path.exists(label_path):
                    print(f"Loading cached HDBSCAN labels from {label_path}")
                    labels = np.load(label_path)
                else:
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

                    # Cache the labels
                    os.makedirs(os.path.dirname(label_path), exist_ok=True)
                    np.save(label_path, labels)
                    print(f"HDBSCAN clustering complete. Unique labels: {len(np.unique(labels))}")

            elif cluster_method == "DC":
                # Create label path for caching
                if float(add_noise) > 0:
                    label_path = f"intermediate_data/{embedding_model}_labels/{theme}_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_noise{add_noise}_{branching}/{embed_name}/DC_{level}_labels.npy"
                else:
                    label_path = f"intermediate_data/{embedding_model}_labels/{theme}_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_{branching}/{embed_name}/DC_{level}_labels.npy"

                if os.path.exists(label_path):
                    print(f"Loading cached DC labels from {label_path}")
                    labels = np.load(label_path)
                else:
                    # Diffusion Condensation uses CPU implementation
                    print("Using CPU Diffusion Condensation...")
                    model = dc(min_clusters=level, max_iterations=5000, k=10, alpha=3)
                    model.fit(embed_data)
                    labels = model.labels_

                    # Cache the labels
                    os.makedirs(os.path.dirname(label_path), exist_ok=True)
                    np.save(label_path, labels)
                    print(f"DC clustering complete. Unique labels: {len(np.unique(labels))}")

            # Find closest available ground truth level
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
            print(f"Scores - FM: {fm_score:.4f}, Rand: {rand:.4f}, ARI: {ari:.4f}, AMI: {ami:.4f}")

            combo_scores["FM"].append(fm_score)
            combo_scores["Rand"].append(rand)
            combo_scores["ARI"].append(ari)
            combo_scores["AMI"].append(ami)

        return embedding_model, embed_name, cluster_method, combo_scores
    except Exception as e:
        print(f"Error in combo ({embedding_model}, {embed_name}, {cluster_method}): {e}")
        return embedding_model, embed_name, cluster_method, combo_scores

# ====================
# Setup & Execution
# ====================
def noise_range(value):
    f = float(value)
    if f < 0 or f > 1:
        raise argparse.ArgumentTypeError("add_noise must be a float between 0 and 1.")
    return f

import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--theme", type=str, required=True)
parser.add_argument("--t", type=float, required=True)
parser.add_argument("--max_sub", type=int, required=True)
parser.add_argument("--depth", type=int, required=True)
parser.add_argument("--synonyms", type=int, required=True)
parser.add_argument("--branching", type=str, required=True)
parser.add_argument("--add_noise", type=noise_range, default=0.0,
                    help="Amount of noise to add (float between 0 and 1, default: 0.0)")
parser.add_argument("--wait", type=str, choices=["True", "False"], default="False", help="Set wait to True or False")

args = parser.parse_args()

theme = args.theme
t = args.t
max_sub = args.max_sub
depth = args.depth
synonyms = args.synonyms
branching = args.branching
add_noise = args.add_noise
wait = args.wait == "True"

# Load generated data
filename = f'data_generation/generated_data/{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_noise{add_noise}_{branching}.csv'

print(f"Waiting for {filename} to be created...")
while not os.path.exists(filename):
    time.sleep(1)
print(f"{filename} detected! Reading file...")

topic_data_original = pd.read_csv(filename)

# Embedding models to use (same as amazon.py)
embedding_model_names = [
    "Qwen/Qwen3-Embedding-0.6B",
    "sentence-transformers/all-MiniLM-L6-v2",
]

embedding_models = {}  # Will store {model_name: {reduction_method: embeddings}}

# Prepare data once (same shuffle for all embedding models)
shuffle_idx = np.random.RandomState(seed=67).permutation(len(topic_data_original))
topic_data = topic_data_original.iloc[shuffle_idx].reset_index(drop=True)
reverse_idx = np.argsort(shuffle_idx)

# Build topic_dict from ground truth categories
topic_dict = {}
for col in topic_data.columns:
    if re.match(r'^category \d+$', col):
        unique_count = len(topic_data[col].unique())
        topic_dict[unique_count] = np.array(topic_data[col])

# Determine cluster levels from hierarchy depth
print(f"Depth: {depth}")
print(f"Building cluster levels by counting unique categories at each level...\n")

cluster_levels = []
for i in reversed(range(0, depth)):
    unique_count = len(topic_data[f'category {i}'].unique())
    print(f"Level {i} (category {i}): {unique_count} unique categories")
    cluster_levels.append(unique_count)

print(f"\nFinal cluster_levels (from deepest to shallowest): {cluster_levels}\n")

# Process each embedding model
for embedding_model in embedding_model_names:
    print(f"\n{'='*60}")
    print(f"Processing embedding model: {embedding_model}")
    print(f"{'='*60}\n")

    os.makedirs(f'intermediate_data/{embedding_model}_results', exist_ok=True)
    os.makedirs(f"intermediate_data/{embedding_model}_embeddings", exist_ok=True)

    # Generate or load embeddings
    if float(add_noise) > 0:
        embed_file = f'intermediate_data/{embedding_model}_embeddings/{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_noise{add_noise}_{branching}_embed.npy'
    else:
        embed_file = f'intermediate_data/{embedding_model}_embeddings/{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_{branching}_embed.npy'

    if not os.path.exists(embed_file):
        embedding_list = get_embeddings(topic_data_original['topic'], model=embedding_model)
        np.save(embed_file, embedding_list)
    else:
        embedding_list = np.load(embed_file)

    os.makedirs(f'intermediate_data/{embedding_model}_reduced_embeddings', exist_ok=True)

    # Shuffle embeddings with the same index as topic_data
    data = np.array(embedding_list)[shuffle_idx]

    # Apply dimensionality reduction methods (same as amazon.py)
    embeddings = np.array(data)
    embedding_methods_for_model = {}

    # PHATE
    print("Running PHATE...")
    reducer_model = phate.PHATE(n_jobs=-2, random_state=67, n_components=300, decay=20, t="auto", n_pca=None)
    embed_phate = reducer_model.fit_transform(data)
    embedding_methods_for_model["PHATE"] = embed_phate
    if float(add_noise) > 0:
        np.save(f"intermediate_data/{embedding_model}_reduced_embeddings/PHATE_{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_noise{add_noise}_{branching}_embed.npy", embed_phate)
    else:
        np.save(f"intermediate_data/{embedding_model}_reduced_embeddings/PHATE_{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_{branching}_embed.npy", embed_phate)

    # PCA using cuML (GPU-accelerated)
    print("Running PCA with cuML (GPU)...")
    pca_model = cuPCA(n_components=300)
    pca_result = pca_model.fit_transform(embeddings)
    embedding_methods_for_model["PCA"] = pca_result.to_output('numpy') if hasattr(pca_result, 'to_output') else np.array(pca_result)
    if float(add_noise) > 0:
        np.save(f"intermediate_data/{embedding_model}_reduced_embeddings/PCA_{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_noise{add_noise}_{branching}_embed.npy", embedding_methods_for_model["PCA"])
    else:
        np.save(f"intermediate_data/{embedding_model}_reduced_embeddings/PCA_{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_{branching}_embed.npy", embedding_methods_for_model["PCA"])

    # UMAP using cuML (GPU-accelerated)
    print("Running UMAP with cuML (GPU)...")
    umap_model = cuUMAP(n_components=300, min_dist=.05, n_neighbors=10)
    umap_result = umap_model.fit_transform(embeddings)
    embedding_methods_for_model["UMAP"] = umap_result.to_output('numpy') if hasattr(umap_result, 'to_output') else np.array(umap_result)
    if float(add_noise) > 0:
        np.save(f"intermediate_data/{embedding_model}_reduced_embeddings/UMAP_{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_noise{add_noise}_{branching}_embed.npy", embedding_methods_for_model["UMAP"])
    else:
        np.save(f"intermediate_data/{embedding_model}_reduced_embeddings/UMAP_{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_{branching}_embed.npy", embedding_methods_for_model["UMAP"])

    # t-SNE using cuML (GPU-accelerated)
    print("Running t-SNE with cuML (GPU)...")
    tsne_model = cuTSNE(n_components=2)
    tsne_result = tsne_model.fit_transform(embeddings)
    embedding_methods_for_model["tSNE"] = tsne_result.to_output('numpy') if hasattr(tsne_result, 'to_output') else np.array(tsne_result)
    if float(add_noise) > 0:
        np.save(f"intermediate_data/{embedding_model}_reduced_embeddings/tSNE_{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_noise{add_noise}_{branching}_embed.npy", embedding_methods_for_model["tSNE"])
    else:
        np.save(f"intermediate_data/{embedding_model}_reduced_embeddings/tSNE_{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_{branching}_embed.npy", embedding_methods_for_model["tSNE"])

    # PaCMAP
    print("Running PaCMAP...")
    pac = pacmap.PaCMAP(n_components=300, random_state=67)
    embedding_methods_for_model["PaCMAP"] = pac.fit_transform(embeddings)
    if float(add_noise) > 0:
        np.save(f"intermediate_data/{embedding_model}_reduced_embeddings/PaCMAP_{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_noise{add_noise}_{branching}_embed.npy", embedding_methods_for_model["PaCMAP"])
    else:
        np.save(f"intermediate_data/{embedding_model}_reduced_embeddings/PaCMAP_{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_{branching}_embed.npy", embedding_methods_for_model["PaCMAP"])

    '''
    # TriMAP
    print("Running TriMAP...")
    tr = trimap.TRIMAP(n_dims=300)
    embedding_methods_for_model["TriMAP"] = tr.fit_transform(embeddings)
    if float(add_noise) > 0:
        np.save(f"{embedding_model}_reduced_embeddings/TriMAP_{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_noise{add_noise}_{branching}_embed.npy", embedding_methods_for_model["TriMAP"])
    else:
        np.save(f"{embedding_model}_reduced_embeddings/TriMAP_{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_{branching}_embed.npy", embedding_methods_for_model["TriMAP"])
    '''
    # Store the embedding methods for this model
    embedding_models[embedding_model] = embedding_methods_for_model

# Run clustering and evaluation
scores_all = defaultdict(lambda: defaultdict(list))

# Create combinations of all models, reduction methods, and clustering methods
combo_params = [
    (embedding_model, embed_name, cluster_method)
    for embedding_model in embedding_models.keys()
    for embed_name in embedding_models[embedding_model].keys()
    for cluster_method in ["Agglomerative", "HDBSCAN", "DC"]
]

# Run each combo sequentially
combo_results = []
for embedding_model, embed_name, cluster_method in tqdm(combo_params, desc="Processing embedding-clustering combos"):
    embed_data = embedding_models[embedding_model][embed_name]
    result = safe_run_combo(embedding_model, embed_name, cluster_method, embed_data, cluster_levels, topic_dict, theme, t, max_sub, depth, synonyms, branching, add_noise)
    combo_results.append(result)

# Collect results
for embedding_model, embed_name, cluster_method, combo_scores in combo_results:
    scores_all[(embedding_model, embed_name, cluster_method)]["FM"] = combo_scores["FM"]
    scores_all[(embedding_model, embed_name, cluster_method)]["Rand"] = combo_scores["Rand"]
    scores_all[(embedding_model, embed_name, cluster_method)]["ARI"] = combo_scores["ARI"]
    scores_all[(embedding_model, embed_name, cluster_method)]["AMI"] = combo_scores["AMI"]

print(f"\n{'='*60}")
print("All clustering and evaluation complete!")
print(f"{'='*60}")

# Save results to CSV
rows = []
for (embedding_model, embed_name, cluster_method), score_dict in scores_all.items():
    n_levels = len(score_dict["FM"])
    for i in range(n_levels):
        rows.append({
            "embedding_model": embedding_model,
            "reduction_method": embed_name,
            "cluster_method": cluster_method,
            "level": cluster_levels[i],
            "FM": score_dict["FM"][i],
            "Rand": score_dict["Rand"][i],
            "ARI": score_dict["ARI"][i],
            "AMI": score_dict["AMI"][i],
        })

# Create DataFrame and save
scores_df = pd.DataFrame(rows)
scores_df = scores_df.sort_values(by=["embedding_model", "reduction_method", "cluster_method", "level"]).reset_index(drop=True)

os.makedirs("results", exist_ok=True)
if float(add_noise) > 0:
    output_file = f"results/{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_noise{add_noise}_{branching}_clustering_scores.csv"
else:
    output_file = f"results/{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_{branching}_clustering_scores.csv"

scores_df.to_csv(output_file, index=False)
print(f"\nResults saved to: {output_file}")