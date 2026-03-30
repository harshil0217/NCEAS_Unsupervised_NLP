"""
Generalized Benchmark Evaluation Pipeline

This script provides a unified pipeline for running dimensionality reduction and clustering
experiments on different benchmark datasets. The dataset is specified via command-line argument.

Usage:
    python eval_pipeline.py --dataset <dataset_name>

    Available datasets: amazon, dbpedia, arxiv, rcv1, wos

Example:
    python eval_pipeline.py --dataset dbpedia
"""

import os
import sys
from dotenv import load_dotenv
import json
import argparse
load_dotenv()

target_folder = "src"

# Use __file__ to get the script's actual location, not the terminal's CWD
current_dir = os.path.dirname(os.path.abspath(__file__))

while os.path.basename(current_dir) != target_folder:
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    if parent_dir == current_dir:
        raise FileNotFoundError(f"{target_folder} not found in the directory tree.")
    current_dir = parent_dir

os.chdir(current_dir)
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
# cuML GPU-accelerated dimensionality reduction
import cuml
from cuml.decomposition import PCA as cuPCA
from cuml.manifold import TSNE as cuTSNE
from cuml.manifold import UMAP as cuUMAP 

# ========================
# Clustering
# ========================
from custom_packages.diffusion_condensation import DiffusionCondensation as dc
from scipy.cluster.hierarchy import fcluster, to_tree, linkage

# cuML GPU-accelerated clustering
from cuml.cluster import AgglomerativeClustering as cuAgglomerativeClustering
from cuml.cluster import HDBSCAN as cuHDBSCAN


# ======================
# Evaluation Metrics
# ======================
from custom_packages.fowlkes_mallows import FowlkesMallows
from custom_packages.dendrogram_purity import dendrogram_purity
from sklearn.metrics import adjusted_rand_score, rand_score, adjusted_mutual_info_score

from tqdm import tqdm

# ==============
# Global Config
# ==============
np.random.seed(67)
warnings.filterwarnings("ignore")

# =====================================
# Dendrogram Purity Sampling Config
# =====================================
DENDROGRAM_PURITY_SAMPLE_SIZE = 2000

# Reload modules if needed
importlib.reload(phate)

def get_linkage_matrix(model):
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  
            else:
                current_count += counts[child_idx - n_samples]  # internal node
        counts[i] = current_count
    
    linkage_matrix = np.column_stack([
        model.children_,   
        model.distances_, 
        counts             
    ])
    return linkage_matrix


# =====================================
# Dendrogram Purity Sampling Functions
# =====================================

def sample_for_dendrogram_purity(labels, sample_size=DENDROGRAM_PURITY_SAMPLE_SIZE, random_state=67):
    """
    Sample indices with weighted sampling to preserve class distribution.

    Args:
        labels: Array of class labels at the lowest hierarchy level
        sample_size: Number of points to sample (default: 2000)
        random_state: Random seed for reproducibility

    Returns:
        sampled_indices: Array of indices into the original data
    """
    rng = np.random.RandomState(random_state)

    n_samples = len(labels)
    if n_samples <= sample_size:
        # If we have fewer points than sample_size, return all indices
        return np.arange(n_samples)

    # Get unique classes and their counts
    unique_classes, class_counts = np.unique(labels, return_counts=True)

    # Calculate how many samples to take from each class (proportional)
    class_proportions = class_counts / n_samples
    samples_per_class = np.round(class_proportions * sample_size).astype(int)

    # Adjust to ensure we get exactly sample_size (handle rounding)
    diff = sample_size - samples_per_class.sum()
    if diff != 0:
        # Add/remove from largest classes
        sorted_idx = np.argsort(class_counts)[::-1]
        for i in range(abs(diff)):
            idx = sorted_idx[i % len(sorted_idx)]
            samples_per_class[idx] += np.sign(diff)

    # Sample from each class
    sampled_indices = []
    for cls, n_samples_cls in zip(unique_classes, samples_per_class):
        cls_indices = np.where(labels == cls)[0]
        # Handle case where class has fewer samples than requested
        n_to_sample = min(n_samples_cls, len(cls_indices))
        if n_to_sample > 0:
            sampled = rng.choice(cls_indices, size=n_to_sample, replace=False)
            sampled_indices.extend(sampled)

    return np.array(sampled_indices)


def build_sampled_tree_for_purity(embed_data, sampled_indices, cluster_method):
    """
    Build a smaller tree from sampled points for dendrogram purity calculation.
    Uses the same clustering algorithm being evaluated to ensure consistency.

    Args:
        embed_data: Full embedding data array
        sampled_indices: Indices of sampled points
        cluster_method: Clustering method to use ("Agglomerative", "HDBSCAN", or "DC")

    Returns:
        tree: ClusterNode tree root
        index_map: Mapping from sampled tree leaf indices to original indices
    """
    # Extract sampled embeddings
    sampled_embeddings = embed_data[sampled_indices]

    if cluster_method == "Agglomerative":
        # Use scipy ward linkage (cuML AgglomerativeClustering doesn't expose linkage tree)
        Z_sampled = linkage(sampled_embeddings, method='ward')
        tree, _ = to_tree(Z_sampled, rd=True)

    elif cluster_method == "HDBSCAN":
        # Use cuML HDBSCAN
        model = cuHDBSCAN(min_cluster_size=5, min_samples=1)
        model.fit(sampled_embeddings)
        Z_sampled = model.single_linkage_tree_.to_numpy()
        tree, _ = to_tree(Z_sampled, rd=True)

    elif cluster_method == "DC":
        # Use Diffusion Condensation
        dc_model = dc(min_clusters=1, max_iterations=5000, k=10, alpha=3)
        dc_model.fit(sampled_embeddings)
        tree = dc_model.tree_

        # Fallback to ward linkage if DC fails to produce a tree
        if tree is None:
            Z_sampled = linkage(sampled_embeddings, method='ward')
            tree, _ = to_tree(Z_sampled, rd=True)

    else:
        raise ValueError(f"Unknown cluster_method: {cluster_method}")

    # Return tree and the indices mapping (leaf i in tree -> sampled_indices[i] in original)
    return tree, sampled_indices


# =====================================
# Dataset-Specific Loading Functions
# =====================================

def load_amazon():
    """Load and preprocess Amazon dataset."""
    amz_40 = pd.read_csv("data/amazon/train_40k.csv")
    amz_10 = pd.read_csv("data/amazon/val_10k.csv")

    amz = pd.concat([amz_40, amz_10])
    amz = amz.drop_duplicates(subset='Title', keep=False).reset_index(drop=True)
    amz = amz.drop_duplicates(subset='productId', keep=False).reset_index(drop=True)

    amz = amz.rename(columns={'Title': 'topic'})
    amz = amz.rename(columns={'Cat1': 'category_0'})
    amz = amz.rename(columns={'Cat2': 'category_1'})
    amz = amz.rename(columns={'Cat3': 'category_2'})

    amz = amz.dropna().reset_index(drop=True)
    amz = amz[amz['topic'].apply(lambda x: isinstance(x, str) and x.strip() != '')].reset_index(drop=True)

    amz.to_csv("data/amazon/amz_data.csv")

    return amz


def load_dbpedia():
    """Load and preprocess DBpedia dataset."""
    def clean_dbpedia(text):
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    db = pd.read_csv('data/dbpedia/DBPEDIA_test.csv')

    db = db.rename(columns={"text": "topic"})
    db = db.rename(columns={"l1": "category_0"})
    db = db.rename(columns={"l2": "category_1"})
    db = db.rename(columns={"l3": "category_2"})

    db['topic'] = db['topic'].astype(str).apply(clean_dbpedia)

    print(db.iloc[0])

    return db


def load_arxiv():
    """Load and preprocess arXiv dataset."""
    arx = pd.read_csv("data/arxiv/arxiv_clean.csv")
    arx = arx.dropna().reset_index(drop=True)
    return arx


def load_rcv1():
    """Load and preprocess RCV1 dataset."""
    rcv1 = pd.read_csv('data/rcv1/rcv1.csv')

    rcv1 = rcv1.drop_duplicates(subset='topic', keep=False).reset_index(drop=True)
    rcv1 = rcv1.drop_duplicates(subset='item_id', keep=False).reset_index(drop=True)

    rcv1 = rcv1.dropna().reset_index(drop=True)
    rcv1 = rcv1[rcv1['topic'].apply(lambda x: isinstance(x, str) and x.strip() != '')].reset_index(drop=True)

    rcv1.to_csv("data/rcv1/rcv1_data.csv")

    return rcv1


def load_wos():
    """Load and preprocess Web of Science dataset."""
    wos = pd.read_excel('data/WebOfScience/Meta-data/Data.xlsx')

    new = []
    for i, row in wos.iterrows():
        result = {}
        result['topic'] = str(row['keywords'])
        result['category_0'] = row['Domain']
        result['category_1'] = row['area']
        new.append(result)

    wos = pd.DataFrame(new)

    return wos


# =====================================
# Dataset Configuration Dictionary
# =====================================

DATASET_CONFIGS = {
    "amazon": {
        "load_function": load_amazon,
        "depth": 3,
        "short": "amz",
        "results_filename": "amazon_clustering_scores.csv",
        "batch_size": 32,
        "reduction_methods": ["PHATE", "PCA", "UMAP", "tSNE", "PaCMAP"],
    },
    "dbpedia": {
        "load_function": load_dbpedia,
        "depth": 3,
        "short": "db",
        "results_filename": "db_clustering_scores.csv",
        "batch_size": 32,
        "reduction_methods": ["PHATE", "PCA", "UMAP", "tSNE", "PaCMAP"],
    },
    "arxiv": {
        "load_function": load_arxiv,
        "depth": 2,
        "short": "arx",
        "results_filename": "arxiv_clustering_scores.csv",
        "batch_size": 32,
        "reduction_methods": ["PHATE", "PCA", "UMAP", "tSNE", "PaCMAP"],
    },
    "rcv1": {
        "load_function": load_rcv1,
        "depth": 2,
        "short": "rcv1",
        "results_filename": "rcv1_clustering_scores.csv",
        "batch_size": 8,
        "reduction_methods": ["PHATE", "PCA", "UMAP", "tSNE", "PaCMAP"],
    },
    "wos": {
        "load_function": load_wos,
        "depth": 2,
        "short": "wos",
        "results_filename": "wos_clustering_scores.csv",
        "batch_size": 64,
        "reduction_methods": ["PHATE", "PCA", "UMAP", "tSNE", "PaCMAP"],
    },
}


# =====================================
# Core Pipeline Functions
# =====================================

def get_embeddings(texts, model_id, batch_size=32):
    """
    Generate embeddings using SentenceTransformer.

    Args:
        texts: List or Series of text inputs
        model_id: Model identifier for SentenceTransformer
        batch_size: Batch size for encoding

    Returns:
        numpy array of embeddings
    """
    print("Using device:", device)
    print(f"Number of texts: {len(texts)}")

    model = SentenceTransformer(
        model_id,
        model_kwargs={"attn_implementation": "sdpa", "device_map": "auto"} if "Qwen" in model_id else {},
        tokenizer_kwargs={"padding_side": "left"} if "Qwen" in model_id else {}
    )

    # Print token statistics
    tok = model.tokenizer(texts.tolist(), truncation=False, padding=False)
    lens = [len(x) for x in tok['input_ids']]
    print(f"Total tokens: {sum(lens):,}")
    print(f"Avg tokens: {sum(lens)/len(lens):.1f}")
    print(f"Max tokens: {max(lens)}")

    print("Generating embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings


def apply_dimensionality_reduction(embeddings, reduction_dir, embed_filename, reduction_methods):
    """
    Apply dimensionality reduction methods to embeddings.

    Args:
        embeddings: Input embeddings (numpy array)
        reduction_dir: Directory to save reduced embeddings
        embed_filename: Filename prefix for saved embeddings
        reduction_methods: List of reduction methods to apply

    Returns:
        Dictionary mapping method names to reduced embeddings
    """
    embedding_methods = {}

    # Define all possible reduction tasks
    all_reduction_tasks = {
        "PHATE": {
            "path": f"{reduction_dir}/PHATE_{embed_filename}.npy",
            "run": lambda: phate.PHATE(n_jobs=-2, random_state=67, n_components=300, decay=20, t="auto", n_pca=None).fit_transform(embeddings)
        },
        "PCA": {
            "path": f"{reduction_dir}/PCA_{embed_filename}.npy",
            "run": lambda: cuPCA(n_components=300).fit_transform(embeddings)
        },
        "UMAP": {
            "path": f"{reduction_dir}/UMAP_{embed_filename}.npy",
            "run": lambda: cuUMAP(n_components=300, min_dist=.05, n_neighbors=10).fit_transform(embeddings)
        },
        "tSNE": {
            "path": f"{reduction_dir}/tSNE_{embed_filename}.npy",
            "run": lambda: cuTSNE(n_components=2).fit_transform(embeddings)
        },
        "PaCMAP": {
            "path": f"{reduction_dir}/PaCMAP_{embed_filename}.npy",
            "run": lambda: pacmap.PaCMAP(n_components=300, random_state=67).fit_transform(embeddings)
        },
        "TriMAP": {
            "path": f"{reduction_dir}/TriMAP_{embed_filename}.npy",
            "run": lambda: trimap.TRIMAP(n_dims=300).fit_transform(embeddings)
        }
    }

    # Filter to only requested methods
    reduction_tasks = {k: v for k, v in all_reduction_tasks.items() if k in reduction_methods}

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

        embedding_methods[method_name] = result

    return embedding_methods


def cluster_combo(embedding_model, embed_name, cluster_method, embedding_models,
                   cluster_levels, topic_dict, label_dir, short, lowest_level_labels):
    embed_data = embedding_models[embedding_model][embed_name]
    combo_scores = {"FM": [], "Rand": [], "ARI": [], "AMI": [], "Dendrogram Purity": []}
    
    print(f"\n{'='*60}")
    print(f"Processing Embedding Method: {embed_name}")
    print(f"Clustering Method: {cluster_method}")
    print(f"Embedding shape: {embed_data.shape}")
    print(f"{'='*60}")

    # Build the tree once per embedding-clustering method combination
    tree = None
    Z = None

    if cluster_method == "Agglomerative":
        print("Using cuML Agglomerative Clustering (GPU)...")
        model = cuAgglomerativeClustering(n_clusters=None, compute_distances=True, connectivity='average')
        model.fit(embed_data)

        # Generate Linkage Matrix
        Z = get_linkage_matrix(model)

        # Save linkage matrix
        linkage_dir = os.path.join(f"intermediate_data/{embedding_model}_linkage", short, embed_name)
        os.makedirs(linkage_dir, exist_ok=True)
        linkage_path = os.path.join(linkage_dir, f"Agglomerative_linkage.npy")
        np.save(linkage_path, Z)
        print(f"Saved linkage matrix to {linkage_path}")

        # Create NodeCluster Tree for dendrogram purity calculation
        tree, node_list = to_tree(Z, rd=True)

    elif cluster_method == "HDBSCAN":
        linkage_path = os.path.join(
            f"intermediate_data/{embedding_model}_linkage", short, embed_name,
            f"HDBSCAN_linkage.npy"
        )

        if os.path.exists(linkage_path):
            print(f"Loading cached HDBSCAN linkage from {linkage_path}")
            Z = np.load(linkage_path)
            tree, node_list = to_tree(Z, rd=True)
        else:
            print("Using cuML HDBSCAN (GPU)...")
            model = cuHDBSCAN(min_cluster_size=5, min_samples=1)
            model.fit(embed_data)

            # Convert GPU result to numpy
            Z = model.single_linkage_tree_.to_numpy()

            # Save linkage matrix
            os.makedirs(os.path.dirname(linkage_path), exist_ok=True)
            np.save(linkage_path, Z)
            print(f"Saved linkage matrix to {linkage_path}")

            # Create NodeCluster Tree for dendrogram purity calculation
            tree, node_list = to_tree(Z, rd=True)

    elif cluster_method == "DC":
        print(f"Running Diffusion Condensation for {embed_name}")
        # Set min_clusters=1 to force complete dendrogram that can be cut at any level
        dc_model = dc(min_clusters=1, max_iterations=5000, k=10, alpha=3)
        dc_model.fit(embed_data)

        # DC builds ClusterNode tree directly (no linkage matrix needed)
        tree = dc_model.tree_
        node_list = dc_model.node_list_

    # Build sampled tree for dendrogram purity calculation (2000 points)
    # Filter out NaN values from lowest_level_labels before sampling
    valid_mask = ~pd.isna(lowest_level_labels)
    valid_indices = np.where(valid_mask)[0]
    valid_labels = lowest_level_labels[valid_mask]

    print(f"Building sampled tree for dendrogram purity ({DENDROGRAM_PURITY_SAMPLE_SIZE} points) using {cluster_method}...")
    # Sample from valid indices only
    sampled_relative_indices = sample_for_dendrogram_purity(valid_labels)
    sampled_indices = valid_indices[sampled_relative_indices]
    sampled_tree, _ = build_sampled_tree_for_purity(embed_data, sampled_indices, cluster_method)
    sampled_labels_for_purity = lowest_level_labels[sampled_indices]
    print(f"Sampled {len(sampled_indices)} points for dendrogram purity")

    # Now iterate through cluster levels
    for level in cluster_levels:
        print(f"Testing cluster level: {level}")

        if cluster_method == "Agglomerative":
            # Slice the dendrogram at the requested level using SciPy
            labels = fcluster(Z, level, criterion='maxclust')
            print(f"Agglomerative clustering complete. Unique labels: {len(np.unique(labels))}")

        elif cluster_method == "HDBSCAN":
            label_path = os.path.join(
                f"intermediate_data/{embedding_model}_labels", short, embed_name,
                f"HDB_{level}_labels.npy"
            )

            if os.path.exists(label_path):
                print(f"Loading cached HDBSCAN labels from {label_path}")
                labels = np.load(label_path)
            else:
                # Slice the tree at the requested level using SciPy
                labels = fcluster(Z, level, criterion='maxclust')

                os.makedirs(os.path.dirname(label_path), exist_ok=True)
                np.save(label_path, labels)
                print(f"HDBSCAN fcluster cut at {level}. Unique labels: {len(np.unique(labels))}")

        elif cluster_method == "DC":
            label_path = os.path.join(
                f"intermediate_data/{embedding_model}_labels", short, embed_name,
                f"DC_{level}_labels.npy"
            )

            if os.path.exists(label_path):
                print(f"Loading cached DC labels from {label_path}")
                labels = np.load(label_path)
            else:
                # Cut the tree at the requested number of clusters
                if dc_model is not None:
                    dc_model.get_labels(n_clusters=level)
                    labels = dc_model.labels_
                else:
                    # Fallback used ward linkage, use fcluster
                    labels = fcluster(Z, level, criterion='maxclust')

                os.makedirs(os.path.dirname(label_path), exist_ok=True)
                np.save(label_path, labels)
                print(f"DC tree cut at {level}. Unique labels: {len(np.unique(labels))}")

        # Match to closest ground truth level
        available_levels = np.array(sorted(topic_dict.keys()))
        closest_level = min(available_levels, key=lambda k: abs(k - level))
        print(f"Ground truth: Using closest level {closest_level} (requested: {level})")

        topic_series = topic_dict[closest_level]
        valid_idx = (~pd.isna(topic_series))
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

        # Compute dendrogram purity using sampled tree and sampled labels
        # The sampled_tree uses indices 0 to len(sampled_indices)-1 as leaf nodes
        # sampled_labels_for_purity contains the corresponding ground truth labels
        dp = dendrogram_purity(sampled_tree, sampled_labels_for_purity)

        print(f"Scores - FM: {fm_score:.4f}, Rand: {rand:.4f}, ARI: {ari:.4f}, AMI: {ami:.4f}, Dendrogram_Purity: {dp:.4f}")

        combo_scores["FM"].append(fm_score)
        combo_scores["Rand"].append(rand)
        combo_scores["ARI"].append(ari)
        combo_scores["AMI"].append(ami)
        combo_scores["Dendrogram Purity"].append(dp)

    return embedding_model, embed_name, cluster_method, combo_scores



def run_pipeline(dataset_name):
    """
    Run the complete evaluation pipeline for a specified dataset.

    Args:
        dataset_name: Name of the dataset to process (e.g., 'amazon', 'dbpedia')
    """
    # Validate dataset name
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")

    config = DATASET_CONFIGS[dataset_name]
    print(f"\n{'='*80}")
    print(f"Running pipeline for dataset: {dataset_name.upper()}")
    print(f"{'='*80}\n")

    # Load dataset using dataset-specific function
    print("Loading dataset...")
    data = config["load_function"]()
    print(f"Dataset shape: {data.shape}\n")

    # Embedding models to test
    embedding_model_names = [
        "Qwen/Qwen3-Embedding-0.6B",
        "sentence-transformers/all-MiniLM-L6-v2",
    ]

    embedding_models = {}  # Will store {model_name: {reduction_method: embeddings}}

    # Prepare data once (same for all embedding models)
    shuffle_idx = np.random.RandomState(seed=67).permutation(len(data))
    topic_data = data.iloc[shuffle_idx].reset_index(drop=True)
    reverse_idx = np.argsort(shuffle_idx)

    # Build topic_dict from ground truth categories
    topic_dict = {}
    for col in topic_data.columns:
        if re.match(r'^category_\d+$', col):
            unique_count = len(topic_data[col].unique())
            topic_dict[unique_count] = np.array(topic_data[col])

    # Determine cluster levels from hierarchy depth
    depth = config["depth"]
    print(f"Depth: {depth}")
    print(f"Building cluster levels by counting unique categories at each level...\n")

    cluster_levels = []
    for i in reversed(range(0, depth)):
        if f'category_{i}' in topic_data.columns:
            unique_count = len(topic_data[f'category_{i}'].unique())
            print(f"Level {i} (category_{i}): {unique_count} unique categories")
            cluster_levels.append(unique_count)

    print(f"\nFinal cluster_levels (from deepest to shallowest): {cluster_levels}\n")

    # Process each embedding model
    for embedding_model in embedding_model_names:
        print(f"\n{'='*60}")
        print(f"Processing embedding model: {embedding_model}")
        print(f"{'='*60}\n")

        os.makedirs(f'intermediate_data/{embedding_model}_results', exist_ok=True)

        reduction_dir = f"intermediate_data/{embedding_model}_reduced_embeddings"
        os.makedirs(reduction_dir, exist_ok=True)

        embedding_dir = f"intermediate_data/{embedding_model}_embeddings"
        os.makedirs(embedding_dir, exist_ok=True)

        embedding_path = f"{embedding_dir}/{config['short']}.npy"

        # Generate or load embeddings
        if os.path.exists(embedding_path):
            print(f"Loading existing embeddings from {embedding_path}")
            embedding_list = np.load(embedding_path)
        else:
            print("Generating embeddings...")
            embedding_list = get_embeddings(
                data['topic'],
                model_id=embedding_model,
                batch_size=config["batch_size"]
            )
            np.save(embedding_path, embedding_list)
            print(f"Saved embeddings to {embedding_path}")

        # Shuffle embeddings with the same index as topic_data
        embeddings = np.array(embedding_list)[shuffle_idx]

        # Apply dimensionality reduction
        embedding_methods_for_model = apply_dimensionality_reduction(
            embeddings=embeddings,
            reduction_dir=reduction_dir,
            embed_filename=config["short"],
            reduction_methods=config["reduction_methods"]
        )

        # Store the final dict for the global embedding_models
        embedding_models[embedding_model] = embedding_methods_for_model

    # Run clustering and evaluation
    scores_all = defaultdict(lambda: defaultdict(list))

    combo_params = [
        (embedding_model, embed_name, cluster_method)
        for embedding_model in embedding_models.keys()
        for embed_name in embedding_models[embedding_model].keys()
        for cluster_method in ["Agglomerative", "HDBSCAN", "DC"]
    ]

    label_dir = f"intermediate_data/{embedding_model}_labels"
    os.makedirs(label_dir, exist_ok=True)

    # Get lowest level labels for dendrogram purity sampling
    # The deepest category (highest cluster count) is the finest granularity
    lowest_level_key = max(topic_dict.keys())
    lowest_level_labels = topic_dict[lowest_level_key]
    print(f"Using lowest level labels with {lowest_level_key} unique classes for dendrogram purity sampling")

    # Run each combo sequentially
    combo_results = []
    for embedding_model, embed_name, cluster_method in tqdm(combo_params, desc="Processing embedding-clustering combos"):
        result = cluster_combo(
            embedding_model,
            embed_name,
            cluster_method,
            embedding_models,
            cluster_levels,
            topic_dict,
            label_dir,
            short=config["short"],
            lowest_level_labels=lowest_level_labels
        )
        combo_results.append(result)

    for embedding_model, embed_name, cluster_method, combo_scores in combo_results:
        scores_all[(embedding_model, embed_name, cluster_method)]["FM"] = combo_scores["FM"]
        scores_all[(embedding_model, embed_name, cluster_method)]["Rand"] = combo_scores["Rand"]
        scores_all[(embedding_model, embed_name, cluster_method)]["ARI"] = combo_scores["ARI"]
        scores_all[(embedding_model, embed_name, cluster_method)]["AMI"] = combo_scores["AMI"]
        scores_all[(embedding_model, embed_name, cluster_method)]["Dendrogram Purity"] = combo_scores["Dendrogram Purity"]

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
                "Dendrogram_Purity": score_dict["Dendrogram Purity"][i],
            })

    # Create DataFrame
    scores_df = pd.DataFrame(rows)

    # Sort for easier viewing
    print(scores_df)
    print(scores_df.columns)
    scores_df = scores_df.sort_values(
        by=["embedding_model", "reduction_method", "cluster_method", "level"]
    ).reset_index(drop=True)

    # Save results
    os.makedirs("results", exist_ok=True)
    results_path = f"results/{config['results_filename']}"
    scores_df.to_csv(results_path, index=False)

    print(f"\nResults saved to: {results_path}")
    print(f"Pipeline complete for dataset: {dataset_name.upper()}\n")


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description="Run benchmark evaluation pipeline on specified dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Available datasets:
            amazon    - Amazon product categories (3 levels)
            dbpedia   - DBpedia ontology topics (3 levels)
            arxiv     - arXiv paper categories (2 levels)
            rcv1      - Reuters RCV1 news categories (2 levels)
            wos       - Web of Science publications (2 levels)

            Example usage:
            python eval_pipeline.py --dataset dbpedia
            python eval_pipeline.py --dataset amazon
                    """
                )

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=list(DATASET_CONFIGS.keys()),
        help='Dataset to process'
    )

    args = parser.parse_args()

    # Run pipeline for specified dataset
    run_pipeline(args.dataset)


if __name__ == "__main__":
    main()