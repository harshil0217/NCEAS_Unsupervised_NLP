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

# ==========================
# Dimensionality Reduction
# ==========================
import phate
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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
from sklearn.metrics import adjusted_rand_score, rand_score


from tqdm import tqdm
# ==============
# Global Config
# ==============
np.random.seed(42)
warnings.filterwarnings("ignore")

# Reload modules if needed
importlib.reload(phate)
from openai import OpenAI
key = os.getenv('GPT_API_KEY')

client = OpenAI(api_key=key)

from groq import Groq
key = os.getenv('GROQ_API_KEY')

llm_client = Groq(api_key=key)


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def export_top_clusters_txt(
    top_clusters,
    embedding_model: str,
    reduction_method: str,
    clustering_level: int,
    clustering_method: str = "HERCULES-DIRECT",
) -> str:
    """
    Export top cluster summaries and key Cluster-object field documentation to a txt file.

    Output path pattern:
        {embedding_model}_results/cluster_documentation/{reduction_method}/level_{clustering_level}/top_clusters_{clustering_method}.txt
    """
    if top_clusters is None:
        return ""

    if not isinstance(top_clusters, (list, tuple)):
        top_clusters = [top_clusters]

    output_dir = Path(f"{embedding_model}_results") / "cluster_documentation" / _safe_slug(reduction_method) / f"level_{clustering_level}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"top_clusters_{_safe_slug(clustering_method)}.txt"

    field_docs = {
        "level": "Depth level in hierarchy (0 = leaf/original item).",
        "id": "Internal sequential cluster ID assigned by Cluster._get_next_id().",
        "original_data_type": "Underlying leaf data type (e.g., 'text', 'numeric', 'image').",
        "children": "Child Cluster nodes (empty for leaf clusters).",
        "parent": "Parent Cluster node, or None when this is a top/root cluster.",
        "title": "Short cluster label/title.",
        "description": "Cluster summary/description.",
        "original_id": "Original source item ID (mainly for level-0 nodes).",
        "_raw_item_data": "Original raw source payload for leaf item(s).",
    }

    lines = [
        "Cluster Object Documentation + Top Cluster Export",
        "=" * 60,
        f"embedding_model: {embedding_model}",
        f"reduction_method: {reduction_method}",
        f"clustering_method: {clustering_method}",
        f"clustering_level: {clustering_level}",
        f"num_top_clusters: {len(top_clusters)}",
        "",
        "Cluster Object Field Documentation",
        "-" * 60,
    ]

    for field_name, field_doc in field_docs.items():
        lines.append(f"- {field_name}: {field_doc}")

    lines.extend(["", "Top Clusters", "-" * 60])

    for idx, cluster in enumerate(top_clusters, start=1):
        children = getattr(cluster, "children", None)
        cluster_info = {
            "level": getattr(cluster, "level", None),
            "id": getattr(cluster, "id", None),
            "original_data_type": getattr(cluster, "original_data_type", None),
            "children_count": len(children) if children is not None else 0,
            "parent_id": getattr(getattr(cluster, "parent", None), "id", None),
            "title": getattr(cluster, "title", None),
            "description": getattr(cluster, "description", None),
            "original_id": getattr(cluster, "original_id", None),
            "_raw_item_data": getattr(cluster, "_raw_item_data", None),
        }

        lines.append(f"\n[Top Cluster {idx}]")
        for key_name, value in cluster_info.items():
            if key_name == "_raw_item_data":
                value_str = str(value)
                if len(value_str) > 500:
                    value_str = value_str[:500] + "... [truncated]"
                lines.append(f"{key_name}: {value_str}")
            else:
                lines.append(f"{key_name}: {value}")

    output_file.write_text("\n".join(lines), encoding="utf-8")
    return str(output_file)

def groq_caller(prompt: str) -> str:
    """
    Generates text using remote Groq/OpenAI client.
    Returns a raw string (potentially containing JSON/markdown) 
    to be processed by _parse_llm_response later.
    """
    # 1. Mirrored System/User structure from your local Gemma function
    messages = [
        {"role": "system", "content": "You are a helpful assistant providing concise summaries."},
        {"role": "user", "content": prompt}
    ]

    try:
        # 2. Remote API Call
        # We don't use response_format="json_object" here because your 
        # parser function expects to handle the raw string/markdown itself.
        response = llm_client.chat.completions.create(
            model="openai/gpt-oss-20b", 
            messages=messages,
            temperature=0.0,  # Mirroring do_sample=False
            max_tokens=16384
        )

        # 3. Accessing the text content
        # Note: Groq/OpenAI use .choices[0].message.content
        decoded_content = response.choices[0].message.content
        
        return decoded_content.strip()

    except Exception as e:
        # 4. Mirrored error handling from your local logic
        print(f"Error during remote Groq generation: {e}")
        return ""

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

def get_embeddings(texts, model="text-embedding-3-small"):
    """
    Fetches embeddings using the specified backend: 'gpt' (OpenAI) or 'sentence-transformers'.
    
    Args:
        texts (list of str): List of text inputs.
        backend (str): 'gpt' or 'sentence-transformers'.
        model (str): Model name for the chosen backend.
        
    Returns:
        list: List of embeddings.
    """
 # Make sure `openai` is configured with your API key
    batch_size = 200
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Fetching GPT embeddings", unit="batch"):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(input=batch, model=model)
        batch_embeddings = [entry.embedding for entry in response.data]
        embeddings.extend(batch_embeddings)

    return embeddings

embedding_model = "text-embedding-3-large" 
os.makedirs(f'{embedding_model}_results', exist_ok=True)
os.makedirs('gpt_embeddings', exist_ok=True)
embedding_list = get_embeddings(amz['topic'], model=embedding_model)

np.save("gpt_embeddings/amz_embed.npy",embedding_list)

embedding_list=np.load("gpt_embeddings/amz_embed.npy")

os.makedirs(f'{embedding_model}_reduced_embeddings', exist_ok=True)

shuffle_idx = np.random.RandomState(seed=42).permutation(len(amz))
# Shuffle both documents and embeddings using the same index
topic_data = amz.iloc[shuffle_idx].reset_index(drop=True)
data = np.array(embedding_list)[shuffle_idx] 
reverse_idx = np.argsort(shuffle_idx)

topic_dict = {}
for col in topic_data.columns:
    if re.match(r'^category_\d+$', col): 
        unique_count = len(topic_data[col].unique())
        topic_dict[unique_count] = np.array(topic_data[col])

reducer_model = phate.PHATE(n_jobs=-2,random_state=42, n_components=300,decay=20,t="auto",n_pca=None) #{'k':10,'alpha':4,'t':3}
embed_phate = reducer_model.fit_transform(data)
np.save(f"{embedding_model}_reduced_embeddings/PHATE_amz_embed.npy",embed_phate)

embed_phate  =np.load(f"{embedding_model}_reduced_embeddings/PHATE_amz_embed.npy")

depth= 3
print(f"Depth: {depth}")
print(f"Building cluster levels by counting unique categories at each level...\n")

cluster_levels=[]
for i in reversed(range(0, depth)):
    unique_count = len(topic_data[f'category_{i}'].unique())
    print(f"Level {i} (category_{i}): {unique_count} unique categories")
    cluster_levels.append(unique_count)

print(f"\nFinal cluster_levels (from deepest to shallowest): {cluster_levels}")

import numpy as np
from sklearn.decomposition import PCA
import umap
import matplotlib.pyplot as plt
include_pca =True
include_umap=True

# Load your embeddings
embeddings = np.array(data)
embedding_methods = {}
# PCA to 2D

embedding_methods["PHATE"]  =embed_phate
if include_pca:
    pca = PCA(n_components=300)
    embedding_methods["PCA"] = pca.fit_transform(embeddings)
np.save(f"{embedding_model}_reduced_embeddings/PCA_amz_embed.npy",embedding_methods["PCA"])

# # UMAP to 2D
if include_umap:
    umap_model = umap.UMAP(n_components=300, random_state=42,min_dist=.05,n_neighbors=10)
    embedding_methods["UMAP"] = umap_model.fit_transform(embeddings)
np.save(f"{embedding_model}_reduced_embeddings/UMAP_amz_embed_new.npy",embedding_methods["UMAP"])
from sklearn.manifold import TSNE

# # # Fit t-SNE
# tsne_model = TSNE(n_components=3, random_state=42)
# embedding_methods["tSNE"] = tsne_model.fit_transform(embeddings)
# np.save(f"{embedding_model}_reduced_embeddings/tSNE_amz_embed.npy",embedding_methods["tSNE"])




scores_all = defaultdict(lambda: defaultdict(list))

for embed_name, embed_data in tqdm(embedding_methods.items()):
    print(f"\n{'='*60}")
    print(f"Processing Embedding Method: {embed_name}")
    print(f"Embedding shape: {embed_data.shape}")
    print(f"{'='*60}")
    
    for cluster_method in ["HERCULES-DIRECT", "Agglomerative", "HDBSCAN","DC"]:
        print(f"\n Clustering Method: {cluster_method}")
        
        for level in cluster_levels:
            print(f"Testing cluster level: {level}")
            
            # Clustering
            if cluster_method == "Agglomerative":
                model = AgglomerativeClustering(n_clusters=level)
                model.fit(embed_data)
                labels = model.labels_
                print(f"Agglomerative clustering complete. Unique labels: {len(np.unique(labels))}")
                
            elif cluster_method == "HDBSCAN":
                model = HDBSCAN(min_cluster_size=level)
                model.fit(embed_data)
                labels = model.labels_
                Z = model.single_linkage_tree_.to_numpy()
                labels = fcluster(Z, i, criterion='maxclust')
                labels[labels == -1] = labels.max() + 1
                print(f"HDBSCAN clustering complete. Unique labels: {len(np.unique(labels))}")
                
            elif cluster_method=="DC":
                model = dc(min_clusters=level, max_iterations=5000,k=10,alpha=3)
                model.fit(embed_data)
                labels =model.labels_
                print(f"DC clustering complete. Unique labels: {len(np.unique(labels))}")
                
            elif cluster_method=='HERCULES-DIRECT':
                print(f"Running HERCULES-DIRECT...")
                hercules = Hercules(
                    level_cluster_counts=[level],
                    representation_mode="direct",
                    text_embedding_client= client,
                    llm_client= groq_caller,
                    verbose=1,
                    existing_embeddings = embed_data,
                    use_existing_embeddings=True,
                )
                
                top_clusters = hercules.cluster(embed_data, topic_seed="Amazon product reviews")
                cluster_doc_path = export_top_clusters_txt(
                    top_clusters=top_clusters,
                    embedding_model=embedding_model,
                    reduction_method=embed_name,
                    clustering_level=level,
                    clustering_method=cluster_method,
                )
                print(f"Exported top cluster documentation to: {cluster_doc_path}")
                labels = hercules.get_level_assignments(level=1)[0]
                print(labels)
                
                
                
                
            # Use topic_dict for comparison
            available_levels = np.array(sorted(topic_dict.keys()))
            closest_level = min(available_levels, key=lambda k: abs(k - level))
            print(f"Ground truth: Using closest level {closest_level} (requested: {level})")

            topic_series = topic_dict[closest_level]
            valid_idx = ~pd.isna(topic_series)
            n_valid = valid_idx.sum()
            print(f"Valid samples for evaluation: {n_valid}/{len(topic_series)}")

            target_lst = topic_series[valid_idx]
            label_lst = labels[valid_idx]

            # Compute metrics
            try:
                fm_score = FowlkesMallows.Bk({level: target_lst}, {level: label_lst})[level]['FM']
            except:
                fm_score = np.nan  # In case of failure
                print(f"WARNING: FM score computation failed!")

            rand = rand_score(target_lst, label_lst)
            ari = adjusted_rand_score(target_lst, label_lst)
            
            print(f"Scores - FM: {fm_score:.4f}, Rand: {rand:.4f}, ARI: {ari:.4f}")
            
            scores_all[(embed_name, cluster_method)]["FM"].append(fm_score)
            scores_all[(embed_name, cluster_method)]["Rand"].append(rand)
            scores_all[(embed_name, cluster_method)]["ARI"].append(ari)

print(f"\n{'='*60}")
print("All clustering and evaluation complete!")
print(f"{'='*60}")



rows = []

for (embed_name, cluster_method), score_dict in scores_all.items():
    n_levels = len(score_dict["FM"])  # assuming all score lists have same length
    for i in range(n_levels):
            rows.append({
                "reduction_method": embed_name,
                "cluster_method": cluster_method,
                "level": cluster_levels[i],  # assumes scores were appended in order
                "FM": score_dict["FM"][i],
                "Rand": score_dict["Rand"][i],
                "ARI": score_dict["ARI"][i],
            })


# Create DataFrame
scores_df = pd.DataFrame(rows)

# Optional: sort for easier viewing
scores_df = scores_df.sort_values(by=["reduction_method", "cluster_method", "level"]).reset_index(drop=True)
write_header = not os.path.exists(f'{embedding_model}_results/other_amz_results.csv')
scores_df.to_csv(f"{embedding_model}_results/other_amz_results.csv",mode='a', index=False, header=write_header)


with open("combo_color_map.json", 'r') as file:
        combo_color_map = json.load(file)

import matplotlib.pyplot as plt

metrics = ['FM', 'Rand', 'ARI']

for metric in metrics:
    plt.figure(figsize=(10, 6))
    for (embed_name, method), metric_scores in scores_all.items():
        if method=="DC":
            method="Diffusion Condensation"
        combo_key = f"{embed_name}_{method}"
        plt.plot(
            cluster_levels, 
            metric_scores[metric], 
            marker='o', 
            label=f"{embed_name} {method}",
            color= combo_color_map.get(combo_key, 'black')
        )
    
    plt.title(f"{metric} Score Across Cluster Levels")
    plt.xlabel("Cluster Level")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

