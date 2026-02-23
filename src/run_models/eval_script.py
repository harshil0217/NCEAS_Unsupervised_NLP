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
import ast
import importlib
import re
import time
import warnings
import json
from filelock import FileLock

# ========================
# Data Manipulation
# ========================
import numpy as np
import pandas as pd

# ===============================
# Machine Learning & Clustering
# ===============================
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, fowlkes_mallows_score, rand_score
from sklearn.model_selection import ParameterGrid
from scipy.cluster.hierarchy import fcluster

# ===========================
# Dimensionality Reduction
# ===========================
import phate
from umap import UMAP
from bertopic.dimensionality import BaseDimensionalityReduction

# ===================
# Topic Modeling
# ===================
from bertopic import BERTopic

# ========================
# Graph-based Clustering
# ========================
from hdbscan import HDBSCAN

# =============================
# Diffusion Condensation
# =============================
from custom_packages.diffusion_condensation import DiffusionCondensation as dc

# ========================
# NLP & Transformers
# ========================
from sentence_transformers import SentenceTransformer

# ========================
# Parallel Processing
# ========================
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

# ========================
# Evaluation Metrics
# ========================
from custom_packages.fowlkes_mallows import FowlkesMallows


# ===================
# Global Config
# ===================
np.random.seed(42)
warnings.filterwarnings("ignore")
importlib.reload(phate)

# ===================
# Embedding Functions
# ===================
def get_embeddings(texts, backend="gpt", model="text-embedding-3-small"):
    if backend == "gpt":
        batch_size = 100
        embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Fetching GPT embeddings", unit="batch"):
            batch = texts[i : i + batch_size]
            response = client.embeddings.create(input=batch, model=model)
            batch_embeddings = [entry.embedding for entry in response.data]
            embeddings.extend(batch_embeddings)
        return embeddings

    elif backend == "sentence-transformers":
        model = SentenceTransformer(model, trust_remote_code=True)
        return model.encode(texts, show_progress_bar=True, convert_to_numpy=True).tolist()

    else:
        raise ValueError(f"Unsupported backend '{backend}'.")

# ===================
# Utility Functions
# ===================
def format_params(params):
    return "_".join(f"{k}{v}" for k, v in sorted(params.items()))

def compare_dicts(dict_str1, dict_str2):
    try:
        dict1 = ast.literal_eval(dict_str1)
        dict2 = ast.literal_eval(dict_str2)
        return dict1 == dict2
    except:
        return False
    
def compute_or_load_phate(data, path, reduction_params, wait_if_locked=True):
    lock_path = path + ".lock"

    # If embedding already exists, return it
    if os.path.exists(path):
        return np.load(path)

    # If lock exists
    if os.path.exists(lock_path):
        if wait_if_locked:
            # Wait until the file is created by another process
            while os.path.exists(lock_path):
                time.sleep(5)
                if os.path.exists(path):
                    return np.load(path)
        else:
            # Skip computation for now
            print(f"PHATE embedding for {path} is currently locked. Skipping.")
            return None

    try:
        # Double-check again in case file was created while entering the try block
        if os.path.exists(path):
            return np.load(path)
        
        # Place lock
        with open(lock_path, "w") as f:
            f.write("locked")

        # Compute embedding
        print("Computing PHATE")
        reducer_model = phate.PHATE(n_jobs=-2, random_state=42, n_pca=None, **reduction_params)
        embed = reducer_model.fit_transform(data)
        np.save(path, embed)
        return embed
    finally:
        # Clean up lock
        if os.path.exists(lock_path):
            os.remove(lock_path)


# ===================
# Main BERTopic Runner
# ===================
def run_bertopic(reduction_method, cluster_method, reduction_params, cluster_params):
    results = []
    bertopic_file = f'{embedding_model}_results/results_all_methods_{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_noise{add_noise}_{branching}.csv'

    if os.path.exists(bertopic_file):
        check_df = pd.read_csv(bertopic_file)
        check_df['reduction_method'] = check_df['reduction_method'].fillna('None')
        row_exists = ((check_df['reduction_method'].astype(str) == str(reduction_method)) &
                      (check_df['cluster_method'].astype(str) == str(cluster_method)) &
                      (check_df['reduction_params'].astype(str) ==str(reduction_params))&
                      (check_df['cluster_params'].astype(str)== str(cluster_params))).any()
                    #   check_df['reduction_params'].apply(lambda x: compare_dicts(x, str(reduction_params))) &
                    #   check_df['cluster_params'].apply(lambda x: compare_dicts(x, str(cluster_params)))).any()
        if row_exists:
            return

    if reduction_method == 'UMAP':
        reducer = UMAP(random_state=42, **reduction_params)
    elif reduction_method == 'PCA':
        reducer = PCA(random_state=42, **reduction_params)
    elif reduction_method == 'PHATE':
        param_str = format_params(reduction_params)
        if float(add_noise)>0:
            phate_path = f"{embedding_model}_reduced_embeddings/phate_embedding_{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_noise{add_noise}_{branching}_{param_str}.npy"
        else:
            phate_path = f"{embedding_model}_reduced_embeddings/phate_embedding_{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_{branching}_{param_str}.npy"

        embed_phate = compute_or_load_phate(data, phate_path, reduction_params,wait_if_locked=wait)
        reducer = BaseDimensionalityReduction()

        if embed_phate is None:
            return

    elif reduction_method == "BASE-PHATE":
        p_base = phate.PHATE(n_jobs=-2, random_state=42, n_pca=None, **reduction_params)
        p_base.fit(data)
        embed_phate = p_base.diff_potential
        reducer = BaseDimensionalityReduction()
    else:
        reducer = BaseDimensionalityReduction()
    for i in topic_dict.keys():
        if cluster_method == 'HDBSCAN':
            cluster_model = HDBSCAN(**cluster_params)
        elif cluster_method == 'Diffusion Condensation':
            cluster_model = dc(min_clusters=i, max_iterations=5000, **cluster_params)
        else:
            cluster_model = AgglomerativeClustering(n_clusters=i, **cluster_params)

        topic_model = BERTopic(hdbscan_model=cluster_model, umap_model=reducer)

        embeddings_to_use = embed_phate if reduction_method in ["PHATE", "BASE-PHATE"] else data
        topics, _ = topic_model.fit_transform(documents=topic_data['topic'], embeddings=embeddings_to_use)

        if cluster_method == 'HDBSCAN':
            Z = cluster_model.single_linkage_tree_.to_numpy()
            labels = fcluster(Z, i, criterion='maxclust')
            labels[labels == -1] = labels.max() + 1
        else:
            labels = np.array(topics)

        NA_mask = pd.isna(topic_dict[i])
        target_lst = topic_dict[i][~NA_mask]
        bertopic_lst = labels[~NA_mask]

        fm = FowlkesMallows.Bk({i: target_lst}, {i: bertopic_lst})
        rand = rand_score(target_lst, bertopic_lst)
        ari = adjusted_rand_score(target_lst, bertopic_lst)

        embed = (
            topic_model.umap_model.embedding_ if reduction_method == 'UMAP' else
            topic_model.umap_model.components_ if reduction_method == 'PCA' else
            embed_phate if reduction_method in ['PHATE', 'BASE-PHATE'] else None
        )
        param_str = format_params(reduction_params)
        cluster_file_name = f"{embedding_model}_clusterings/clustering{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_noise{add_noise}_{branching}_{param_str}_{reduction_method}_{cluster_method}_level_{i}.npy"
        np.save(cluster_file_name, labels)

        if embed is not None:
            embed_file_name = f"{embedding_model}_reduced_embeddings/{reduction_method}_embedding_{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_noise{add_noise}_{branching}_{param_str}.npy"
            np.save(embed_file_name, embed)

        results.append({
            'reduction_method': reduction_method,
            'cluster_method': cluster_method,
            'level': i,
            'reduction_params': reduction_params,
            'cluster_params': cluster_params,
            'FM': fm[i]['FM'],
            'E_FM': fm[i]['E_FM'],
            'V_FM': fm[i]['V_FM'],
            'Rand': rand,
            'ARI': ari,
            'topics': topics,
            'idx': shuffle_idx,
            'embed': embed
        })

    return results

# ====================
# Setup & Execution
# ====================
def noise_range(value):
    f = float(value)
    if f < 0 or f > 1:
        raise argparse.ArgumentTypeError("add_noise must be a float between 0 and 1.")
    return f

from openai import OpenAI
key = os.getenv('GPT_API_KEY')

client = OpenAI(api_key=key)
import argparse

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

wait = args.wait == "True"  # Converts "True" to True, "False" to False



filename = f'data_generation/generated_data/{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_noise{add_noise}_{branching}.csv'


print(f"Waiting for {filename} to be created...")
while not os.path.exists(filename): time.sleep(1)
print(f"{filename} detected! Reading file...")

topic_data = pd.read_csv(filename)
num_seed_topics = len(topic_data['category 0'].unique())
os.makedirs('gpt_embeddings', exist_ok=True)

# Embeddings
backend = 'gpt'
embedding_model = "text-embedding-3-large"
if float(add_noise)>0:
    embed_file = f'gpt_embeddings/{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_noise{add_noise}_{embedding_model}_embed.npy'
else:
    embed_file = f'gpt_embeddings/{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_{embedding_model}_embed.npy'


if not os.path.exists(embed_file):
    embedding_list = get_embeddings(topic_data['topic'], backend=backend, model=embedding_model)
    
    np.save(embed_file, embedding_list)
else:
    embedding_list = np.load(embed_file)

shuffle_idx = np.random.RandomState(seed=42).permutation(len(topic_data))
topic_data = topic_data.iloc[shuffle_idx].reset_index(drop=True)
data = np.array(embedding_list)[shuffle_idx]
reverse_idx = np.argsort(shuffle_idx)

# Topic hierarchy dictionary
topic_dict = {
    len(topic_data[col].unique()): np.array(topic_data[col])
    for col in topic_data.columns if re.match(r'^category \d+$', col)
}

# Search space
reduction_methods = ['UMAP', 'PCA', 'PHATE'] #'None', 'BASE-PHATE'
cluster_methods = ['Diffusion Condensation', 'HDBSCAN', 'Agglomerative']

umap_params = {'n_components': [300],'min_dist':[.05,.1],'n_neighbors':[10,25]}
pca_params = {'n_components': [300]} 
phate_params = {'n_components': [300], 'decay': [10,20], 't': [7,'auto']}
base_phate_params = {'t': ['auto', 7, 11]}
hdbscan_params = {'cluster_selection_epsilon': [0], 'cluster_selection_method': ['eom']} 
agg_params = {'linkage': ['ward']}
diffusion_params = {'k': [10], 't': [3], 'alpha': [4], 'bandwidth_norm': ['max']}


param_list = []
for r_method in reduction_methods:
    for c_method in cluster_methods:
        r_grid = (
            umap_params if r_method == 'UMAP' else
            pca_params if r_method == 'PCA' else
            phate_params if r_method == 'PHATE' else
            base_phate_params if r_method == 'BASE-PHATE' else [{}]
        )
        c_grid = (
            hdbscan_params if c_method == 'HDBSCAN' else
            agg_params if c_method == 'Agglomerative' else
            diffusion_params
        )
        for r_params in ParameterGrid(r_grid) if r_grid else [{}]:
            for c_params in ParameterGrid(c_grid):
                param_list.append((r_method, c_method, r_params, c_params))

os.makedirs(f'{embedding_model}_results', exist_ok=True)

bertopic_file = f'{embedding_model}_results/results_all_methods_{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_noise{add_noise}_{branching}.csv'

lock_file = bertopic_file + '.lock'  # lock file to avoid write conflicts
os.makedirs(f'{embedding_model}_reduced_embeddings', exist_ok=True)
os.makedirs(f'{embedding_model}_clusterings', exist_ok=True)
# Function that safely appends to CSV
def safe_run_and_append(params, file_path, lock_path):
    try:
        result = run_bertopic(*params)
        if isinstance(result, list):
            result = [item for item in result if isinstance(item, dict)]
        elif isinstance(result, dict):
            result = [result]
        else:
            result = []

        if result:
            df = pd.DataFrame(result)
            with FileLock(lock_path):
                # Only write header if file does not exist
                write_header = not os.path.exists(file_path)
                df.to_csv(file_path, mode='a', index=False, header=write_header)
        return result
    except Exception as e:
        print(f"Error with params {params}: {e}")
        return []

# Run parallel jobs and append results as they finish
with tqdm_joblib(tqdm(desc="Processing", total=len(param_list))):
    with Parallel(n_jobs=-4, backend="loky") as parallel:
        all_results = parallel(delayed(safe_run_and_append)(
            params,
            file_path=bertopic_file,
            lock_path=lock_file
        ) for params in param_list)