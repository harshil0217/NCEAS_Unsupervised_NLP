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

# Change the working directory to the folder where "src" is found
os.chdir(current_dir)

# Add 'src' directory to sys.path
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
import argparse

torch.cuda.empty_cache()

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



# ========================
# Clustering
# ========================
from pyhercules import Hercules
from transformers import AutoModelForCausalLM, AutoTokenizer



# ======================
# Evaluation Metrics
# ======================
from custom_packages.fowlkes_mallows import FowlkesMallows
from sklearn.metrics import adjusted_rand_score, rand_score, adjusted_mutual_info_score


from tqdm import tqdm
from joblib import Parallel, delayed
# ==============
# Global Config
# ==============
np.random.seed(42)
warnings.filterwarnings("ignore")


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
    arx = pd.read_csv("data/arxiv/arxiv_raw_30k.csv")
    arx["topic"] = arx["text"]
    arx["category_0"] = arx["label"].apply(lambda x: x.split(".")[0])
    arx["category_1"] = arx["label"].apply(lambda x: x.split(".")[1])

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
    wos = pd.read_excel('data/WebOfScience/Data.xlsx')

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
        "results_filename": "amazon_herc_clustering_scores.csv",
        "topic_seed": "Amazon Product Reviews",
        "batch_size": 32,
    },
    "dbpedia": {
        "load_function": load_dbpedia,
        "depth": 3,
        "short": "db",
        "results_filename": "db_herc_clustering_scores.csv",
        "topic_seed": 'Wikipedia Summaries',
        "batch_size": 32,
    },
    "arxiv": {
        "load_function": load_arxiv,
        "depth": 2, 
        "short": "arx",
        "results_filename": "arxiv_herc_clustering_scores.csv",
        "topic_seed": "arxiv abstracts",
        "batch_size": 32,
    },
    "rcv1": {
        "load_function": load_rcv1,
        "depth": 2,
        "short": "rcv1",
        "results_filename": "rcv1_herc_clustering_scores.csv",
        "topic_seed": "reuters headlines",
        "batch_size": 8,
    },
    "wos": {
        "load_function": load_wos,
        "depth": 2,
        "short": "wos",
        "results_filename": "wos_herc_clustering_scores.csv",
        "topic_seed": "web of science articles",
        "batch_size": 64,
    },
}


# =====================================
# Core Pipeline Functions
# =====================================

model_name = "Qwen/Qwen3-4B-Instruct-2507"

tokenizer = AutoTokenizer.from_pretrained(model_name)
# Define the 4-bit configuration

model = AutoModelForCausalLM.from_pretrained(model_name,torch_dtype="auto", device_map = 'auto')
print(model.hf_device_map)


def qwen_caller(prompt: str) -> str:
    """
    generates text using the Qwen model and returns the generated text.
    """
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant providing concise summaries."},
        {"role": "user", "content": prompt}
    ]
    
    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize = False)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    generated_ids = model.generate(**inputs, 
                                   max_new_tokens=16384)
    
    output_ids = generated_ids[0][len(inputs["input_ids"][0]):].tolist() # Get only the generated part
    
    
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")
    
    print(content)
    return content

def get_sentence_transformer_embeddings(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Embeds text using a specified SentenceTransformer model.
    """
    if not texts:
        return np.array([])

    model = SentenceTransformer(model_name, device=device)

    try:
        # encode() handles the list input and returns a numpy array by default
        embeddings = model.encode(
            texts, 
            convert_to_numpy=True, 
            show_progress_bar=False
        )
        
        # Ensure output is 2D even for single strings
        if embeddings.ndim == 1:
            embeddings = embeddings[np.newaxis, :]
            
        return embeddings

    except Exception as e:
        print(f"Error during embedding generation with '{model_name}': {e}")
    


def run_pipeline(dataset_name, rep_mode):


    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")

    config = DATASET_CONFIGS[dataset_name]


    print(f"\n{'='*80}")
    print(f"Running pipeline for dataset: {dataset_name.upper()}")
    print(f"{'='*80}\n")

    embedding_model_names = [
        "Qwen/Qwen3-Embedding-0.6B",
        "sentence-transformers/all-MiniLM-L6-v2",
    ]

    data = config['load_function']()

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
    depth = config['depth']
    print(f"Depth: {depth}")
    print(f"Building cluster levels by counting unique categories at each level...\n")

    cluster_levels = []
    for i in reversed(range(0, depth)):
        unique_count = len(topic_data[f'category_{i}'].unique())
        print(f"Level {i} (category_{i}): {unique_count} unique categories")
        cluster_levels.append(unique_count)

    print(f"\nFinal cluster_levels (from deepest to shallowest): {cluster_levels}\n")


    scores_all = defaultdict(lambda: defaultdict(list))


    for model_name in embedding_model_names:
        save_path = f"../../hercules_run/{config['short']}/{rep_mode}"
        hercules = None

        if os.path.exists(save_path):
            print(f"Loading existing model from {save_path}...")
            hercules = Hercules.load_model(filepath=save_path, 
                                            text_embedding_client=get_sentence_transformer_embeddings,
                                            llm_client = quen_caller)[0]
        else:
            torch.cuda.empty_cache()
            hercules = Hercules(
                level_cluster_counts=cluster_levels,
                representation_mode=rep_mode,
                text_embedding_client=get_sentence_transformer_embeddings,
                llm_client=qwen_caller,
                verbose=2,
                llm_initial_batch_size=32
            )
            
            # 3. Run clustering
            top_clusters = hercules.cluster(data['topic'].tolist(), topic_seed=config['topic_seed'])


            # Save model
            hercules.save_model(filepath=save_path, top_clusters=top_clusters)

        #get labels
        
        cluster_df = hercules.get_cluster_membership_dataframe(include_l0_details=data['topic'].tolist())

        print("printing number of l1 labels")
        print(np.unique(cluster_df['L1_cluster_title']))

        print("printing number of l2 labels")
        print(np.unique(cluster_df['L2_cluster_title']))

        os.makedirs(os.path.dirname(f"results/cluster_assignments/{config['short']}_{rep_mode}.csv"), exist_ok=True)
        cluster_df.to_csv(f"results/cluster_assignments/{config['short']}_{rep_mode}.csv", index = False)

        for i, cluster_level in enumerate(cluster_levels):
            labels = hercules.get_level_assignments(level=i+1)[0]
            print(labels)

            topic_series = topic_dict[cluster_level]
            valid_idx = ~pd.isna(topic_series)
            target_lst = topic_series[valid_idx]
            label_lst = labels[valid_idx]

            try:
                fm_score = FowlkesMallows.Bk({cluster_level: target_lst}, {cluster_level: label_lst})[cluster_level]['FM']
            except Exception:
                fm_score = np.nan
                print("WARNING: FM score computation failed!")

            rand = rand_score(target_lst, label_lst)
            ari = adjusted_rand_score(target_lst, label_lst)
            print(f"Scores - FM: {fm_score:.4f}, Rand: {rand:.4f}, ARI: {ari:.4f}")
            ami = adjusted_mutual_info_score(target_lst, label_lst)

        scores_all[model_name] = [np.mean(fm_score), np.mean(rand), np.mean(ari), np.mean(ami)]
        
    # Convert scores_all to a DataFrame and save to csv
    scores_df = pd.DataFrame.from_dict(scores_all, orient = 'index')
    scores_df.reset_index(inplace=True)
    scores_df.columns = ['Embedding Model Name', 'FM', 'Rand', 'ARI', 'AMI']
    results_filename = config['results_filename'].replace('.csv', f'_{rep_mode}.csv')
    scores_df.to_csv(f"results/{results_filename}", index=False)

def main():
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
            python herc_pipeline.py --dataset dbpedia --rep_mode direct
            python herc_pipeline.py --dataset amazon --rep_mode description
                    """
                )

    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        choices=list(DATASET_CONFIGS.keys()),
        help='Dataset to process'
    )

    parser.add_argument(
        '--rep_mode',
        type=str,
        required=True,
        choices=['direct', 'description'],
        help='Hercules representation mode: "direct" uses raw text, "description" uses LLM-generated cluster descriptions'
    )

    args = parser.parse_args()

    # Run pipeline for specified dataset
    run_pipeline(args.dataset, args.rep_mode)


if __name__ == "__main__":
    main()
