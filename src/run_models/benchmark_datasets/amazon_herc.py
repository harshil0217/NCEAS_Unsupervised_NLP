


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

print("Amazon HERC dataset loaded and preprocessed successfully.")


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
    
    
embedding_model_names = [
    "Qwen/Qwen3-Embedding-0.6B",
    "sentence-transformers/all-MiniLM-L6-v2",
]

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


scores_all = defaultdict(lambda: defaultdict(list))




def get_sentence_transformer_embeddings(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Embeds text using a specified SentenceTransformer model.
    """
    if not texts:
        return np.array([])

    model = SentenceTransformer(model_name)

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
        return dummy_text_embedding(texts)


rep_mode = 'direct'
for model_name in embedding_model_names:
    torch.cuda.empty_cache()

    combo_scores = {"FM": [], "Rand": [], "ARI": [], "AMI": []}
    
    hercules = Hercules(
    level_cluster_counts=cluster_levels,
    representation_mode=rep_mode,
    text_embedding_client=get_sentence_transformer_embeddings,
    llm_client=qwen_caller,
    verbose=2,
    llm_initial_batch_size=32
    )
    
    # 3. Run clustering
    top_clusters = hercules.cluster(amz['topic'].tolist(), topic_seed="amazon reviews")


    #get labels

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

    scores_all[model_name] = [fm_score, rand, ari, ami]
    
# Convert scores_all to a DataFrame and save to csv
scores_df = pd.DataFrame.from_dict(scores_all, orient = 'index')
scores_df.columns = ['Embedding Model Name', 'FM', 'Rand', 'ARI', 'AMI']

scores_df = scores_df.explode(['FM', 'Rand', 'ARI', 'AMI'])
scores_df['Level'] = cluster_levels * 2
scores_df = scores_df.reset_index(drop=True)
scores_df.to_csv(f"results/amazon_herc_scores_{rep_mode}.csv", index=False)
    
    
    