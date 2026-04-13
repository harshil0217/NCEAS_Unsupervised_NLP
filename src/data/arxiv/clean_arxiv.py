import os
import sys
import pandas as pd
import re
import warnings
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, rand_score

# Move to repo root
target_folder = "NCEAS_Unsupervised_NLP"
current_dir = os.getcwd()

while os.path.basename(current_dir) != target_folder:
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    if parent_dir == current_dir:
        raise FileNotFoundError(f"{target_folder} not found.")
    current_dir = parent_dir

os.chdir(current_dir)

# Add repo root
sys.path.insert(0, current_dir)

# Add src so custom_packages works
sys.path.insert(0, os.path.join(current_dir, "src"))

warnings.filterwarnings("ignore")
np.random.seed(42)

# Load arXiv Dataset
df = pd.read_csv("src/data/arxiv/arxiv_30k_clean.csv")
df_new = df.rename(columns={"text": "topic"})
df_new["category_0"] = df_new["label"].apply(lambda x: x.split(".")[0])
df_new["category_1"] = df_new["label"].apply(lambda x: x.split(".")[1])
df_new = df_new[["topic", "category_0", "category_1"]]

df_new = df_new.dropna().reset_index(drop=True)

df_new = df_new[
    df_new["topic"].apply(lambda x: isinstance(x, str) and x.strip() != "")
].reset_index(drop=True)

df_new.to_csv("src/data/arxiv/arxiv_clean.csv", index=False)

print(df_new.shape)
