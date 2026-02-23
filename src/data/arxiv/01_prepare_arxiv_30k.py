import os
import json
import pandas as pd
from tqdm import tqdm
import numpy as np

# ----------------------------
# CONFIG
# ----------------------------

# Update this path if needed
JSON_PATH = os.path.expanduser(
    "~/.cache/kagglehub/datasets/Cornell-University/arxiv/versions/274/arxiv-metadata-oai-snapshot.json"
)

OUTPUT_DIR = "src/data/arxiv"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_FILE = os.path.join(OUTPUT_DIR, "arxiv_30k_clean.csv")

np.random.seed(42)

# ----------------------------
# STREAM DATA
# ----------------------------

print("Reading arXiv JSON...")

records = []

with open(JSON_PATH, "r") as f:
    for line in tqdm(f):
        paper = json.loads(line)

        if paper["categories"].startswith(("cs.", "physics.")):
            records.append({
                "text": paper["title"] + " " + paper["abstract"],
                "categories": paper["categories"]
            })

print("Total filtered papers:", len(records))

# ----------------------------
# CONVERT TO DATAFRAME
# ----------------------------

df = pd.DataFrame(records)

df = df.sample(30000, random_state=42).reset_index(drop=True)

df["label"] = df["categories"].str.split().str[0]

df = df[["text", "label"]]

# ----------------------------
# SAVE
# ----------------------------

df.to_csv(OUTPUT_FILE, index=False)

print("Saved:", OUTPUT_FILE)
print("Final dataset shape:", df.shape)