import os
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ----------------------------
# Check files
# ----------------------------
print("Current directory:", os.getcwd())
print("Files in directory:", os.listdir())

# ----------------------------
# Load Data
# ----------------------------
csv_file = "arxiv_30k_clean.csv"

if not os.path.exists(csv_file):
    raise FileNotFoundError(f"{csv_file} not found in current directory.")

df = pd.read_csv(csv_file)
texts = df["text"].astype(str).tolist()

print("Loaded documents:", len(texts))

# ----------------------------
# Load Qwen Model
# ----------------------------
model_name = "Qwen/Qwen3-Embedding-0.6B"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

model = SentenceTransformer(model_name, device=device)

# ----------------------------
# Generate Embeddings
# ----------------------------
embedding_array = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("Embedding shape:", embedding_array.shape)

# ----------------------------
# Save
# ----------------------------
np.save("arxiv_qwen_embeddings.npy", embedding_array)
df.to_csv("arxiv_30k_metadata.csv", index=False)

print("ArXiv Qwen embeddings saved successfully.")
