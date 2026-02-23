import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import torch
import os

# =====================
# Config
# =====================
DATA_PATH = "src/data/arxiv/arxiv_30k_clean.csv"
OUTPUT_PATH = "src/data/arxiv/arxiv_qwen_embeddings.npy"
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

# =====================
# Load Data
# =====================
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
texts = df["text"].astype(str).tolist()
print(f"Loaded {len(texts)} documents")

# =====================
# Load Model
# =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = SentenceTransformer(MODEL_NAME, device=device)

# =====================
# Generate Embeddings
# =====================
print("Generating embeddings...")
embeddings = model.encode(
    texts,
    batch_size=32,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("Embedding shape:", embeddings.shape)

# =====================
# Save
# =====================
np.save(OUTPUT_PATH, embeddings)
print(f"Saved embeddings to {OUTPUT_PATH}")