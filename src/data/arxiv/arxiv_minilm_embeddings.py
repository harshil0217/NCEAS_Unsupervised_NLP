import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ----------------------------
# Load Data
# ----------------------------
df = pd.read_csv("arxiv_30k_clean.csv")
texts = df["text"].astype(str).tolist()

# ----------------------------
# Load MiniLM Model
# ----------------------------
model_name = "all-MiniLM-L6-v2"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

model = SentenceTransformer(model_name, device=device)

# ----------------------------
# Generate Embeddings
# ----------------------------
embedding_array = model.encode(
    texts,
    batch_size=64,  # MiniLM can handle larger batch size
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("Embedding shape:", embedding_array.shape)

# ----------------------------
# Save
# ----------------------------
np.save("arxiv_minilm_embeddings.npy", embedding_array)

print("Done.")