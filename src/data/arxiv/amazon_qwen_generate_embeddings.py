import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# ----------------------------
# Load Data
# ----------------------------
csv_file = "arxiv_30k_clean.csv"
df = pd.read_csv(csv_file)

texts = df["topic"].tolist()
print("Total texts:", len(texts))

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
    batch_size=24,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("Embedding shape:", embedding_array.shape)

# ----------------------------
# Save
# ----------------------------
np.save("arxiv_qwen_embeddings.npy", embedding_array)
df.to_csv("arxiv_metadata.csv", index=False)

print("arXiv embeddings saved successfully.")
