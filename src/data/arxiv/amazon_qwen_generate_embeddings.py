import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# ----------------------------
# Load Data
# ----------------------------
df = pd.read_csv("amz_data.csv")
texts = df["topic"].tolist()

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
np.save("amazon_qwen_embeddings.npy", embedding_array)
df.to_csv("amazon_metadata.csv", index=False)

print("Amazon embeddings saved successfully.")
