import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# ----------------------------
# Load Data
# ----------------------------
df = pd.read_csv("data/arxiv/arxiv_30k.csv")
texts = df["topic"].tolist()

# ----------------------------
# Load Qwen Model
# ----------------------------
model_name = "Qwen/Qwen3-Embedding-0.6B"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = model.to(device)
model.eval()

# ----------------------------
# Embedding Function
# ----------------------------
def embed_texts(texts, batch_size=64, max_length=512):
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]

        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )

        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        embeddings = outputs.last_hidden_state.mean(dim=1)
        all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)

# ----------------------------
# Generate Embeddings
# ----------------------------
embedding_array = embed_texts(texts)

print("Embedding shape:", embedding_array.shape)

# ----------------------------
# Save
# ----------------------------
np.save("arxiv_qwen_embeddings.npy", embedding_array)

print("Saved arxiv_qwen_embeddings.npy successfully.")
