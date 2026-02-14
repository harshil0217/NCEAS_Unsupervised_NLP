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
import json
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# =========================
# 1. Load & Sample Dataset
# =========================

file_path = "arxiv-metadata-oai-snapshot.json"

records = []

with open(file_path, "r") as f:
    for line in tqdm(f):
        paper = json.loads(line)
        if paper["categories"].startswith(("cs.", "physics.")):
            records.append({
                "topic": paper["title"] + " " + paper["abstract"],
                "categories": paper["categories"]
            })

df_arxiv = pd.DataFrame(records)

# Sample 30k
df_arxiv = df_arxiv.sample(30000, random_state=42).reset_index(drop=True)

# =========================
# 2. Extract Hierarchy
# =========================

def extract_categories(cat_string):
    primary = cat_string.split()[0]
    top_level = primary.split('.')[0]
    return top_level, primary

df_arxiv[["category_0", "category_1"]] = df_arxiv["categories"].apply(
    lambda x: pd.Series(extract_categories(x))
)

# =========================
# 3. Load Qwen Model
# =========================

model_name = "Qwen/Qwen3-Embedding-0.6B"
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Using device:", device)

model = SentenceTransformer(model_name, device=device)

# =========================
# 4. Generate Embeddings
# =========================

embedding_array = model.encode(
    df_arxiv["topic"].tolist(),
    batch_size=64,  # GPU can handle this
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True
)

print("Embedding shape:", embedding_array.shape)

# =========================
# 5. Save
# =========================

np.save("arxiv_qwen_embeddings.npy", embedding_array)
df_arxiv.to_csv("arxiv_30k_metadata.csv", index=False)

print("Done.")
