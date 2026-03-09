# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an NCEAS (National Center for Ecological Analysis and Synthesis) research project focused on unsupervised NLP analysis for US fisheries discourse and narratives. The project evaluates dimensionality reduction methods (particularly PHATE) combined with various clustering algorithms on textual embeddings to advance hierarchical topic modeling.

**Key Research Goal**: Compare dimensionality reduction techniques (PHATE, PCA, UMAP, t-SNE, PaCMAP, TriMAP) with clustering methods (Agglomerative, HDBSCAN, Diffusion Condensation) using evaluation metrics (Fowlkes-Mallows, ARI, Rand, AMI).

## Environment Setup

The project uses **conda** for environment management. Always work within the conda environment:

```bash
# Create environment
conda env create --prefix ./envs --file environment.yml
conda activate ./envs

# Or activate existing .venv (Python 3.10)
source .venv/bin/activate
```

**Important**: This project requires Python 3.10 and uses both conda (environment.yml) and pip (requirements.txt) for dependencies.

## Running Scripts

### Key Entry Points

All Python scripts in `src/` expect to be run from the repository root and use a common pattern to navigate to the `src/` directory:

```python
# Standard pattern in most scripts
target_folder = "src"
current_dir = os.getcwd()
while os.path.basename(current_dir) != target_folder:
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    current_dir = parent_dir
os.chdir(current_dir)
sys.path.insert(0, current_dir)
```

### Running Benchmark Experiments

**Amazon Benchmark** (benchmark dataset):
```bash
cd /home/harshil0216/NCEAS_Unsupervised_NLP
python src/run_models/benchmark_datasets/amazon.py
```

**Evaluation Script** (primary evaluation pipeline):
```bash
python src/run_models/eval_script.py \
  --theme <theme_name> \
  --t <float> \
  --max_sub <int> \
  --depth <int> \
  --synonyms <int> \
  --branching <constant|decreasing|increasing|random> \
  --add_noise <0.0-1.0> \
  --wait <True|False>
```

### Generating Synthetic Data

```bash
python src/data_generation/generate.py \
  --theme <theme> \
  --t <float> \
  --max_sub <int> \
  --depth <int> \
  --synonyms <int> \
  --branching <strategy>
```

This uses the Groq API to generate hierarchical topic structures for testing.

## Architecture

### Directory Structure

```
src/
├── custom_packages/         # Custom implementations
│   ├── diffusion_condensation.py   # Diffusion Condensation clustering
│   ├── hercules.py                 # HERCULES hierarchical clustering
│   ├── fowlkes_mallows.py          # FM index implementation
│   ├── clusters.py                 # Clustering utilities
│   └── hierarchical_kmeans_gpu.py  # GPU-accelerated k-means
├── data_generation/         # Synthetic data generation
│   └── generate.py          # LLM-based hierarchy generation
├── run_models/              # Experiment runners
│   ├── eval_script.py       # Main evaluation pipeline
│   └── benchmark_datasets/  # Benchmark experiments
│       ├── amazon.py        # Amazon dataset benchmark
│       └── amazon_herc.py   # Amazon with HERCULES
├── data/                    # Datasets (not in git)
│   ├── amazon/              # Amazon product categories
│   ├── dbpedia/             # DBpedia topics
│   └── WebOfScience/        # WoS documents
├── gpt_embeddings/          # Cached OpenAI embeddings
├── <model>_reduced_embeddings/  # Dimensionality reduction outputs
├── <model>_clusterings/     # Clustering results
└── <model>_results/         # Evaluation metrics (CSV files)
```

### Core Pipeline Flow

1. **Data Loading** → Load hierarchical text data with ground truth categories
2. **Embedding Generation** → Create embeddings using OpenAI API or SentenceTransformers
3. **Shuffling** → Shuffle data with a fixed random seed (42) for reproducibility
4. **Dimensionality Reduction** → Apply PHATE/PCA/UMAP/t-SNE/PaCMAP/TriMAP
5. **Clustering** → Apply Agglomerative/HDBSCAN/Diffusion Condensation at multiple hierarchy levels
6. **Evaluation** → Compute FM, Rand, ARI, AMI metrics against ground truth
7. **Results Storage** → Save to CSV files and numpy arrays

### Key Custom Packages

**Diffusion Condensation** (`custom_packages/diffusion_condensation.py`):
- Novel clustering algorithm that iteratively merges data points through diffusion
- Uses k-NN graph with configurable parameters: k, alpha, t, merge_threshold
- Returns cluster labels via `.labels_` attribute

**HERCULES** (`custom_packages/hercules.py`):
- Hierarchical Embedding-based Recursive Clustering Using LLMs for Efficient Summarization
- Performs hierarchical k-means with LLM-generated cluster summaries
- Supports text, numeric, and image data

**Fowlkes-Mallows** (`custom_packages/fowlkes_mallows.py`):
- Custom implementation of FM index for hierarchical clustering evaluation
- Returns FM score with expected value (E_FM) and variance (V_FM)
- Use via `FowlkesMallows.Bk(ground_truth_dict, predicted_dict)`

### Parallel Processing Pattern

The codebase uses **joblib with threading backend** for parallel processing:

```python
from joblib import Parallel, delayed
from tqdm_joblib import tqdm_joblib

with tqdm_joblib(tqdm(desc="Processing", total=len(items))):
    with Parallel(n_jobs=-4, backend="threading") as parallel:
        results = parallel(delayed(process_fn)(item) for item in items)
```

**Note**: Use `backend="threading"` (not "loky") to avoid memory issues with large embeddings.

### File Locking Pattern

For concurrent writes to shared files (CSV results, numpy embeddings):

```python
from filelock import FileLock

lock_file = output_path + '.lock'
with FileLock(lock_file):
    write_header = not os.path.exists(output_path)
    df.to_csv(output_path, mode='a', index=False, header=write_header)
```

### PHATE Embedding Caching

PHATE computations are expensive and cached with lock files:

```python
phate_path = f"{embedding_model}_reduced_embeddings/phate_embedding_{params}.npy"
embed_phate = compute_or_load_phate(data, phate_path, reduction_params, wait_if_locked=True)
```

If a `.lock` file exists, the process waits (if `wait=True`) or skips (if `wait=False`).

## API Keys and Environment Variables

The project requires API keys stored in `.env` (not in version control):

```bash
GPT_API_KEY=<your_openai_key>
GROQ_API_KEY=<your_groq_key>
```

Scripts use `python-dotenv` to load these:
```python
from dotenv import load_dotenv
load_dotenv()
key = os.getenv('GPT_API_KEY')
```

## Data Conventions

### Path Requirements
- **All file paths must be relative** (never absolute)
- Data files are stored in `src/data/` but are NOT committed to git
- Example data is provided for testing; partner data is under NDA

### Dataset Structure
CSV files must have:
- `topic` column: Text to embed
- `category_0`, `category_1`, ..., `category_N`: Hierarchical ground truth labels

Example:
```csv
topic,category_0,category_1,category_2
"Product title","Electronics","Computers","Laptops"
```

### Shuffling Pattern
Always shuffle with seed=42 and maintain reverse index:
```python
shuffle_idx = np.random.RandomState(seed=42).permutation(len(data))
topic_data = topic_data.iloc[shuffle_idx].reset_index(drop=True)
embeddings = embeddings[shuffle_idx]
reverse_idx = np.argsort(shuffle_idx)  # To restore original order
```

## Evaluation Metrics

The project evaluates clustering quality using:

- **Fowlkes-Mallows (FM)**: Custom implementation that returns score, expected value, and variance
- **Adjusted Rand Index (ARI)**: `sklearn.metrics.adjusted_rand_score`
- **Rand Index**: `sklearn.metrics.rand_score`
- **Adjusted Mutual Information (AMI)**: `sklearn.metrics.adjusted_mutual_info_score`

Results are stored as CSV files with columns:
```
reduction_method, cluster_method, level, reduction_params, cluster_params, FM, E_FM, V_FM, Rand, ARI, AMI
```

## Common Patterns

### Topic Dictionary Creation
```python
import re
topic_dict = {}
for col in df.columns:
    if re.match(r'^category_\d+$', col):
        unique_count = len(df[col].unique())
        topic_dict[unique_count] = np.array(df[col])
```

### Checking for Existing Results
```python
if os.path.exists(results_file):
    check_df = pd.read_csv(results_file)
    row_exists = ((check_df['reduction_method'] == reduction_method) &
                  (check_df['cluster_method'] == cluster_method) &
                  (check_df['reduction_params'] == str(reduction_params))).any()
    if row_exists:
        return  # Skip computation
```

### Handling NaN in Evaluation
```python
NA_mask = pd.isna(ground_truth)
ground_truth_clean = ground_truth[~NA_mask]
predictions_clean = predictions[~NA_mask]
fm_score = FowlkesMallows.Bk({level: ground_truth_clean}, {level: predictions_clean})
```

## GPU Usage

The codebase supports CUDA for embeddings:
```python
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer(model_name, device=device)
```

GPU-accelerated k-means is available via `custom_packages/hierarchical_kmeans_gpu.py`.

## Notebooks

Interactive analysis notebooks are in `notebooks/`:
- `demo.ipynb`: Demonstration notebook showing the full pipeline

Start Jupyter from the repository root:
```bash
jupyter notebook
```

## Important Notes

- **Working Directory**: Scripts navigate to `src/` directory at runtime; run from repo root
- **Random Seed**: Always use seed=42 for reproducibility
- **Parallel Jobs**: Use `n_jobs=-4` (leave 4 cores free) for stability
- **PHATE Parameters**: PHATE typically uses `n_components=300`, `n_jobs=-2`, `random_state=42`, `n_pca=None`
- **Output Naming**: Output files follow pattern: `{embedding_model}_{output_type}/{filename}_{params}.{ext}`
- **Module Reloading**: Scripts often include `importlib.reload(phate)` to ensure latest version
