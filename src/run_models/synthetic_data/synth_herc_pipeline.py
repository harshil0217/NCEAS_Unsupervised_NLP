# ========================
# Environment Configuration
# ========================
from dotenv import load_dotenv
load_dotenv()
import os
import sys

# Set the target folder name you want to reach
target_folder = "src"

# Get the current working directory
current_dir = os.getcwd()

# Loop to move up the directory tree until we reach the target folder
while os.path.basename(current_dir) != target_folder:
    parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
    if parent_dir == current_dir:
        # If we reach the root directory and haven't found the target, exit
        raise FileNotFoundError(f"{target_folder} not found in the directory tree.")
    current_dir = parent_dir

# Change the working directory to the folder where "src" is found
os.chdir(current_dir)

# Add the "src" directory to sys.path
sys.path.insert(0, current_dir)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ========================
# Standard Libraries
# ========================
import importlib
import re
import time
import warnings
import argparse
from collections import defaultdict

# ========================
# Data Manipulation
# ========================
import numpy as np
import pandas as pd

# ========================
# Machine Learning
# ========================
import torch

torch.cuda.empty_cache()

# ========================
# NLP & Transformers
# ========================
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"

# ========================
# Clustering
# ========================
from pyhercules import Hercules

# ========================
# Evaluation Metrics
# ========================
from custom_packages.fowlkes_mallows import FowlkesMallows
from sklearn.metrics import adjusted_rand_score, rand_score, adjusted_mutual_info_score

# ========================
# Utilities
# ========================
from tqdm import tqdm

# ===================
# Global Config
# ===================
np.random.seed(42)
warnings.filterwarnings("ignore")

# ===================
# LLM Setup
# ===================
model_name = "Qwen/Qwen3-4B-Instruct-2507"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map='auto')
print(model.hf_device_map)


def qwen_caller(prompt: str) -> str:
    """
    Generates text using the Qwen model and returns the generated text.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant providing concise summaries."},
        {"role": "user", "content": prompt}
    ]

    text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=16384)

    output_ids = generated_ids[0][len(inputs["input_ids"][0]):].tolist()  # Get only the generated part

    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip("\n")

    print(content)
    return content


def get_sentence_transformer_embeddings(texts: list[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Embeds text using a specified SentenceTransformer model.
    """
    if not texts:
        return np.array([])

    model = SentenceTransformer(model_name, device=device)

    try:
        # encode() handles the list input and returns a numpy array by default
        embeddings = model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=False
        )

        # Ensure output is 2D even for single strings
        if embeddings.ndim == 1:
            embeddings = embeddings[np.newaxis, :]

        return embeddings

    except Exception as e:
        print(f"Error during embedding generation with '{model_name}': {e}")


# ====================
# Main Pipeline
# ====================
def run_synth_herc_pipeline(theme, t, max_sub, depth, synonyms, branching, add_noise, rep_mode):
    """
    Run Hercules pipeline on synthetic hierarchical data.

    Args:
        theme: Theme name for synthetic data
        t: Temperature parameter
        max_sub: Maximum subtopics
        depth: Hierarchy depth
        synonyms: Number of synonyms
        branching: Branching pattern (constant, decreasing, increasing, random)
        add_noise: Noise level (0.0-1.0)
        rep_mode: Hercules representation mode ('direct' or 'description')
    """
    print(f"\n{'='*80}")
    print(f"Running Hercules pipeline for synthetic data: {theme}")
    print(f"Parameters: t={t}, max_sub={max_sub}, depth={depth}, synonyms={synonyms}")
    print(f"Branching: {branching}, Noise: {add_noise}, Rep Mode: {rep_mode}")
    print(f"{'='*80}\n")

    # Load generated synthetic data
    filename = f'data_generation/generated_data/{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_noise{add_noise}_{branching}.csv'

    if not os.path.exists(filename):
        raise FileNotFoundError(f"Synthetic data file not found: {filename}\nPlease generate it first using generate.py")

    print(f"Loading data from: {filename}")
    topic_data_original = pd.read_csv(filename)

    # Prepare data (same shuffle for consistency)
    shuffle_idx = np.random.RandomState(seed=67).permutation(len(topic_data_original))
    topic_data = topic_data_original.iloc[shuffle_idx].reset_index(drop=True)
    reverse_idx = np.argsort(shuffle_idx)

    # Build topic_dict from ground truth categories
    topic_dict = {}
    for col in topic_data.columns:
        if re.match(r'^category \d+$', col):
            unique_count = len(topic_data[col].unique())
            topic_dict[unique_count] = np.array(topic_data[col])

    # Determine cluster levels from hierarchy depth
    print(f"Depth: {depth}")
    print(f"Building cluster levels by counting unique categories at each level...\n")

    cluster_levels = []
    for i in reversed(range(0, depth)):
        unique_count = len(topic_data[f'category {i}'].unique())
        print(f"Level {i} (category {i}): {unique_count} unique categories")
        cluster_levels.append(unique_count)

    print(f"\nFinal cluster_levels (from deepest to shallowest): {cluster_levels}\n")

    # Embedding models to use
    embedding_model_names = [
        "Qwen/Qwen3-Embedding-0.6B",
        "sentence-transformers/all-MiniLM-L6-v2",
    ]

    scores_all = defaultdict(lambda: defaultdict(list))

    # Process each embedding model
    for embedding_model in embedding_model_names:
        print(f"\n{'='*60}")
        print(f"Processing embedding model: {embedding_model}")
        print(f"{'='*60}\n")

        # Create save path for Hercules model
        safe_theme = theme.replace(" ", "_").replace("/", "_")
        save_path = f"../../hercules_run/synthetic/{safe_theme}_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_noise{add_noise}_{branching}/{rep_mode}/{embedding_model.replace('/', '_')}"

        hercules = None

        if os.path.exists(save_path):
            print(f"Loading existing Hercules model from {save_path}...")
            hercules = Hercules.load_model(
                filepath=save_path,
                text_embedding_client=get_sentence_transformer_embeddings,
                llm_client=qwen_caller
            )[0]
        else:
            print(f"Creating new Hercules model...")
            torch.cuda.empty_cache()
            hercules = Hercules(
                level_cluster_counts=cluster_levels,
                representation_mode=rep_mode,
                text_embedding_client=get_sentence_transformer_embeddings,
                llm_client=qwen_caller,
                verbose=2,
                llm_initial_batch_size=32
            )

            # Run clustering on synthetic topics
            topic_seed = f"Synthetic hierarchical data about {theme}"
            print(f"Running Hercules clustering with topic seed: {topic_seed}")
            top_clusters = hercules.cluster(topic_data['topic'].tolist(), topic_seed=topic_seed)

            # Save model
            print(f"Saving Hercules model to {save_path}...")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            hercules.save_model(filepath=save_path, top_clusters=top_clusters)

        # Get cluster assignments
        cluster_df = hercules.get_cluster_membership_dataframe(include_l0_details=topic_data['topic'].tolist())

        print(f"Hercules clustering complete. Evaluating against ground truth...")

        # Print cluster statistics
        for i in range(1, len(cluster_levels) + 1):
            col_name = f'L{i}_cluster_title'
            if col_name in cluster_df.columns:
                unique_clusters = cluster_df[col_name].nunique()
                print(f"Level {i}: {unique_clusters} clusters")

        # Save cluster assignments
        assignments_dir = f"results/cluster_assignments/synthetic"
        os.makedirs(assignments_dir, exist_ok=True)
        assignment_file = f"{assignments_dir}/{safe_theme}_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_noise{add_noise}_{branching}_{rep_mode}_{embedding_model.replace('/', '_')}.csv"
        cluster_df.to_csv(assignment_file, index=False)
        print(f"Cluster assignments saved to: {assignment_file}")

        # Evaluate at each level
        for i, cluster_level in enumerate(cluster_levels):
            level_idx = i + 1
            print(f"\nEvaluating level {level_idx} (target clusters: {cluster_level})...")

            labels = hercules.get_level_assignments(level=level_idx)[0]

            # Get ground truth for this level
            topic_series = topic_dict[cluster_level]
            valid_idx = ~pd.isna(topic_series)
            target_lst = topic_series[valid_idx]
            label_lst = labels[valid_idx]

            # Compute metrics
            try:
                fm_score = FowlkesMallows.Bk({cluster_level: target_lst}, {cluster_level: label_lst})[cluster_level]['FM']
            except Exception as e:
                fm_score = np.nan
                print(f"WARNING: FM score computation failed: {e}")

            rand = rand_score(target_lst, label_lst)
            ari = adjusted_rand_score(target_lst, label_lst)
            ami = adjusted_mutual_info_score(target_lst, label_lst)

            print(f"Level {level_idx} Scores - FM: {fm_score:.4f}, Rand: {rand:.4f}, ARI: {ari:.4f}, AMI: {ami:.4f}")

            # Store scores
            scores_all[(embedding_model, level_idx)]["level"].append(cluster_level)
            scores_all[(embedding_model, level_idx)]["FM"].append(fm_score)
            scores_all[(embedding_model, level_idx)]["Rand"].append(rand)
            scores_all[(embedding_model, level_idx)]["ARI"].append(ari)
            scores_all[(embedding_model, level_idx)]["AMI"].append(ami)

    # Save results to CSV
    print(f"\n{'='*60}")
    print("Preparing results for export...")
    print(f"{'='*60}\n")

    rows = []
    for (embedding_model, level_idx), metrics in scores_all.items():
        for i in range(len(metrics["level"])):
            rows.append({
                "embedding_model": embedding_model,
                "cluster_method": "HERCULES",
                "representation_mode": rep_mode,
                "level": metrics["level"][i],
                "level_idx": level_idx,
                "FM": metrics["FM"][i],
                "Rand": metrics["Rand"][i],
                "ARI": metrics["ARI"][i],
                "AMI": metrics["AMI"][i],
            })

    # Create DataFrame and save
    scores_df = pd.DataFrame(rows)
    scores_df = scores_df.sort_values(by=["embedding_model", "level_idx"]).reset_index(drop=True)

    os.makedirs("results", exist_ok=True)
    if float(add_noise) > 0:
        output_file = f"results/{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_noise{add_noise}_{branching}_herc_{rep_mode}_scores.csv"
    else:
        output_file = f"results/{theme}_hierarchy_t{t}_maxsub{max_sub}_depth{depth}_synonyms{synonyms}_{branching}_herc_{rep_mode}_scores.csv"

    scores_df.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    print(f"\n{'='*60}")
    print("Hercules pipeline complete!")
    print(f"{'='*60}\n")


# ====================
# Setup & Execution
# ====================
def noise_range(value):
    f = float(value)
    if f < 0 or f > 1:
        raise argparse.ArgumentTypeError("add_noise must be a float between 0 and 1.")
    return f


def main():
    parser = argparse.ArgumentParser(
        description="Run Hercules hierarchical clustering on synthetic data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python synth_herc_pipeline.py \\
        --theme "Energy_Ecosystems_and_Humans" \\
        --t 1.0 \\
        --max_sub 5 \\
        --depth 3 \\
        --synonyms 0 \\
        --branching random \\
        --add_noise 0.0 \\
        --rep_mode direct

Representation modes:
    direct      - Use raw text for cluster representation
    description - Use LLM-generated cluster descriptions
        """
    )

    parser.add_argument("--theme", type=str, required=True,
                        help="Theme name (must match generated data file)")
    parser.add_argument("--t", type=float, required=True,
                        help="Temperature parameter")
    parser.add_argument("--max_sub", type=int, required=True,
                        help="Maximum number of subtopics")
    parser.add_argument("--depth", type=int, required=True,
                        help="Hierarchy depth")
    parser.add_argument("--synonyms", type=int, required=True,
                        help="Number of synonyms")
    parser.add_argument("--branching", type=str, required=True,
                        choices=["constant", "decreasing", "increasing", "random"],
                        help="Branching pattern")
    parser.add_argument("--add_noise", type=noise_range, default=0.0,
                        help="Amount of noise to add (float between 0 and 1)")
    parser.add_argument("--rep_mode", type=str, required=True,
                        choices=["direct", "description"],
                        help="Hercules representation mode")

    args = parser.parse_args()

    # Run pipeline
    run_synth_herc_pipeline(
        theme=args.theme,
        t=args.t,
        max_sub=args.max_sub,
        depth=args.depth,
        synonyms=args.synonyms,
        branching=args.branching,
        add_noise=args.add_noise,
        rep_mode=args.rep_mode
    )


if __name__ == "__main__":
    main()
