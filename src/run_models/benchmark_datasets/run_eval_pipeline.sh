#!/bin/bash

# run_eval_pipeline.sh
# Shell script to run the generalized benchmark evaluation pipeline
#
# Usage:
#   ./run_eval_pipeline.sh <dataset_name>
#
# Available datasets: amazon, dbpedia, arxiv, rcv1, wos
#
# Examples:
#   bash run_eval_pipeline.sh dbpedia
#   bash run_eval_pipeline.sh amazon
#
# Compute Config:
#   8 Cores
#   96 GB RAM
#   1 x A100 GPU


# Check if dataset argument is provided
if [ $# -eq 0 ]; then
    echo "Error: No dataset specified"
    echo ""
    echo "Usage: ./run_eval_pipeline.sh <dataset_name>"
    echo ""
    echo "Available datasets:"
    echo "  amazon    - Amazon product reviews (3 levels)"
    echo "  dbpedia   - DBpedia topics (3 levels)"
    echo "  arxiv     - arXiv paper categories (2 levels)"
    echo "  rcv1      - Reuters RCV1 news categories (2 levels)"
    echo "  wos       - Web of Science (2 levels)"
    echo ""
    echo "Example:"
    echo "  ./run_eval_pipeline.sh dbpedia"
    exit 1
fi

DATASET=$1

# Validate dataset name
case $DATASET in
    amazon|dbpedia|arxiv|rcv1|wos)
        echo "Running evaluation pipeline for dataset: $DATASET"
        ;;
    *)
        echo "Error: Unknown dataset '$DATASET'"
        echo ""
        echo "Available datasets: amazon, dbpedia, arxiv, rcv1, wos"
        exit 1
        ;;
esac

# Navigate to repository root (assumes script is in src/run_models/benchmark_datasets/)
cd "$(dirname "$0")/../../.."

# Run the pipeline
python src/run_models/benchmark_datasets/eval_pipeline.py --dataset "$DATASET"

