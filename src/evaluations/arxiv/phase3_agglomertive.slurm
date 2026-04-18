#!/bin/bash
#SBATCH --job-name=agg_eval
#SBATCH --partition=general-l
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/agg_%j.out
#SBATCH --error=logs/agg_%j.err

mkdir -p logs

cd /mnt/home/alkhal13/phase3 || { echo "FAILED: cannot cd"; exit 1; }

source ~/qwen_env/bin/activate

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2

echo "Running from: $(pwd)"
ls

python phase3_agglomerative.py