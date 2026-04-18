#!/bin/bash
#SBATCH --job-name=dc_eval
#SBATCH --partition=general-l
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dc_%j.out
#SBATCH --error=logs/dc_%j.err

mkdir -p logs

echo "======================================"
echo "Job started on $(hostname)"
echo "Time: $(date)"
echo "======================================"

cd /mnt/home/alkhal13/phase3 || { echo "FAILED: cannot cd"; exit 1; }

source ~/qwen_env/bin/activate || { echo "FAILED ENV"; exit 1; }

# CPU control (good)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2

echo "Running Diffusion Condensation (Phase 3 FINAL)..."

start_time=$(date +%s)

python phase4_DC_eval.py   # ✅ FIXED

end_time=$(date +%s)

echo "Total runtime: $((end_time - start_time)) seconds"
echo "DONE"