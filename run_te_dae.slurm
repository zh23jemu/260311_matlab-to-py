#!/bin/bash
#SBATCH -A mfk-gsr2
#SBATCH -J te_dae
#SBATCH -p ISP
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o te_dae_%j.out

set -euo pipefail

FEATURE_MODE=${FEATURE_MODE:-bottleneck_relu}
SEED=${SEED:-42}
WUC=${WUC:-0.01}
OUTPUT_DIR=${OUTPUT_DIR:-outputs_${FEATURE_MODE}_s${SEED}_w${WUC}}
REPORT_SUFFIX=${REPORT_SUFFIX:-${FEATURE_MODE}_s${SEED}_w${WUC}}

cd /net/8k3/e0fs01/users/xj62kv/260311_matlab-to-py || exit 1
source venv/bin/activate || exit 1

echo "[SLURM] FEATURE_MODE=${FEATURE_MODE}"
echo "[SLURM] SEED=${SEED}"
echo "[SLURM] WUC=${WUC}"
echo "[SLURM] OUTPUT_DIR=${OUTPUT_DIR}"
echo "[SLURM] REPORT_SUFFIX=${REPORT_SUFFIX}"

python main.py \
  --dae-epochs 2500 \
  --clf-epochs 300 \
  --device cpu \
  --seed "${SEED}" \
  --wuc "${WUC}" \
  --output-dir "${OUTPUT_DIR}" \
  --dae-feature-mode "${FEATURE_MODE}"

mkdir -p report_assets
cp "${OUTPUT_DIR}/metrics/metrics.json" "report_assets/metrics_${REPORT_SUFFIX}.json"
cp "${OUTPUT_DIR}/figures/figure_4_12.png" "report_assets/figure_4_12_${REPORT_SUFFIX}.png"
