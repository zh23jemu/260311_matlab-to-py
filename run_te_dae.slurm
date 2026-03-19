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

cd /net/8k3/e0fs01/users/xj62kv/260311_matlab-to-py || exit 1
source venv/bin/activate || exit 1

python main.py \
  --dae-epochs 2500 \
  --clf-epochs 300 \
  --device cpu \
  --output-dir outputs

mkdir -p report_assets
cp outputs/metrics/metrics.json report_assets/metrics.json
cp outputs/figures/figure_4_12.png report_assets/figure_4_12.png
