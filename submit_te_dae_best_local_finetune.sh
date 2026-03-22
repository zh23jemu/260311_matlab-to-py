#!/bin/bash
set -euo pipefail

# Fine-tune around the current best overall setup.
# Fixed setup:
#   FEATURE_MODE=bottleneck_relu
#   SEED=42
#   CLF_LR=0.0002
# Current best anchor:
#   WUC=0.025, CLF_EPOCHS=500
# Run on the Slurm login node from the project root:
#   bash submit_te_dae_best_local_finetune.sh

sbatch --export=ALL,FEATURE_MODE=bottleneck_relu,SEED=42,WUC=0.024,CLF_LR=0.0002,CLF_EPOCHS=500,OUTPUT_DIR=outputs_bottleneck_s42_w0024_clr2e4_ce500,REPORT_SUFFIX=bottleneck_s42_w0024_clr2e4_ce500 run_te_dae.slurm
sbatch --export=ALL,FEATURE_MODE=bottleneck_relu,SEED=42,WUC=0.025,CLF_LR=0.0002,CLF_EPOCHS=600,OUTPUT_DIR=outputs_bottleneck_s42_w0025_clr2e4_ce600,REPORT_SUFFIX=bottleneck_s42_w0025_clr2e4_ce600 run_te_dae.slurm
sbatch --export=ALL,FEATURE_MODE=bottleneck_relu,SEED=42,WUC=0.026,CLF_LR=0.0002,CLF_EPOCHS=500,OUTPUT_DIR=outputs_bottleneck_s42_w0026_clr2e4_ce500,REPORT_SUFFIX=bottleneck_s42_w0026_clr2e4_ce500 run_te_dae.slurm
sbatch --export=ALL,FEATURE_MODE=bottleneck_relu,SEED=42,WUC=0.025,CLF_LR=0.0002,CLF_EPOCHS=700,OUTPUT_DIR=outputs_bottleneck_s42_w0025_clr2e4_ce700,REPORT_SUFFIX=bottleneck_s42_w0025_clr2e4_ce700 run_te_dae.slurm
sbatch --export=ALL,FEATURE_MODE=bottleneck_relu,SEED=42,WUC=0.026,CLF_LR=0.0002,CLF_EPOCHS=600,OUTPUT_DIR=outputs_bottleneck_s42_w0026_clr2e4_ce600,REPORT_SUFFIX=bottleneck_s42_w0026_clr2e4_ce600 run_te_dae.slurm
