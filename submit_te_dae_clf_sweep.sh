#!/bin/bash
set -euo pipefail

# Sweep classifier hyperparameters around the current best backbone setup.
# Fixed setup:
#   FEATURE_MODE=bottleneck_relu
#   SEED=42
#   WUC=0.025
# Run on the Slurm login node from the project root:
#   bash submit_te_dae_clf_sweep.sh

sbatch --export=ALL,FEATURE_MODE=bottleneck_relu,SEED=42,WUC=0.025,CLF_LR=0.00005,CLF_EPOCHS=300,OUTPUT_DIR=outputs_bottleneck_s42_w0025_clr5e5_ce300,REPORT_SUFFIX=bottleneck_s42_w0025_clr5e5_ce300 run_te_dae.slurm
sbatch --export=ALL,FEATURE_MODE=bottleneck_relu,SEED=42,WUC=0.025,CLF_LR=0.0001,CLF_EPOCHS=300,OUTPUT_DIR=outputs_bottleneck_s42_w0025_clr1e4_ce300,REPORT_SUFFIX=bottleneck_s42_w0025_clr1e4_ce300 run_te_dae.slurm
sbatch --export=ALL,FEATURE_MODE=bottleneck_relu,SEED=42,WUC=0.025,CLF_LR=0.0002,CLF_EPOCHS=300,OUTPUT_DIR=outputs_bottleneck_s42_w0025_clr2e4_ce300,REPORT_SUFFIX=bottleneck_s42_w0025_clr2e4_ce300 run_te_dae.slurm
sbatch --export=ALL,FEATURE_MODE=bottleneck_relu,SEED=42,WUC=0.025,CLF_LR=0.0001,CLF_EPOCHS=400,OUTPUT_DIR=outputs_bottleneck_s42_w0025_clr1e4_ce400,REPORT_SUFFIX=bottleneck_s42_w0025_clr1e4_ce400 run_te_dae.slurm
sbatch --export=ALL,FEATURE_MODE=bottleneck_relu,SEED=42,WUC=0.025,CLF_LR=0.0001,CLF_EPOCHS=500,OUTPUT_DIR=outputs_bottleneck_s42_w0025_clr1e4_ce500,REPORT_SUFFIX=bottleneck_s42_w0025_clr1e4_ce500 run_te_dae.slurm
