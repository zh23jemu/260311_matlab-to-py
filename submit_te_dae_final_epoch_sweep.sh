#!/bin/bash
set -euo pipefail

# Final narrow sweep on classifier epochs only.
# Fixed setup:
#   FEATURE_MODE=bottleneck_relu
#   SEED=42
#   WUC=0.025
#   CLF_LR=0.0002
# Current best anchor:
#   CLF_EPOCHS=700
# Run on the Slurm login node from the project root:
#   bash submit_te_dae_final_epoch_sweep.sh

sbatch --export=ALL,FEATURE_MODE=bottleneck_relu,SEED=42,WUC=0.025,CLF_LR=0.0002,CLF_EPOCHS=800,OUTPUT_DIR=outputs_bottleneck_s42_w0025_clr2e4_ce800,REPORT_SUFFIX=bottleneck_s42_w0025_clr2e4_ce800 run_te_dae.slurm
sbatch --export=ALL,FEATURE_MODE=bottleneck_relu,SEED=42,WUC=0.025,CLF_LR=0.0002,CLF_EPOCHS=900,OUTPUT_DIR=outputs_bottleneck_s42_w0025_clr2e4_ce900,REPORT_SUFFIX=bottleneck_s42_w0025_clr2e4_ce900 run_te_dae.slurm
sbatch --export=ALL,FEATURE_MODE=bottleneck_relu,SEED=42,WUC=0.025,CLF_LR=0.0002,CLF_EPOCHS=1000,OUTPUT_DIR=outputs_bottleneck_s42_w0025_clr2e4_ce1000,REPORT_SUFFIX=bottleneck_s42_w0025_clr2e4_ce1000 run_te_dae.slurm
