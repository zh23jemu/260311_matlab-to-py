#!/bin/bash
set -euo pipefail

# Narrow sweep above the current best wuc=0.025.
# Run on the Slurm login node from the project root:
#   bash submit_te_dae_wuc_high.sh

sbatch --export=ALL,FEATURE_MODE=bottleneck_relu,SEED=42,WUC=0.028,OUTPUT_DIR=outputs_bottleneck_s42_w0028,REPORT_SUFFIX=bottleneck_s42_w0028 run_te_dae.slurm
sbatch --export=ALL,FEATURE_MODE=bottleneck_relu,SEED=42,WUC=0.03,OUTPUT_DIR=outputs_bottleneck_s42_w0030,REPORT_SUFFIX=bottleneck_s42_w0030 run_te_dae.slurm
sbatch --export=ALL,FEATURE_MODE=bottleneck_relu,SEED=42,WUC=0.032,OUTPUT_DIR=outputs_bottleneck_s42_w0032,REPORT_SUFFIX=bottleneck_s42_w0032 run_te_dae.slurm
sbatch --export=ALL,FEATURE_MODE=bottleneck_relu,SEED=42,WUC=0.035,OUTPUT_DIR=outputs_bottleneck_s42_w0035,REPORT_SUFFIX=bottleneck_s42_w0035 run_te_dae.slurm
sbatch --export=ALL,FEATURE_MODE=bottleneck_relu,SEED=42,WUC=0.04,OUTPUT_DIR=outputs_bottleneck_s42_w0040,REPORT_SUFFIX=bottleneck_s42_w0040 run_te_dae.slurm
