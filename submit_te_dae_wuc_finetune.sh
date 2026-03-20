#!/bin/bash
set -euo pipefail

# Fine sweep around the current best setup for F11/F8/F13.
# Run on the Slurm login node from the project root:
#   bash submit_te_dae_wuc_finetune.sh

sbatch --export=ALL,FEATURE_MODE=bottleneck_relu,SEED=42,WUC=0.015,OUTPUT_DIR=outputs_bottleneck_s42_w0015,REPORT_SUFFIX=bottleneck_s42_w0015 run_te_dae.slurm
sbatch --export=ALL,FEATURE_MODE=bottleneck_relu,SEED=42,WUC=0.018,OUTPUT_DIR=outputs_bottleneck_s42_w0018,REPORT_SUFFIX=bottleneck_s42_w0018 run_te_dae.slurm
sbatch --export=ALL,FEATURE_MODE=bottleneck_relu,SEED=42,WUC=0.020,OUTPUT_DIR=outputs_bottleneck_s42_w0020,REPORT_SUFFIX=bottleneck_s42_w0020 run_te_dae.slurm
sbatch --export=ALL,FEATURE_MODE=bottleneck_relu,SEED=42,WUC=0.022,OUTPUT_DIR=outputs_bottleneck_s42_w0022,REPORT_SUFFIX=bottleneck_s42_w0022 run_te_dae.slurm
sbatch --export=ALL,FEATURE_MODE=bottleneck_relu,SEED=42,WUC=0.025,OUTPUT_DIR=outputs_bottleneck_s42_w0025,REPORT_SUFFIX=bottleneck_s42_w0025 run_te_dae.slurm
