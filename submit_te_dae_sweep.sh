#!/bin/bash
set -euo pipefail

# Submit a small sweep focused on improving F11/F8/F13.
# Run on the Slurm login node from the project root:
#   bash submit_te_dae_sweep.sh

sbatch --export=ALL,FEATURE_MODE=bottleneck_relu,SEED=42,WUC=0.01,OUTPUT_DIR=outputs_bottleneck_s42_w001,REPORT_SUFFIX=bottleneck_s42_w001 run_te_dae.slurm
sbatch --export=ALL,FEATURE_MODE=bottleneck_relu,SEED=7,WUC=0.01,OUTPUT_DIR=outputs_bottleneck_s7_w001,REPORT_SUFFIX=bottleneck_s7_w001 run_te_dae.slurm
sbatch --export=ALL,FEATURE_MODE=bottleneck_relu,SEED=123,WUC=0.01,OUTPUT_DIR=outputs_bottleneck_s123_w001,REPORT_SUFFIX=bottleneck_s123_w001 run_te_dae.slurm

sbatch --export=ALL,FEATURE_MODE=bottleneck_relu,SEED=42,WUC=0.005,OUTPUT_DIR=outputs_bottleneck_s42_w0005,REPORT_SUFFIX=bottleneck_s42_w0005 run_te_dae.slurm
sbatch --export=ALL,FEATURE_MODE=bottleneck_relu,SEED=42,WUC=0.02,OUTPUT_DIR=outputs_bottleneck_s42_w002,REPORT_SUFFIX=bottleneck_s42_w002 run_te_dae.slurm
