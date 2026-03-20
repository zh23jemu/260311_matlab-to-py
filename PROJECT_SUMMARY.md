# Project Summary

## Goal

This project is a Python reproduction of the workflow described in `去噪自编码器py编程.docx`.

The target workflow is:

1. Load TE process data from `CNN/data567.mat`.
2. Remove columns 46 and 50 from the original 52-dimensional signals, leaving 50 features.
3. Standardize all train/test data with z-score statistics computed from `d00`.
4. Add Gaussian noise to the DAE training set.
5. Train a denoising autoencoder.
6. Extract encoded features from the DAE.
7. Standardize encoded features again using encoded `d00` statistics.
8. Train a classifier on the encoded fault features.
9. Export metrics, figures, and model checkpoints.

## Repository Structure

- `main.py`: thin entrypoint, calls `te_dae.pipeline.main()`.
- `te_dae/data.py`: TE dataset loading, feature column removal, z-score preprocessing, noise injection.
- `te_dae/models.py`: DAE and classifier model definitions.
- `te_dae/train.py`: training loops, feature extraction, prediction, confusion matrix.
- `te_dae/pipeline.py`: end-to-end orchestration, logging, metrics/figure/model export.
- `te_dae/plotting.py`: plotting helpers for Figures 4.2 to 4.12 and DAE training.
- `CNN/`: original MATLAB reference code and `data567.mat`.
- `report_assets/`: git-tracked selected training results used for sync/review.
- `outputs/`: default runtime output directory, ignored by git.
- `run_te_dae.slurm`: single Slurm job wrapper for one training run.
- `submit_te_dae_sweep.sh`: batch submission script for seed/wuc sweep.
- `submit_te_dae_wuc_finetune.sh`: finer sweep around the current best `wuc`.

## Key Data Conventions

- Training data uses `d00_6` to `d20_6`.
- Test data is built by stacking `_5` and `_7` splits.
- Fault ids used for classifier training are:
  `1, 2, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20`
- Columns removed are zero-based indices `45` and `49`, corresponding to original columns 46 and 50.
- The special test sample count convention matches the MATLAB workflow:
  `F6` uses `247` samples, all other selected faults use `2000`.

## Model Conventions

### DAE

- Architecture: `50 -> 50 -> 45 -> 40 -> 45 -> 50 -> 50`
- Hidden activations: `LeakyReLU`
- Training optimizer: `SGD(momentum=0.9)`
- Default DAE training parameters:
  - learning rate: `0.0001`
  - epochs: `2500`
  - batch size: `32`

### Classifier

- Architecture: `40 -> 400 -> 250 -> 17`
- Hidden activations: `Tanh`
- Training optimizer: `SGD(momentum=0.9)`
- Default classifier training parameters:
  - learning rate: `0.0001`
  - epochs: `300`
  - batch size: `300`

## Feature Extraction Modes

`te_dae/pipeline.py` supports:

- `bottleneck_relu`
- `fc3_linear`

Current default is `bottleneck_relu`.

This is important because the project compared two interpretations of the MATLAB DAE feature layer:

- `bottleneck_relu`: encoded output after the third activation.
- `fc3_linear`: linear output of `fc3` before the third activation.

Current experiments indicate `bottleneck_relu` is the better default.

## Runtime Output Conventions

`main.py` writes to `--output-dir`, defaulting to `outputs`.

Expected structure under an output directory:

- `figures/`
- `metrics/`
- `models/`

Key files:

- `metrics/metrics.json`
- `figures/figure_4_12.png`
- `models/dae.pt`
- `models/classifier.pt`

`report_assets/` should only contain selected synced results intended for git tracking and comparison.

## Metrics File Conventions

`metrics.json` currently records:

- `device`
- `seed`
- `wuc`
- `dae_epochs`
- `clf_epochs`
- `dae_feature_mode`
- `mean_accuracy`
- `per_fault_accuracy`

## Logging Conventions

Training now emits progress logs to stdout for Slurm log inspection:

- `[PIPELINE]`: dataset loading, standardization, feature encoding, save stages.
- `[DAE]`: start message and periodic DAE loss updates.
- `[CLF]`: start message and periodic classifier loss/accuracy updates.
- `[EVAL]`: per-fault test accuracy.
- `[RESULT]`: final mean accuracy and output directory.

## Slurm Conventions

`run_te_dae.slurm` accepts these environment variables:

- `FEATURE_MODE`
- `SEED`
- `WUC`
- `OUTPUT_DIR`
- `REPORT_SUFFIX`

Default behavior:

- `FEATURE_MODE=bottleneck_relu`
- `SEED=42`
- `WUC=0.01`
- `OUTPUT_DIR=outputs_${FEATURE_MODE}_s${SEED}_w${WUC}`
- `REPORT_SUFFIX=${FEATURE_MODE}_s${SEED}_w${WUC}`

After training, the script copies selected artifacts into `report_assets/`:

- `metrics_${REPORT_SUFFIX}.json`
- `figure_4_12_${REPORT_SUFFIX}.png`

This avoids collisions between concurrent Slurm jobs.

## Current Experimental Status

### Feature mode comparison

- `bottleneck_relu` and `fc3_linear` were both tested.
- `fc3_linear` did not materially improve the difficult classes.
- `bottleneck_relu` remains the preferred mode.

### Difficult classes

The main remaining alignment problem versus the document-style target performance is:

- `F8`
- `F11`
- `F13`

### Best current targeted result for `F8/F11/F13`

From the tracked sweeps, the best targeted configuration so far is:

- `FEATURE_MODE=bottleneck_relu`
- `SEED=42`
- `WUC=0.02`

Tracked result file:

- `report_assets/metrics_bottleneck_s42_w002.json`

This setting improves `F8/F11/F13` over the earlier baseline, but still does not fully align with the document-level expectation for those classes.

### Best current overall mean accuracy

The best overall mean accuracy in the recent sweep is:

- `FEATURE_MODE=bottleneck_relu`
- `SEED=42`
- `WUC=0.005`

Tracked result file:

- `report_assets/metrics_bottleneck_s42_w0005.json`

This setting is more balanced overall, but slightly weaker than `WUC=0.02` on the specific hard classes.

## Recommended Next Step

Continue from:

- `FEATURE_MODE=bottleneck_relu`
- `SEED=42`

Then fine-sweep `WUC` around the currently best targeted region:

- `0.015`
- `0.018`
- `0.020`
- `0.022`
- `0.025`

Use:

- `submit_te_dae_wuc_finetune.sh`

