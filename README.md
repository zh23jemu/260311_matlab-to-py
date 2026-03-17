# TE DAE Python

Python reproduction of the workflow described in `去噪自编码器py编程.docx`.

## Setup

```powershell
venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Quick smoke test

```powershell
venv\Scripts\python.exe main.py --dae-epochs 2 --clf-epochs 2 --device cpu
```

## Full training

```powershell
venv\Scripts\python.exe main.py --dae-epochs 2500 --clf-epochs 300 --device cpu
```

If you have CUDA available:

```powershell
venv\Scripts\python.exe main.py --dae-epochs 2500 --clf-epochs 300 --device cuda
```

## Outputs

- `outputs/models/dae.pt`
- `outputs/models/classifier.pt`
- `outputs/metrics/metrics.json`
- `outputs/figures/figure_4_2.png` to `figure_4_12.png`
- `outputs/figures/dae_training.png`
