#!/bin/bash
#SBATCH -A mfk-gsr2
#SBATCH -J te_dae
#SBATCH -p ISP
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o te_dae_%j.out

cd /net/8k3/e0fs01/users/xj62kv/260311_matlab-to-py
source venv/bin/activate
python main.py --dae-epochs 2500 --clf-epochs 300 --device cpu
