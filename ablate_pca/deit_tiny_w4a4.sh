#!/bin/bash
#SBATCH -J DEIT_TINY_W4A4_PCA
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -p gpu_computervision_long
#SBATCH --gres=gpu:1
#SBATCH --tmp=5G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-email-address>
#SBATCH -t 4-00:00:00

source /home/alz07xz/project/PD-Quant/pd_quant/bin/activate
echo "Starting DeiT-Tiny W4A4 PCA ablation experiment at $(date)"
python run_script_seed.py --arch deit_tiny --w-bit 4 --a-bit 4 --seeds 1001 1002 1003 --sleep 0.5 --alpha-list 0.5 --num-clusters-list 64 --pca-dim-list 1 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 80 85 90 95 100 105 110 115 120 125 130 135 140 145 150 155 160 165 170 175 180 185 190 195 200 205 210 215 220
echo "Completed DeiT-Tiny W4A4 PCA ablation experiment at $(date)"
