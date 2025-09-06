#!/bin/bash
#SBATCH -J DEIT_SMALL_W2A4_CLUSTERS
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -p gpu_computervision_long
#SBATCH --gres=gpu:1
#SBATCH --tmp=5G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-email-address>
#SBATCH -t 4-00:00:00

source /home/alz07xz/project/PD-Quant/pd_quant/bin/activate
echo "Starting DeiT-Small W2A4 cluster ablation experiment at $(date)"
python ../run_script_seed.py --arch deit_small --w-bit 2 --a-bit 4 --seeds 1001 1002 1003 --sleep 0.5 --alpha-list 0.5 --num-clusters-list 1 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128 136 144 152 160 168 176 184 192 200 208 216 224 232 240 248 256 --pca-dim-list 50
echo "Completed DeiT-Small W2A4 cluster ablation experiment at $(date)"
