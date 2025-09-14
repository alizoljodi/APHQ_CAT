#!/bin/bash
#SBATCH -J SWIN_BASE_W4A6_QDROP
#SBATCH -c 8
#SBATCH --mem=128G
#SBATCH -p gpu_computervision_long
#SBATCH --gres=gpu:1
#SBATCH --tmp=5G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<your-email-address>
#SBATCH -t 4-00:00:00

source /home/alz07xz/project/PD-Quant/pd_quant/bin/activate
echo "Starting Swin-Base W4A6 QDROP experiment at $(date)"
python ../run_script_seed.py --arch swin_base --w-bit 4 --a-bit 6 --seeds 1001 1002 1003 --sleep 0.5 --config ../configs/4bit/qdrop_baseline.py --alpha-list 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 --num-clusters-list 8 16 32 64 128 256 --pca-dim-list 1 25 50 75 100 125 150 175 200 220
echo "Completed Swin-Base W4A6 QDROP experiment at $(date)"
