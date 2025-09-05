# Alpha Ablation Experiments

This folder contains SLURM batch scripts for running alpha ablation experiments across different architectures and bit precision combinations.

## Scripts Overview

Each script runs `run_script_seed.py` with the following parameters:
- **Architectures**: `deit_tiny`, `vit_tiny`, `vit_small`, `vit_base`, `deit_small`, `deit_base`, `swin_tiny`, `swin_small`, `swin_base`
- **Bit Combinations**: W2A2, W2A4, W4A2, W4A4
- **Seeds**: 3407, 42, 123 (3 seeds for statistical significance)
- **Alpha Values**: 0.3, 0.5, 0.7 (3 alpha values for ablation)
- **Cluster Numbers**: 32, 64, 128 (3 cluster configurations)
- **PCA Dimensions**: 25, 50, 100 (3 PCA configurations)

## Total Experiments

- **9 architectures** × **4 bit combinations** × **3 seeds** × **3 alpha values** × **3 cluster numbers** × **3 PCA dimensions** = **2,916 total experiments**

## Script Naming Convention

Scripts follow the pattern: `{architecture}_{bit_combination}.sh`

Examples:
- `deit_tiny_w2a2.sh` - DeiT-Tiny with 2-bit weights and 2-bit activations
- `vit_small_w4a4.sh` - ViT-Small with 4-bit weights and 4-bit activations
- `swin_base_w2a4.sh` - Swin-Base with 2-bit weights and 4-bit activations

## SLURM Configuration

Each script is configured with:
- **Job Name**: Architecture and bit combination (e.g., `DEIT_TINY_W2A2`)
- **CPUs**: 8 cores
- **Memory**: 128GB RAM
- **GPU**: 1 GPU
- **Partition**: `gpu_computervision_long`
- **Time Limit**: 4 days (4-00:00:00)
- **Temporary Storage**: 5GB

## Usage

To submit a job:
```bash
sbatch ablate_alpha/deit_tiny_w2a2.sh
```

To submit all jobs:
```bash
for script in ablate_alpha/*.sh; do
    sbatch "$script"
done
```

## Output

Each experiment will create:
- Individual results for each seed
- Statistical analysis (mean, std) across seeds
- Both baseline and reconstructed results
- Comprehensive CSV and JSON outputs

## Email Notifications

Remember to update the email address in each script:
```bash
#SBATCH --mail-user=<your-email-address>
```

## Expected Runtime

Each script runs approximately 3-4 days depending on the architecture size and bit precision combination.
