# Cluster Ablation Experiments

This directory contains SLURM batch scripts for running cluster ablation experiments across different architectures and bit-width combinations.

## Overview

These experiments systematically test the effect of different cluster numbers on model performance while keeping other parameters fixed:
- **Alpha**: Fixed at 0.5
- **PCA Dimensions**: Fixed at 50
- **Clusters**: Varied from 1, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256

## Architectures and Bit-Width Combinations

### Models Tested:
- **DeiT**: Tiny, Small, Base
- **ViT**: Tiny, Small, Base  
- **Swin**: Tiny, Small, Base

### Bit-Width Combinations:
- W2A2 (2-bit weights, 2-bit activations)
- W2A4 (2-bit weights, 4-bit activations)
- W4A2 (4-bit weights, 2-bit activations)
- W4A4 (4-bit weights, 4-bit activations)

## Total Experiments

- **36 scripts** (9 architectures × 4 bit-width combinations)
- **33 cluster values** per script
- **3 seeds** per configuration
- **Total configurations per script**: 33 × 3 = 99 experiments
- **Total experiments across all scripts**: 36 × 99 = 3,564 experiments

## Usage

### Submit All Experiments:
```bash
cd ablate_clusters
./run_all.sh
```

### Submit Individual Experiments:
```bash
sbatch deit_tiny_w2a2.sh
```

### Monitor Jobs:
```bash
squeue -u $USER
```

### Cancel All Jobs:
```bash
scancel -u $USER
```

## Output Files

Each experiment will generate:
- `{script_name}.out` - SLURM output file
- Results will be saved in timestamped directories by `run_script_seed.py`

## Expected Runtime

Each script is allocated 4 days (4-00:00:00) to complete all cluster ablation experiments.

## Cluster Values Tested

The cluster values follow a specific pattern:
- Start with 1 (baseline)
- Then multiples of 8: 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152, 160, 168, 176, 184, 192, 200, 208, 216, 224, 232, 240, 248, 256

This allows for systematic analysis of how cluster count affects model performance across different architectures and quantization levels.
