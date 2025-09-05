# PCA Ablation Experiments

This directory contains SLURM batch scripts for running PCA ablation experiments across different architectures and bit-width combinations.

## Overview

These experiments systematically test the effect of different PCA dimensions on model performance while keeping other parameters fixed:
- **Alpha**: Fixed at 0.5
- **Clusters**: Fixed at 64
- **PCA Dimensions**: Varied from 1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220

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
- **45 PCA values** per script
- **3 seeds** per configuration
- **Total configurations per script**: 45 × 3 = 135 experiments
- **Total experiments across all scripts**: 36 × 135 = 4,860 experiments

## Usage

### Submit All Experiments:
```bash
cd ablate_pca
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

Each script is allocated 4 days (4-00:00:00) to complete all PCA ablation experiments.

## PCA Values Tested

The PCA values follow a specific pattern:
- Start with 1 (baseline)
- Then multiples of 5: 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195, 200, 205, 210, 215, 220

This allows for systematic analysis of how PCA dimensionality affects model performance across different architectures and quantization levels.

## Analysis Benefits

This systematic PCA ablation will help identify:
- **Optimal PCA dimensions** for each architecture/bit-width combination
- **Performance scaling** with PCA dimensionality
- **Architecture-specific** PCA requirements
- **Bit-width sensitivity** to PCA dimensionality
- **Dimensionality reduction trade-offs** between computational efficiency and model performance
