# Multi-Seed Experiment Runner

This module (`run_script_seed.py`) allows you to run `test_quant.py` experiments across multiple random seeds systematically, making it easy to evaluate the robustness and statistical significance of your quantization experiments.

## Features

- **Multi-seed execution**: Run experiments across multiple random seeds
- **Parameter sweep**: Test different combinations of alpha values, cluster numbers, and PCA dimensions
- **Automatic result collection**: Collect and parse results from all experiments
- **Statistical analysis**: Calculate mean, standard deviation, and best results across seeds
- **Flexible output**: Save results in both JSON and CSV formats
- **Error handling**: Robust error handling with timeout protection

## Usage

### Basic Usage

```bash
python run_script_seed.py \
    --arch deit_tiny \
    --w-bit 4 \
    --a-bit 4 \
    --seeds 3407 42 123 \
    --alpha-list 0.3 0.5 0.7 \
    --num-clusters-list 32 64 128 \
    --pca-dim-list 25 50 100
```

### Advanced Usage

```bash
python run_script_seed.py \
    --arch deit_base \
    --w-bit 3 \
    --a-bit 3 \
    --seeds 3407 42 123 456 789 \
    --alpha-list 0.2 0.4 0.6 0.8 \
    --num-clusters-list 16 32 64 128 256 \
    --pca-dim-list 10 25 50 100 \
    --calib-size 2000 \
    --calib-batch-size 64 \
    --val-batch-size 1000 \
    --device cuda:0 \
    --output-dir ./my_experiments \
    --reconstruct-mlp
```

## Parameters

### Required Parameters
- `--arch`: Model architecture (vit_tiny, vit_small, vit_base, vit_large, deit_tiny, deit_small, deit_base, swin_tiny, swin_small, swin_base, swin_base_384)
- `--w-bit`: Weight bit precision
- `--a-bit`: Activation bit precision
- `--seeds`: List of random seeds to test
- `--alpha-list`: List of alpha values for blending
- `--num-clusters-list`: List of cluster numbers for K-means
- `--pca-dim-list`: List of PCA dimensions

### Optional Parameters
- `--config`: Path to config file (default: APHQ-ViT config)
- `--dataset`: Path to dataset (default: ImageNet path)
- `--calib-size`: Calibration set size (default: 1000)
- `--calib-batch-size`: Calibration batch size (default: 32)
- `--val-batch-size`: Validation batch size (default: 500)
- `--num-workers`: Number of data loading workers (default: 8)
- `--device`: Device to use (default: cuda)
- `--output-dir`: Output directory (default: ./experiment_results)
- `--reconstruct-mlp`: Enable MLP reconstruction
- `--calibrate`: Enable calibration (default: True)
- `--optimize`: Enable optimization (default: True)

## Output Structure

The script creates a timestamped output directory with the following structure:

```
experiment_results/
└── deit_tiny_w4_a4_20241201_143022/
    ├── experiment.log                    # Main experiment log
    ├── experiment_summary.json          # Detailed results in JSON format
    ├── experiment_summary.csv           # Results in CSV format
    ├── seed_3407/                      # Individual seed results
    │   ├── stdout.txt
    │   └── stderr.txt
    ├── seed_42/
    │   ├── stdout.txt
    │   └── stderr.txt
    └── seed_123/
        ├── stdout.txt
        └── stderr.txt
```

## Output Files

### experiment_summary.json
Contains all results with metadata:
```json
{
  "args": {...},
  "results": [
    {
      "seed": 3407,
      "alpha": 0.5,
      "num_clusters": 64,
      "pca_dim": 50,
      "top1_accuracy": 75.23,
      "top5_accuracy": 92.15
    },
    ...
  ],
  "timestamp": "2024-12-01T14:30:22"
}
```

### experiment_summary.csv
CSV format for easy analysis:
```csv
seed,alpha,num_clusters,pca_dim,top1_accuracy,top5_accuracy
3407,0.5,64,50,75.23,92.15
42,0.5,64,50,74.89,91.87
...
```

## Example Results

The script provides statistical analysis:

```
EXPERIMENT SUMMARY
================================================================================
BEST RESULT ACROSS ALL SEEDS:
  Seed: 42
  Alpha: 0.5
  Clusters: 64
  PCA_dim: 50
  Top-1 Accuracy: 75.45%
  Top-5 Accuracy: 92.30%

STATISTICS PER PARAMETER COMBINATION:
Alpha    Clusters   PCA_dim    Mean Top-1   Std Top-1    Best Top-1  
--------------------------------------------------------------------------------
0.30     32        25         72.15        0.45         72.89      
0.30     32        50         73.22        0.38         73.67      
0.30     64        25         73.45        0.52         74.12      
...
```

## Tips for Effective Experiments

1. **Start small**: Begin with a few seeds (3-5) and limited parameter combinations
2. **Use meaningful seeds**: Choose seeds that are commonly used in literature (3407, 42, 123)
3. **Parameter ranges**: Test a reasonable range around expected optimal values
4. **Resource management**: Consider GPU memory and time constraints
5. **Incremental testing**: Test parameter combinations incrementally to identify promising ranges

## Troubleshooting

- **Timeout errors**: Increase timeout in the script or reduce parameter combinations
- **Memory errors**: Reduce batch sizes or use smaller models
- **CUDA errors**: Check GPU availability and memory usage
- **File not found**: Ensure `test_quant.py` is in the same directory

## Integration with Existing Workflows

This script is designed to work seamlessly with your existing `test_quant.py` setup. It simply calls `test_quant.py` with different parameters and collects the results, so all your existing configurations and dependencies remain unchanged.
