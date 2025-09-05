#!/usr/bin/env python3
"""
Example script showing how to use run_script_seed.py
"""

import subprocess
import sys

def run_example():
    """Run an example experiment with multiple seeds"""
    
    # Example command for running experiments
    cmd = [
        sys.executable, 'run_script_seed.py',
        '--arch', 'deit_tiny',
        '--w-bit', '4',
        '--a-bit', '4',
        '--seeds', '3407', '42', '123',  # Multiple seeds
        '--alpha-list', '0.3', '0.5', '0.7',  # Multiple alpha values
        '--num-clusters-list', '32', '64', '128',  # Multiple cluster numbers
        '--pca-dim-list', '25', '50', '100',  # Multiple PCA dimensions
        '--calib-size', '1000',
        '--calib-batch-size', '32',
        '--val-batch-size', '500',
        '--device', 'cuda',
        '--output-dir', './experiment_results'
    ]
    
    print("Running example experiment...")
    print(f"Command: {' '.join(cmd)}")
    
    # Run the command
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("Experiment completed successfully!")
    else:
        print("Experiment failed!")

if __name__ == "__main__":
    run_example()
