#!/usr/bin/env python3
"""
Demo script showing the enhanced run_script_seed.py functionality
with comprehensive logging and statistics.
"""

import subprocess
import sys
import json
import os

def run_demo():
    """Run a demo experiment with enhanced logging"""
    
    print("="*80)
    print("DEMO: Enhanced Multi-Seed Experiment Runner")
    print("="*80)
    
    # Demo command with multiple seeds and parameters
    cmd = [
        sys.executable, 'run_script_seed.py',
        '--arch', 'deit_tiny',
        '--w-bit', '4',
        '--a-bit', '4',
        '--seeds', '3407', '42', '123',  # Multiple seeds for statistics
        '--alpha-list', '0.3', '0.5', '0.7',  # Multiple alpha values
        '--num-clusters-list', '32', '64',  # Multiple cluster numbers
        '--pca-dim-list', '25', '50',  # Multiple PCA dimensions
        '--calib-size', '1000',
        '--calib-batch-size', '32',
        '--val-batch-size', '500',
        '--device', 'cuda',
        '--output-dir', './demo_experiment_results',
        '--sleep', '0.5'  # Sleep between runs
    ]
    
    print("Command to run:")
    print(' '.join(cmd))
    print("\nThis will:")
    print("• Run test_quant.py for 3 seeds (3407, 42, 123)")
    print("• Test 3 alpha values × 2 cluster numbers × 2 PCA dimensions = 12 combinations per seed")
    print("• Total: 36 experiments")
    print("• Save individual results for each seed")
    print("• Calculate mean and standard deviation for each parameter combination")
    print("• Generate comprehensive statistics and reports")
    
    response = input("\nRun the demo? (y/n): ")
    if response.lower() != 'y':
        print("Demo cancelled.")
        return
    
    print("\nRunning demo experiment...")
    print("This may take some time depending on your hardware.")
    
    # Run the command
    result = subprocess.run(cmd)
    
    if result.returncode == 0:
        print("\n✓ Demo experiment completed successfully!")
        print("\nCheck the following files in ./demo_experiment_results/:")
        print("• experiment.log - Complete experiment log")
        print("• experiment_summary.json - All results in JSON format")
        print("• experiment_summary.csv - Results in CSV format")
        print("• individual_seed_results.json - Results organized by seed")
        print("• detailed_statistics.json - Comprehensive statistics")
        
        # Show a sample of the results
        try:
            results_dir = './demo_experiment_results'
            if os.path.exists(results_dir):
                # Find the most recent timestamped directory
                subdirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
                if subdirs:
                    latest_dir = max(subdirs)
                    stats_file = os.path.join(results_dir, latest_dir, 'detailed_statistics.json')
                    
                    if os.path.exists(stats_file):
                        with open(stats_file, 'r') as f:
                            stats = json.load(f)
                        
                        print(f"\nSample Results from {latest_dir}:")
                        print("-" * 50)
                        print(f"Total experiments: {stats['overall_stats']['total_experiments']}")
                        print(f"Overall Mean Top-1: {stats['overall_stats']['mean_top1']:.2f}% ± {stats['overall_stats']['std_top1']:.2f}%")
                        print(f"Overall Mean Top-5: {stats['overall_stats']['mean_top5']:.2f}% ± {stats['overall_stats']['std_top5']:.2f}%")
                        print(f"Best Top-1: {stats['overall_stats']['best_top1']:.2f}%")
                        print(f"Best Top-5: {stats['overall_stats']['best_top5']:.2f}%")
                        
                        print(f"\nTop 3 Parameter Combinations:")
                        param_stats = list(stats['parameter_combinations'].items())
                        param_stats.sort(key=lambda x: x[1]['mean_top1'], reverse=True)
                        
                        for i, (key, data) in enumerate(param_stats[:3]):
                            print(f"{i+1}. α={data['alpha']:.2f}, clusters={data['num_clusters']}, pca={data['pca_dim']}")
                            print(f"   Mean Top-1: {data['mean_top1']:.2f}% ± {data['std_top1']:.2f}%")
                            print(f"   Mean Top-5: {data['mean_top5']:.2f}% ± {data['std_top5']:.2f}%")
                            print(f"   Best Top-1: {data['best_top1']:.2f}%")
                            print()
        except Exception as e:
            print(f"Could not display sample results: {e}")
            
    else:
        print("✗ Demo experiment failed!")
        print("Check the error messages above for details.")

def show_usage_examples():
    """Show usage examples"""
    print("\n" + "="*80)
    print("USAGE EXAMPLES")
    print("="*80)
    
    examples = [
        {
            "title": "Basic Multi-Seed Experiment",
            "cmd": [
                "python run_script_seed.py",
                "--arch deit_tiny --w-bit 4 --a-bit 4",
                "--seeds 3407 42 123",
                "--alpha-list 0.5 --num-clusters-list 64 --pca-dim-list 50"
            ]
        },
        {
            "title": "Comprehensive Parameter Sweep",
            "cmd": [
                "python run_script_seed.py",
                "--arch deit_base --w-bit 3 --a-bit 3",
                "--seeds 3407 42 123 456 789",
                "--alpha-list 0.2 0.4 0.6 0.8",
                "--num-clusters-list 16 32 64 128",
                "--pca-dim-list 25 50 100"
            ]
        },
        {
            "title": "Quick Test with Limited Parameters",
            "cmd": [
                "python run_script_seed.py",
                "--arch deit_tiny --w-bit 4 --a-bit 4",
                "--seeds 3407 42",
                "--alpha-list 0.3 0.5 0.7",
                "--num-clusters-list 32 64",
                "--pca-dim-list 25 50",
                "--calib-size 500"
            ]
        }
    ]
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['title']}:")
        print("   " + " ".join(example['cmd']))

if __name__ == "__main__":
    print("Enhanced Multi-Seed Experiment Runner Demo")
    print("This script demonstrates the enhanced functionality for running")
    print("test_quant.py across multiple seeds with comprehensive statistics.")
    
    while True:
        print("\nOptions:")
        print("1. Run demo experiment")
        print("2. Show usage examples")
        print("3. Exit")
        
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            run_demo()
        elif choice == '2':
            show_usage_examples()
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please select 1, 2, or 3.")
