#!/usr/bin/env python3
"""
Script to run test_quant.py for multiple seeds with specified parameters.
This module allows systematic experimentation across different random seeds.
"""

import os
import sys
import argparse
import json
import time
import re
import statistics
import pandas as pd
import io
import glob
from datetime import datetime
import logging
from typing import Optional, Tuple, List

def convert_tensor_to_scalar(value):
    """Convert tensor values to scalar values, handling various formats."""
    if isinstance(value, str) and 'tensor(' in value:
        # Extract the numeric value from tensor string
        try:
            # Handle tensor(0.1234) format
            match = re.search(r'tensor\(([^)]+)\)', value)
            if match:
                return float(match.group(1))
        except:
            pass
    elif hasattr(value, 'item'):  # If it's a tensor object
        return value.item()
    return value

def get_args_parser():
    parser = argparse.ArgumentParser(description='Run test_quant.py for multiple seeds')
    
    # Model parameters
    parser.add_argument('--arch', type=str, default='deit_tiny',
                        choices=['vit_tiny', 'vit_small', 'vit_base', 'vit_large',
                                 'deit_tiny', 'deit_small', 'deit_base', 
                                 'swin_tiny', 'swin_small', 'swin_base', 'swin_base_384'],
                        help='Model architecture')
    parser.add_argument('--w-bit', type=int, default=4, help='Weight bit precision')
    parser.add_argument('--a-bit', type=int, default=4, help='Activation bit precision')
    
    # Cluster affine correction parameters
    parser.add_argument('--alpha-list', type=float, nargs='+', default=[0.5], 
                        help='List of alpha values for blending (e.g., --alpha-list 0.3 0.5 0.7)')
    parser.add_argument('--num-clusters-list', type=int, nargs='+', default=[64], 
                        help='List of cluster numbers (e.g., --num-clusters-list 32 64 128)')
    parser.add_argument('--pca-dim-list', type=int, nargs='+', default=[50], 
                        help='List of PCA dimensions (e.g., --pca-dim-list 25 50 100)')
    
    # Seed parameters
    parser.add_argument('--seeds', type=int, nargs='+', default=[3407, 42, 123, 456, 789],
                        help='List of seeds to run (e.g., --seeds 3407 42 123)')
    
    # Other test_quant.py parameters
    parser.add_argument('--config', type=str, 
                        default="./../configs/4bit/best.py",
                        help="File path to import Config class from")
    parser.add_argument('--dataset', default="/home/alz07xz/imagenet",
                        help='path to dataset')
    parser.add_argument('--calib-size', type=int, default=1000, help="size of calibration set")
    parser.add_argument('--calib-batch-size', type=int, default=32, help="batchsize of calibration set")
    parser.add_argument('--val-batch-size', default=500, type=int, help="batchsize of validation set")
    parser.add_argument('--num-workers', default=8, type=int, help="number of data loading workers")
    parser.add_argument('--device', default="cuda", type=str, help="device")
    
    # Reconstruction and optimization flags
    parser.add_argument('--reconstruct-mlp', action='store_true', default=False, 
                        help='reconstruct mlp with ReLU function.')
    parser.add_argument('--calibrate', action='store_true', default=True, help="Calibrate the model")
    parser.add_argument('--optimize', action='store_true', default=True, help="Optimize the model")
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='./experiment_results',
                        help='Directory to save experiment results')
    parser.add_argument('--save-individual-results', action='store_true', default=True,
                        help='Save individual results for each seed')
    parser.add_argument('--save-summary', action='store_true', default=True,
                        help='Save summary of all experiments')
    parser.add_argument('--sleep', type=float, default=0.5, 
                        help='Seconds to sleep between runs')
    
    return parser

def run_single_experiment(args, seed, output_dir):
    """Run test_quant.py for a single seed using os.system"""
    
    # Create command string
    cmd_parts = [
        sys.executable, '../test_quant.py',
        '--model', args.arch,
        '--w_bit', str(args.w_bit),
        '--a_bit', str(args.a_bit),
        '--seed', str(seed),
        '--config', args.config,
        '--dataset', args.dataset,
        '--calib-size', str(args.calib_size),
        '--calib-batch-size', str(args.calib_batch_size),
        '--val-batch-size', str(args.val_batch_size),
        '--num-workers', str(args.num_workers),
        '--device', args.device,
        '--alpha-list'] + [str(a) for a in args.alpha_list] + [
        '--num-clusters-list'] + [str(c) for c in args.num_clusters_list] + [
        '--pca-dim-list'] + [str(p) for p in args.pca_dim_list]
    
    # Add optional flags
    if args.reconstruct_mlp:
        cmd_parts.append('--reconstruct-mlp')
    if args.calibrate:
        cmd_parts.append('--calibrate')
    if args.optimize:
        cmd_parts.append('--optimize')
    
    # Create seed-specific output directory
    seed_output_dir = os.path.join(output_dir, f'seed_{seed}')
    os.makedirs(seed_output_dir, exist_ok=True)
    
    # Convert to command string
    cmd = ' '.join(cmd_parts)
    
    # Log command
    logging.info(f"Running experiment for seed {seed}")
    logging.info(f"Command: {cmd}")
    logging.info(f"Working directory: {os.getcwd()}")
    
    # Run the command using os.system
    start_time = time.time()
    returncode = os.system(cmd)
    end_time = time.time()
    execution_time = end_time - start_time
    
    logging.info(f"Experiment for seed {seed} completed in {execution_time:.2f} seconds")
    
    if returncode == 0:
        logging.info(f"SUCCESS: Experiment for seed {seed} completed successfully")
        return True, ""
    else:
        error_msg = f"ERROR: Experiment for seed {seed} failed with return code {returncode}"
        logging.error(error_msg)
        return False, error_msg

def parse_results_from_file(seed, args):
    """Parse results from saved files since os.system doesn't capture output"""
    results = []
    baseline_results = []  # Store non-reconstructed baseline results
    
    # Look for saved results files - test_quant.py should save results to checkpoint directory
    # The test_quant.py creates a timestamped directory in ./checkpoint/quant_result/
    checkpoint_base = './checkpoint/quant_result'
    
    if not os.path.exists(checkpoint_base):
        logging.warning(f"No checkpoint directory found for seed {seed}")
        return results, baseline_results
    
    # Find the most recent timestamped directory
    try:
        subdirs = [d for d in os.listdir(checkpoint_base) if os.path.isdir(os.path.join(checkpoint_base, d))]
        if not subdirs:
            logging.warning(f"No subdirectories found in {checkpoint_base} for seed {seed}")
            return results, baseline_results
        
        # Sort by modification time to get the most recent
        subdirs.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoint_base, x)), reverse=True)
        latest_dir = subdirs[0]
        latest_path = os.path.join(checkpoint_base, latest_dir)
        
        logging.info(f"Looking for results in: {latest_path}")
        
        # Look for output.log file which contains the results
        log_file = os.path.join(latest_path, 'output.log')
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                output_text = f.read()
            
            # Parse the output text for results
            lines = output_text.split('\n')
            
            # Track current parameters for baseline results
            current_alpha = None
            current_clusters = None
            current_pca_dim = None
            
            # Look for individual result lines and summary section
            in_summary = False
            for line in lines:
                if "SUMMARY OF ALL RESULTS" in line:
                    in_summary = True
                    continue
                elif in_summary and line.strip() and not line.startswith('=') and not line.startswith('-'):
                    # Parse result line: Alpha Clusters PCA_dim Top-1 Top-5
                    parts = line.split()
                    if len(parts) >= 5:
                        try:
                            alpha = float(parts[0])
                            clusters = int(parts[1])
                            pca_dim = int(parts[2])
                            top1 = float(parts[3])
                            top5 = float(parts[4])
                            results.append({
                                'alpha': alpha,
                                'num_clusters': clusters,
                                'pca_dim': pca_dim,
                                'top1_accuracy': top1,
                                'top5_accuracy': top5
                            })
                        except (ValueError, IndexError):
                            continue
                elif "Testing:" in line and "alpha=" in line:
                    # Parse individual test results: Testing: alpha=0.5, clusters=64, pca_dim=50
                    try:
                        alpha_start = line.find('alpha=') + 6
                        alpha_end = line.find(',', alpha_start)
                        current_alpha = float(line[alpha_start:alpha_end])
                        
                        clusters_start = line.find('clusters=') + 9
                        clusters_end = line.find(',', clusters_start)
                        current_clusters = int(line[clusters_start:clusters_end])
                        
                        pca_start = line.find('pca_dim=') + 8
                        current_pca_dim = int(line[pca_start:])
                    except (ValueError, IndexError):
                        continue
                elif "Result: Top-1:" in line and "Top-5:" in line:
                    # Parse individual result: Result: Top-1: 75.23%, Top-5: 92.15%
                    try:
                        top1_start = line.find('Top-1: ') + 7
                        top1_end = line.find('%', top1_start)
                        top1 = float(line[top1_start:top1_end])
                        
                        top5_start = line.find('Top-5: ') + 7
                        top5_end = line.find('%', top5_start)
                        top5 = float(line[top5_start:top5_end])
                        
                        # Check if we have current parameters (this is a baseline result)
                        if current_alpha is not None and current_clusters is not None and current_pca_dim is not None:
                            baseline_results.append({
                                'alpha': current_alpha,
                                'num_clusters': current_clusters,
                                'pca_dim': current_pca_dim,
                                'top1_accuracy': top1,
                                'top5_accuracy': top5,
                                'type': 'baseline'  # Mark as baseline result
                            })
                            logging.info(f"Captured baseline result: α={current_alpha:.2f}, clusters={current_clusters}, pca={current_pca_dim}: "
                                       f"Top-1={top1:.2f}%, Top-5={top5:.2f}%")
                        
                        # If we have reconstructed results, update the most recent one
                        if len(results) > 0:
                            # Update the most recent result with actual accuracies
                            results[-1]['top1_accuracy'] = top1
                            results[-1]['top5_accuracy'] = top5
                            results[-1]['type'] = 'reconstructed'  # Mark as reconstructed result
                    except (ValueError, IndexError):
                        continue
            
            logging.info(f"Parsed {len(results)} reconstructed results from log file for seed {seed}")
            logging.info(f"Parsed {len(baseline_results)} baseline results from log file for seed {seed}")
        else:
            logging.warning(f"No output.log found in {latest_path} for seed {seed}")
            
    except Exception as e:
        logging.error(f"Error parsing results for seed {seed}: {e}")
    
    return results, baseline_results

def save_summary_results(all_results, all_baseline_results, args, output_dir):
    """Save summary of all experiments with detailed statistics"""
    
    # Flatten reconstructed results
    flattened_results = []
    for seed, results in all_results.items():
        for result in results:
            result_copy = result.copy()
            result_copy['seed'] = seed
            result_copy['type'] = 'reconstructed'
            flattened_results.append(result_copy)
    
    # Flatten baseline results
    flattened_baseline_results = []
    for seed, results in all_baseline_results.items():
        for result in results:
            result_copy = result.copy()
            result_copy['seed'] = seed
            result_copy['type'] = 'baseline'
            flattened_baseline_results.append(result_copy)
    
    # Combine all results for overall analysis
    all_flattened_results = flattened_results + flattened_baseline_results
    
    # Save detailed results as JSON
    summary_file = os.path.join(output_dir, 'experiment_summary.json')
    with open(summary_file, 'w') as f:
        json.dump({
            'args': vars(args),
            'reconstructed_results': flattened_results,
            'baseline_results': flattened_baseline_results,
            'all_results': all_flattened_results,
            'timestamp': datetime.now().isoformat()
        }, f, indent=2)
    
    # Save CSV format for reconstructed results
    csv_file = os.path.join(output_dir, 'experiment_summary.csv')
    with open(csv_file, 'w') as f:
        f.write('seed,type,alpha,num_clusters,pca_dim,top1_accuracy,top5_accuracy\n')
        for result in all_flattened_results:
            f.write(f"{result['seed']},{result['type']},{result['alpha']},{result['num_clusters']},"
                   f"{result['pca_dim']},{result['top1_accuracy']},{result['top5_accuracy']}\n")
    
    # Save individual seed results
    individual_results_file = os.path.join(output_dir, 'individual_seed_results.json')
    with open(individual_results_file, 'w') as f:
        json.dump({
            'reconstructed_results': all_results,
            'baseline_results': all_baseline_results
        }, f, indent=2)
    
    # Print comprehensive summary statistics
    logging.info("\n" + "="*100)
    logging.info("COMPREHENSIVE EXPERIMENT SUMMARY")
    logging.info("="*100)
    
    if all_flattened_results:
        # Find best reconstructed result across all seeds
        best_reconstructed = max(flattened_results, key=lambda x: x['top1_accuracy']) if flattened_results else None
        best_baseline = max(flattened_baseline_results, key=lambda x: x['top1_accuracy']) if flattened_baseline_results else None
        
        logging.info(f"BEST RECONSTRUCTED RESULT ACROSS ALL SEEDS:")
        if best_reconstructed:
            logging.info(f"  Seed: {best_reconstructed['seed']}")
            logging.info(f"  Alpha: {best_reconstructed['alpha']}")
            logging.info(f"  Clusters: {best_reconstructed['num_clusters']}")
            logging.info(f"  PCA_dim: {best_reconstructed['pca_dim']}")
            logging.info(f"  Top-1 Accuracy: {best_reconstructed['top1_accuracy']:.2f}%")
            logging.info(f"  Top-5 Accuracy: {best_reconstructed['top5_accuracy']:.2f}%")
        
        logging.info(f"\nBEST BASELINE RESULT ACROSS ALL SEEDS:")
        if best_baseline:
            logging.info(f"  Seed: {best_baseline['seed']}")
            logging.info(f"  Alpha: {best_baseline['alpha']}")
            logging.info(f"  Clusters: {best_baseline['num_clusters']}")
            logging.info(f"  PCA_dim: {best_baseline['pca_dim']}")
            logging.info(f"  Top-1 Accuracy: {best_baseline['top1_accuracy']:.2f}%")
            logging.info(f"  Top-5 Accuracy: {best_baseline['top5_accuracy']:.2f}%")
        
        # Calculate statistics per parameter combination for reconstructed results
        logging.info(f"\nRECONSTRUCTED RESULTS - STATISTICS PER PARAMETER COMBINATION:")
        logging.info(f"{'Alpha':<8} {'Clusters':<10} {'PCA_dim':<10} {'Mean Top-1':<12} {'Std Top-1':<12} {'Mean Top-5':<12} {'Std Top-5':<12} {'Best Top-1':<12}")
        logging.info("-" * 100)
        
        # Group reconstructed results by parameter combination
        reconstructed_param_groups = {}
        for result in flattened_results:
            key = (result['alpha'], result['num_clusters'], result['pca_dim'])
            if key not in reconstructed_param_groups:
                reconstructed_param_groups[key] = {'top1': [], 'top5': []}
            reconstructed_param_groups[key]['top1'].append(result['top1_accuracy'])
            reconstructed_param_groups[key]['top5'].append(result['top5_accuracy'])
        
        # Sort by mean Top-1 accuracy for better readability
        sorted_reconstructed_groups = sorted(reconstructed_param_groups.items(), 
                                           key=lambda x: sum(x[1]['top1'])/len(x[1]['top1']), 
                                           reverse=True)
        
        for (alpha, clusters, pca_dim), accuracies in sorted_reconstructed_groups:
            top1_accs = accuracies['top1']
            top5_accs = accuracies['top5']
            
            mean_top1 = sum(top1_accs) / len(top1_accs)
            std_top1 = (sum((x - mean_top1)**2 for x in top1_accs) / len(top1_accs))**0.5
            best_top1 = max(top1_accs)
            
            mean_top5 = sum(top5_accs) / len(top5_accs)
            std_top5 = (sum((x - mean_top5)**2 for x in top5_accs) / len(top5_accs))**0.5
            
            logging.info(f"{alpha:<8.2f} {clusters:<10} {pca_dim:<10} {mean_top1:<12.2f} {std_top1:<12.2f} {mean_top5:<12.2f} {std_top5:<12.2f} {best_top1:<12.2f}")
        
        # Calculate statistics per parameter combination for baseline results
        logging.info(f"\nBASELINE RESULTS - STATISTICS PER PARAMETER COMBINATION:")
        logging.info(f"{'Alpha':<8} {'Clusters':<10} {'PCA_dim':<10} {'Mean Top-1':<12} {'Std Top-1':<12} {'Mean Top-5':<12} {'Std Top-5':<12} {'Best Top-1':<12}")
        logging.info("-" * 100)
        
        # Group baseline results by parameter combination
        baseline_param_groups = {}
        for result in flattened_baseline_results:
            key = (result['alpha'], result['num_clusters'], result['pca_dim'])
            if key not in baseline_param_groups:
                baseline_param_groups[key] = {'top1': [], 'top5': []}
            baseline_param_groups[key]['top1'].append(result['top1_accuracy'])
            baseline_param_groups[key]['top5'].append(result['top5_accuracy'])
        
        # Sort by mean Top-1 accuracy for better readability
        sorted_baseline_groups = sorted(baseline_param_groups.items(), 
                                      key=lambda x: sum(x[1]['top1'])/len(x[1]['top1']), 
                                      reverse=True)
        
        for (alpha, clusters, pca_dim), accuracies in sorted_baseline_groups:
            top1_accs = accuracies['top1']
            top5_accs = accuracies['top5']
            
            mean_top1 = sum(top1_accs) / len(top1_accs)
            std_top1 = (sum((x - mean_top1)**2 for x in top1_accs) / len(top1_accs))**0.5
            best_top1 = max(top1_accs)
            
            mean_top5 = sum(top5_accs) / len(top5_accs)
            std_top5 = (sum((x - mean_top5)**2 for x in top5_accs) / len(top5_accs))**0.5
            
            logging.info(f"{alpha:<8.2f} {clusters:<10} {pca_dim:<10} {mean_top1:<12.2f} {std_top1:<12.2f} {mean_top5:<12.2f} {std_top5:<12.2f} {best_top1:<12.2f}")
        
        # Detailed per-seed breakdown
        logging.info(f"\nDETAILED PER-SEED BREAKDOWN:")
        logging.info(f"{'Seed':<8} {'Alpha':<8} {'Clusters':<10} {'PCA_dim':<10} {'Top-1':<10} {'Top-5':<10}")
        logging.info("-" * 70)
        
        # Sort by seed for consistent display
        sorted_results = sorted(flattened_results, key=lambda x: (x['seed'], x['alpha'], x['num_clusters'], x['pca_dim']))
        for result in sorted_results:
            logging.info(f"{result['seed']:<8} {result['alpha']:<8.2f} {result['num_clusters']:<10} {result['pca_dim']:<10} "
                        f"{result['top1_accuracy']:<10.2f} {result['top5_accuracy']:<10.2f}")
        
        # Overall statistics for reconstructed results
        reconstructed_top1 = [r['top1_accuracy'] for r in flattened_results]
        reconstructed_top5 = [r['top5_accuracy'] for r in flattened_results]
        
        # Overall statistics for baseline results
        baseline_top1 = [r['top1_accuracy'] for r in flattened_baseline_results]
        baseline_top5 = [r['top5_accuracy'] for r in flattened_baseline_results]
        
        logging.info(f"\nOVERALL STATISTICS - RECONSTRUCTED RESULTS:")
        if reconstructed_top1:
            reconstructed_mean_top1 = sum(reconstructed_top1) / len(reconstructed_top1)
            reconstructed_std_top1 = (sum((x - reconstructed_mean_top1)**2 for x in reconstructed_top1) / len(reconstructed_top1))**0.5
            reconstructed_mean_top5 = sum(reconstructed_top5) / len(reconstructed_top5)
            reconstructed_std_top5 = (sum((x - reconstructed_mean_top5)**2 for x in reconstructed_top5) / len(reconstructed_top5))**0.5
            
            logging.info(f"  Total reconstructed experiments: {len(flattened_results)}")
            logging.info(f"  Overall Mean Top-1: {reconstructed_mean_top1:.2f}% ± {reconstructed_std_top1:.2f}%")
            logging.info(f"  Overall Mean Top-5: {reconstructed_mean_top5:.2f}% ± {reconstructed_std_top5:.2f}%")
            logging.info(f"  Best Top-1: {max(reconstructed_top1):.2f}%")
            logging.info(f"  Best Top-5: {max(reconstructed_top5):.2f}%")
            logging.info(f"  Worst Top-1: {min(reconstructed_top1):.2f}%")
            logging.info(f"  Worst Top-5: {min(reconstructed_top5):.2f}%")
        
        logging.info(f"\nOVERALL STATISTICS - BASELINE RESULTS:")
        if baseline_top1:
            baseline_mean_top1 = sum(baseline_top1) / len(baseline_top1)
            baseline_std_top1 = (sum((x - baseline_mean_top1)**2 for x in baseline_top1) / len(baseline_top1))**0.5
            baseline_mean_top5 = sum(baseline_top5) / len(baseline_top5)
            baseline_std_top5 = (sum((x - baseline_mean_top5)**2 for x in baseline_top5) / len(baseline_top5))**0.5
            
            logging.info(f"  Total baseline experiments: {len(flattened_baseline_results)}")
            logging.info(f"  Overall Mean Top-1: {baseline_mean_top1:.2f}% ± {baseline_std_top1:.2f}%")
            logging.info(f"  Overall Mean Top-5: {baseline_mean_top5:.2f}% ± {baseline_std_top5:.2f}%")
            logging.info(f"  Best Top-1: {max(baseline_top1):.2f}%")
            logging.info(f"  Best Top-5: {max(baseline_top5):.2f}%")
            logging.info(f"  Worst Top-1: {min(baseline_top1):.2f}%")
            logging.info(f"  Worst Top-5: {min(baseline_top5):.2f}%")
        
        # Save detailed statistics to file
        stats_file = os.path.join(output_dir, 'detailed_statistics.json')
        stats_data = {
            'overall_stats': {
                'total_experiments': len(flattened_results),
                'mean_top1': overall_mean_top1,
                'std_top1': overall_std_top1,
                'mean_top5': overall_mean_top5,
                'std_top5': overall_std_top5,
                'best_top1': max(all_top1),
                'best_top5': max(all_top5),
                'worst_top1': min(all_top1),
                'worst_top5': min(all_top5)
            },
            'parameter_combinations': {}
        }
        
        for (alpha, clusters, pca_dim), accuracies in param_groups.items():
            key = f"alpha_{alpha}_clusters_{clusters}_pca_{pca_dim}"
            stats_data['parameter_combinations'][key] = {
                'alpha': alpha,
                'num_clusters': clusters,
                'pca_dim': pca_dim,
                'mean_top1': sum(accuracies['top1']) / len(accuracies['top1']),
                'std_top1': (sum((x - sum(accuracies['top1'])/len(accuracies['top1']))**2 for x in accuracies['top1']) / len(accuracies['top1']))**0.5,
                'mean_top5': sum(accuracies['top5']) / len(accuracies['top5']),
                'std_top5': (sum((x - sum(accuracies['top5'])/len(accuracies['top5']))**2 for x in accuracies['top5']) / len(accuracies['top5']))**0.5,
                'best_top1': max(accuracies['top1']),
                'best_top5': max(accuracies['top5']),
                'num_seeds': len(accuracies['top1'])
            }
        
        with open(stats_file, 'w') as f:
            json.dump(stats_data, f, indent=2)
        
        logging.info(f"\nDetailed statistics saved to: {stats_file}")
        logging.info(f"Individual seed results saved to: {individual_results_file}")
        logging.info(f"Summary CSV saved to: {csv_file}")

def analyze_results_comprehensive(all_results, all_baseline_results, args, output_dir):
    """Comprehensive analysis similar to the provided code pattern"""
    logging.info(f"\n{'='*100}")
    logging.info("COMPREHENSIVE RESULTS ANALYSIS")
    logging.info(f"{'='*100}")
    
    # Flatten reconstructed results
    flattened_results = []
    for seed, results in all_results.items():
        for result in results:
            result_copy = result.copy()
            result_copy['seed'] = seed
            result_copy['type'] = 'reconstructed'
            flattened_results.append(result_copy)
    
    # Flatten baseline results
    flattened_baseline_results = []
    for seed, results in all_baseline_results.items():
        for result in results:
            result_copy = result.copy()
            result_copy['seed'] = seed
            result_copy['type'] = 'baseline'
            flattened_baseline_results.append(result_copy)
    
    # Combine all results
    all_flattened_results = flattened_results + flattened_baseline_results
    
    valid_results = [r for r in all_flattened_results if r is not None]
    failed_seeds = [seed for seed, results in all_results.items() if not results]
    
    logging.info(f"📊 Analysis Summary:")
    logging.info(f"  - Total seeds: {len(args.seeds)}")
    logging.info(f"  - Successful runs: {len(args.seeds) - len(failed_seeds)}")
    logging.info(f"  - Failed runs: {len(failed_seeds)}")
    logging.info(f"  - Reconstructed results: {len(flattened_results)}")
    logging.info(f"  - Baseline results: {len(flattened_baseline_results)}")
    if failed_seeds:
        logging.info(f"  - Failed seed numbers: {failed_seeds}")
    
    if not valid_results:
        logging.error("❌ No valid results to analyze!")
        return
    
    # Create DataFrames for analysis
    df_reconstructed = pd.DataFrame(flattened_results) if flattened_results else pd.DataFrame()
    df_baseline = pd.DataFrame(flattened_baseline_results) if flattened_baseline_results else pd.DataFrame()
    df_all = pd.DataFrame(valid_results)
    
    # Clean numeric columns for reconstructed results
    if not df_reconstructed.empty and 'top1_accuracy' in df_reconstructed.columns:
        df_reconstructed['top1_accuracy'] = df_reconstructed['top1_accuracy'].apply(convert_tensor_to_scalar)
        df_reconstructed['top1_accuracy'] = pd.to_numeric(df_reconstructed['top1_accuracy'], errors='coerce')
        df_reconstructed['top5_accuracy'] = df_reconstructed['top5_accuracy'].apply(convert_tensor_to_scalar)
        df_reconstructed['top5_accuracy'] = pd.to_numeric(df_reconstructed['top5_accuracy'], errors='coerce')
    
    # Clean numeric columns for baseline results
    if not df_baseline.empty and 'top1_accuracy' in df_baseline.columns:
        df_baseline['top1_accuracy'] = df_baseline['top1_accuracy'].apply(convert_tensor_to_scalar)
        df_baseline['top1_accuracy'] = pd.to_numeric(df_baseline['top1_accuracy'], errors='coerce')
        df_baseline['top5_accuracy'] = df_baseline['top5_accuracy'].apply(convert_tensor_to_scalar)
        df_baseline['top5_accuracy'] = pd.to_numeric(df_baseline['top5_accuracy'], errors='coerce')
    
    logging.info(f"✅ Reconstructed DataFrame shape: {df_reconstructed.shape}")
    logging.info(f"✅ Baseline DataFrame shape: {df_baseline.shape}")
    logging.info(f"✅ Combined DataFrame shape: {df_all.shape}")
    
    # Analyze reconstructed results
    if not df_reconstructed.empty:
        logging.info(f"\nRECONSTRUCTED RESULTS ANALYSIS:")
        logging.info(f"{'Alpha':<8} {'Clusters':<10} {'PCA_dim':<10} {'Mean Top-1':<12} {'Std Top-1':<12} {'Mean Top-5':<12} {'Std Top-5':<12}")
        logging.info("-" * 100)
        
        grouped_reconstructed = df_reconstructed.groupby(['alpha', 'num_clusters', 'pca_dim'])
        reconstructed_summary = []
        
        for name, group in grouped_reconstructed:
            alpha, num_clusters, pca_dim = name
            
            top1_mean = group['top1_accuracy'].mean()
            top1_std = group['top1_accuracy'].std() if len(group) > 1 else 0.0
            top5_mean = group['top5_accuracy'].mean()
            top5_std = group['top5_accuracy'].std() if len(group) > 1 else 0.0
            
            reconstructed_summary.append({
                'alpha': alpha, 'num_clusters': num_clusters, 'pca_dim': pca_dim,
                'top1_mean': top1_mean, 'top1_std': top1_std,
                'top5_mean': top5_mean, 'top5_std': top5_std
            })
        
        reconstructed_summary.sort(key=lambda x: x['top1_mean'], reverse=True)
        
        for result in reconstructed_summary:
            logging.info(f"{result['alpha']:<8.2f} {result['num_clusters']:<10} {result['pca_dim']:<10} "
                       f"{result['top1_mean']:<12.2f} {result['top1_std']:<12.2f} "
                       f"{result['top5_mean']:<12.2f} {result['top5_std']:<12.2f}")
    
    # Analyze baseline results
    if not df_baseline.empty:
        logging.info(f"\nBASELINE RESULTS ANALYSIS:")
        logging.info(f"{'Alpha':<8} {'Clusters':<10} {'PCA_dim':<10} {'Mean Top-1':<12} {'Std Top-1':<12} {'Mean Top-5':<12} {'Std Top-5':<12}")
        logging.info("-" * 100)
        
        grouped_baseline = df_baseline.groupby(['alpha', 'num_clusters', 'pca_dim'])
        baseline_summary = []
        
        for name, group in grouped_baseline:
            alpha, num_clusters, pca_dim = name
            
            top1_mean = group['top1_accuracy'].mean()
            top1_std = group['top1_accuracy'].std() if len(group) > 1 else 0.0
            top5_mean = group['top5_accuracy'].mean()
            top5_std = group['top5_accuracy'].std() if len(group) > 1 else 0.0
            
            baseline_summary.append({
                'alpha': alpha, 'num_clusters': num_clusters, 'pca_dim': pca_dim,
                'top1_mean': top1_mean, 'top1_std': top1_std,
                'top5_mean': top5_mean, 'top5_std': top5_std
            })
        
        baseline_summary.sort(key=lambda x: x['top1_mean'], reverse=True)
        
        for result in baseline_summary:
            logging.info(f"{result['alpha']:<8.2f} {result['num_clusters']:<10} {result['pca_dim']:<10} "
                       f"{result['top1_mean']:<12.2f} {result['top1_std']:<12.2f} "
                       f"{result['top5_mean']:<12.2f} {result['top5_std']:<12.2f}")
    
    # Overall statistics
    logging.info(f"\nOVERALL STATISTICS:")
    logging.info(f"  Total experiments: {len(valid_results)}")
    
    if not df_reconstructed.empty:
        reconstructed_top1 = df_reconstructed['top1_accuracy'].dropna()
        reconstructed_top5 = df_reconstructed['top5_accuracy'].dropna()
        logging.info(f"  Reconstructed - Mean Top-1: {reconstructed_top1.mean():.2f}% ± {reconstructed_top1.std():.2f}%")
        logging.info(f"  Reconstructed - Mean Top-5: {reconstructed_top5.mean():.2f}% ± {reconstructed_top5.std():.2f}%")
        logging.info(f"  Reconstructed - Best Top-1: {reconstructed_top1.max():.2f}%")
        logging.info(f"  Reconstructed - Best Top-5: {reconstructed_top5.max():.2f}%")
    
    if not df_baseline.empty:
        baseline_top1 = df_baseline['top1_accuracy'].dropna()
        baseline_top5 = df_baseline['top5_accuracy'].dropna()
        logging.info(f"  Baseline - Mean Top-1: {baseline_top1.mean():.2f}% ± {baseline_top1.std():.2f}%")
        logging.info(f"  Baseline - Mean Top-5: {baseline_top5.mean():.2f}% ± {baseline_top5.std():.2f}%")
        logging.info(f"  Baseline - Best Top-1: {baseline_top1.max():.2f}%")
        logging.info(f"  Baseline - Best Top-5: {baseline_top5.max():.2f}%")
    
    # Save comprehensive analysis
    analysis_file = os.path.join(output_dir, 'comprehensive_analysis.json')
    analysis_data = {
        'overall_stats': {
            'total_experiments': len(valid_results),
            'reconstructed_experiments': len(flattened_results),
            'baseline_experiments': len(flattened_baseline_results),
            'failed_seeds': failed_seeds
        },
        'reconstructed_results': {
            'summary': reconstructed_summary if 'reconstructed_summary' in locals() else [],
            'overall_stats': {
                'mean_top1': reconstructed_top1.mean() if not df_reconstructed.empty else None,
                'std_top1': reconstructed_top1.std() if not df_reconstructed.empty else None,
                'mean_top5': reconstructed_top5.mean() if not df_reconstructed.empty else None,
                'std_top5': reconstructed_top5.std() if not df_reconstructed.empty else None,
                'best_top1': reconstructed_top1.max() if not df_reconstructed.empty else None,
                'best_top5': reconstructed_top5.max() if not df_reconstructed.empty else None,
                'worst_top1': reconstructed_top1.min() if not df_reconstructed.empty else None,
                'worst_top5': reconstructed_top5.min() if not df_reconstructed.empty else None
            }
        },
        'baseline_results': {
            'summary': baseline_summary if 'baseline_summary' in locals() else [],
            'overall_stats': {
                'mean_top1': baseline_top1.mean() if not df_baseline.empty else None,
                'std_top1': baseline_top1.std() if not df_baseline.empty else None,
                'mean_top5': baseline_top5.mean() if not df_baseline.empty else None,
                'std_top5': baseline_top5.std() if not df_baseline.empty else None,
                'best_top1': baseline_top1.max() if not df_baseline.empty else None,
                'best_top5': baseline_top5.max() if not df_baseline.empty else None,
                'worst_top1': baseline_top1.min() if not df_baseline.empty else None,
                'worst_top5': baseline_top5.min() if not df_baseline.empty else None
            }
        }
    }
    
    with open(analysis_file, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    logging.info(f"\nComprehensive analysis saved to: {analysis_file}")

def preflight_check(args):
    """Simple pre-flight checks"""
    logging.info("Checking basic requirements...")
    
    # Check if test_quant.py exists
    if not os.path.exists('../test_quant.py'):
        logging.error(f"ERROR: test_quant.py not found in {os.getcwd()}")
        return False
    
    # Check if config file exists
    if not os.path.exists(args.config):
        logging.error(f"ERROR: Config file not found: {args.config}")
        return False
    
    # Check if dataset path exists
    if not os.path.exists(args.dataset):
        logging.error(f"ERROR: Dataset path not found: {args.dataset}")
        return False
    
    logging.info("Basic checks passed")
    return True

def main():
    parser = get_args_parser()
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.arch}_w{args.w_bit}_a{args.a_bit}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    log_file = os.path.join(output_dir, 'experiment.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Starting multi-seed experiment")
    logging.info(f"Architecture: {args.arch}")
    logging.info(f"Weight bits: {args.w_bit}")
    logging.info(f"Activation bits: {args.a_bit}")
    logging.info(f"Seeds: {args.seeds}")
    logging.info(f"Alpha values: {args.alpha_list}")
    logging.info(f"Cluster numbers: {args.num_clusters_list}")
    logging.info(f"PCA dimensions: {args.pca_dim_list}")
    logging.info(f"Output directory: {output_dir}")
    
    # Perform pre-flight checks
    if not preflight_check(args):
        logging.error("ERROR: Pre-flight checks failed. Aborting experiment.")
        return
    
    # Run experiments for each seed
    all_results = {}
    all_baseline_results = {}  # Store baseline results separately
    successful_seeds = []
    failed_seeds = []
    
    logging.info(f"\nStarting experiments for {len(args.seeds)} seeds...")
    logging.info(f"Total parameter combinations: {len(args.alpha_list) * len(args.num_clusters_list) * len(args.pca_dim_list)}")
    logging.info(f"Total experiments: {len(args.seeds) * len(args.alpha_list) * len(args.num_clusters_list) * len(args.pca_dim_list)}")
    
    for i, seed in enumerate(args.seeds):
        logging.info(f"\n{'='*60}")
        logging.info(f"Running experiment {i+1}/{len(args.seeds)} for seed {seed}")
        logging.info(f"{'='*60}")
        
        success, error_msg = run_single_experiment(args, seed, output_dir)
        
        if success:
            successful_seeds.append(seed)
            # Parse results from saved files (both reconstructed and baseline)
            results, baseline_results = parse_results_from_file(seed, args)
            all_results[seed] = results
            all_baseline_results[seed] = baseline_results
            logging.info(f"Seed {seed} completed successfully")
        else:
            failed_seeds.append(seed)
            all_results[seed] = []
            all_baseline_results[seed] = []
            logging.error(f"Seed {seed} failed: {error_msg}")
        
        # Sleep between runs
        if i < len(args.seeds) - 1:  # Don't sleep after the last run
            logging.info(f"Sleeping for {args.sleep} seconds before next seed...")
            time.sleep(args.sleep)
    
    logging.info(f"\n{'='*60}")
    logging.info(f"EXPERIMENT SUMMARY")
    logging.info(f"{'='*60}")
    logging.info(f"Successful seeds: {len(successful_seeds)}/{len(args.seeds)}")
    logging.info(f"Failed seeds: {len(failed_seeds)}/{len(args.seeds)}")
    
    if successful_seeds:
        logging.info(f"Successful: {successful_seeds}")
    
    if failed_seeds:
        logging.error(f"Failed: {failed_seeds}")
    
    # Save summary
    if args.save_summary:
        save_summary_results(all_results, all_baseline_results, args, output_dir)
    
    # Comprehensive analysis
    analyze_results_comprehensive(all_results, all_baseline_results, args, output_dir)
    
    # Final summary
    logging.info(f"\n{'='*100}")
    logging.info(f"🎉 ALL EXPERIMENTS COMPLETED!")
    logging.info(f"{'='*100}")
    logging.info(f"Successful seeds: {successful_seeds}")
    logging.info(f"Failed seeds: {failed_seeds}")
    logging.info(f"Results saved to: {output_dir}")
    logging.info(f"{'='*100}")

if __name__ == "__main__":
    main()
