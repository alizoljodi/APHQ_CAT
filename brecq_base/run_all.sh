#!/bin/bash

# Script to run all BRECQ baseline experiments using sbatch
# This script submits all .sh files in the brecq_base directory

echo "Starting BRECQ baseline experiments..."
echo "Submitting all shell scripts in brecq_base/ directory"
echo "=================================================="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Script directory: $SCRIPT_DIR"

# Counter for tracking submissions
count=0

# Loop through all .sh files in the current directory (brecq_base)
for script in "$SCRIPT_DIR"/*.sh; do
    # Skip this run_all.sh script itself
    if [[ "$(basename "$script")" == "run_all.sh" ]]; then
        continue
    fi
    
    # Extract the base name without extension for output file naming
    script_name=$(basename "$script" .sh)
    
    # Create output and error file names
    output_file="${script_name}.out"
    error_file="${script_name}.err"
    
    echo "Submitting: $script"
    echo "  Output file: $output_file"
    echo "  Error file: $error_file"
    
    # Submit the job using sbatch
    sbatch --output="$output_file" --error="$error_file" "$script"
    
    # Check if submission was successful
    if [ $? -eq 0 ]; then
        echo "  ✅ Successfully submitted"
        ((count++))
    else
        echo "  ❌ Failed to submit"
    fi
    
    echo ""
done

echo "=================================================="
echo "Submission Summary:"
echo "  Total scripts submitted: $count"
echo "  All BRECQ baseline experiments have been queued"
echo ""
echo "To monitor job status, use:"
echo "  squeue -u \$USER"
echo ""
echo "To cancel all your jobs, use:"
echo "  scancel -u \$USER"
echo ""
echo "Job output files will be created in this directory:"
echo "  $SCRIPT_DIR"
