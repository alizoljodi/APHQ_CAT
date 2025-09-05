#!/bin/bash

# Script to submit all cluster ablation experiments to SLURM
# This script will submit all .sh files in the current directory using sbatch

echo "Starting submission of all cluster ablation experiments..."
echo "Total scripts to submit: $(ls *.sh | grep -v run_all.sh | wc -l)"
echo ""

# Counter for tracking submissions
count=0

# Loop through all .sh files except run_all.sh itself
for script in *.sh; do
    # Skip run_all.sh to avoid submitting itself
    if [ "$script" = "run_all.sh" ]; then
        continue
    fi
    
    # Extract the base name without .sh extension for output file naming
    base_name=$(basename "$script" .sh)
    
    # Create output file name
    output_file="${base_name}.out"
    
    # Submit the job
    echo "Submitting: $script"
    echo "Output file: $output_file"
    
    # Submit with sbatch and capture job ID
    job_id=$(sbatch --output="$output_file" "$script" 2>&1)
    
    if [ $? -eq 0 ]; then
        echo "✓ Successfully submitted: $script (Job ID: $job_id)"
        ((count++))
    else
        echo "✗ Failed to submit: $script"
        echo "Error: $job_id"
    fi
    
    echo "---"
done

echo ""
echo "Submission complete!"
echo "Successfully submitted: $count scripts"
echo ""
echo "To monitor job status, use:"
echo "  squeue -u \$USER"
echo ""
echo "To cancel all jobs, use:"
echo "  scancel -u \$USER"
echo ""
echo "Output files will be created as: {script_name}.out"
