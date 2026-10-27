#!/bin/bash
#SBATCH --job-name=causal_tracing  # Job name
#SBATCH --output=/scratch/davide/sbatch_logs/%x_%j.out  # Output file with job name and ID
#SBATCH --error=/scratch/davide/sbatch_logs/%x_%j.err   # Error file with job name and ID
#SBATCH --partition=titans    # Specify the GPU partition
#SBATCH --gres=gpu:1          # Request 1 GPU (adjust if needed)
#SBATCH --cpus-per-task=6     # Number of CPU cores per task
#SBATCH --mem=32G             # Memory per node (adjust as needed)
#SBATCH --time=7-00:00:00     # Set time limit (3 days)
#SBATCH --nodes=1             # Number of nodes required
#SBATCH --mail-type=ALL       # Send email on job start, end, and fail
#SBATCH --mail-user=s220331@student.dtu.dk  # Your email address
#SBATCH --export=ALL

# Activate environment
source ~/.bashrc
source activate my_transformers_env

# Set environment variables
export CUDA_LAUNCH_BLOCKING=1
#export CUDA_VISIBLE_DEVICES=0
#NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)

#export CUDA_VISIBLE_DEVICES
# Run the causal tracing script
srun python causal_tracing_composition.py \
    --dataset # add the full path now !!!! .....composition.2000.200.12.6 \
    --model_dir /dtu-compute/s220331/composition/outputs_BIG_new \
    --save_path /dtu-compute/s220331/composition/CAUSAL_T \
    --num_layer 8 \
    --wd 0.03
