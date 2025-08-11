#!/bin/bash
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:ampere:1
#SBATCH -p res-gpu-small
#SBATCH --qos=normal
#SBATCH -t 08:00:00
#SBATCH --job-name=Isla
#SBATCH --mem=28G

#SBATCH -o %x_%j.out            # Set output file name 
#SBATCH -e %x_%j.err            # Set error file name 
#SBATCH --mail-type=ALL         # Send email for all job states
#SBATCH --mail-user=rui.carvalho@durham.ac.uk  # TODO: Update with your email address if different


module purge
# Load CUDA 11.8 with cuDNN 8.7 - specific version to match environment.yml
module load cuda/11.8-cudnn8.7


# Initialize conda
source /home3/grtq36/anaconda3/etc/profile.d/conda.sh

# Activate your conda environment
conda activate new_env

# Add these diagnostic lines
echo "=== Environment Information ==="
echo "Modules loaded:"
module list
echo "Python path: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "CUDA paths:"
echo "PATH: $PATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CUDA_HOME: $CUDA_HOME"
echo ""

echo "=== GPU Information ==="
echo "Node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CUDA Version: $(nvcc --version 2>/dev/null || echo 'nvcc not found')"
echo ""

# Try different GPU detection methods
echo "Method 1: nvidia-smi"
nvidia-smi || echo "nvidia-smi failed"
echo ""

echo "Method 2: nvidia-debugdump"
nvidia-debugdump -l || echo "nvidia-debugdump failed"
echo "======================"

# Run the script with increased CUDA memory settings
python step1.py
if [ $? -ne 0 ]; then
    echo "ERROR: step1.py failed with exit code $?"
    exit 1
fi

python step2_ollama.py
if [ $? -ne 0 ]; then
    echo "ERROR: step2_ollama.py failed with exit code $?"
    exit 1
fi

python step3_ag2.py
if [ $? -ne 0 ]; then
    echo "ERROR: step3_ag2.py failed with exit code $?"
    exit 1
fi

python step4.py
if [ $? -ne 0 ]; then
    echo "ERROR: step4.py failed with exit code $?"
    exit 1
fi

python fine_tune.py
if [ $? -ne 0 ]; then
    echo "ERROR: fine_tune.py failed with exit code $?"

python export_reports.py
if [ $? -ne 0 ]; then
    echo "WARNING: export_reports.py failed with exit code $? (continuing)"
fi

echo "Job finished on $(date)." # Added finish message