#!/bin/bash
#SBATCH --partition=a100-cu117
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --job-name=würstchen
#SBATCH --comment proj

export NCCL_PROTO=simple

export FI_EFA_FORK_SAFE=1
export FI_LOG_LEVEL=1
export FI_EFA_USE_DEVICE_RDMA=1 # use for p4dn

export NCCL_DEBUG=info
export PYTHONFAULTHANDLER=1

export CUDA_LAUNCH_BLOCKING=0
export OMPI_MCA_mtl_base_verbose=1
export FI_EFA_ENABLE_SHM_TRANSFER=0
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_TREE_THRESHOLD=0

export PYTHONWARNINGS="ignore"
export CXX=g++

eval "$(conda shell.bash hook)"
conda activate env
cd würstchen
rm dist_file4
srun --comment proj python3 train_stage_B.py



















