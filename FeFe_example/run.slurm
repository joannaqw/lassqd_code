#!/bin/bash -l            
#SBATCH --job-name=%e-3 
#SBATCH --account=pi-lgagliardi
#SBATCH --qos=lgagliardi
#SBATCH --time=130:00:00
#SBATCH --nodes=1
##SBATCH --partition=lgagliardi-ld,lgagliardi-amd,bigmem
#SBATCH --partition=lgagliardi-ld,lgagliardi-hm
##SBATCH --partition=build
#SBATCH --mem-per-cpu=10G
#SBATCH --cpus-per-task=14
#SBATCH --error=sqd.err
 
ulimit -s unlimited
export OMP_NUM_THREADS=14
#export PYSCF_MAX_MEMORY=180000
conda activate lassqd
#module load mkl/2023.1
# pyscf, pyscf-forge, mrh and lassqd are all installed in the env (see README.md);
# clear these so no other copy on PYTHONPATH can shadow them
export PYTHONPATH=
export PYSCF_EXT_PATH=

python input.py > lassqd_max.out
