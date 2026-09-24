#!/bin/bash -l                                                                                                           
#SBATCH --job-name=sqd
#SBATCH --account=pi-lgagliardi
#SBATCH --qos=lgagliardi
#SBATCH --time=10:00:00
#SBATCH --nodes=1
##SBATCH --partition=lgagliardi-ld,lgagliardi-amd,bigmem
#SBATCH --partition=lgagliardi-amd
##SBATCH --partition=build
#SBATCH --mem-per-cpu=15G
#SBATCH --cpus-per-task=4
#SBATCH --error=sqsd.err
 
ulimit -s unlimited
export OMP_NUM_THREADS=4
 
conda activate lassqd
# pyscf, pyscf-forge, mrh and lassqd are all installed in the env (see README.md);
# clear these so no other copy on PYTHONPATH can shadow them
export PYTHONPATH=
export PYSCF_EXT_PATH=

python input_pdft.py > fefe_run_pdft.out
