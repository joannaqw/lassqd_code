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
 
export PYTHONPATH=
export PYSCF_EXT_PATH=
export PYSCF=/home/qjwang/.local/lib/python3.9/site-packages/pyscf
export MRH=/project/lgagliardi/qjwang:/project/lgagliardi/qjwang/mrh
export SHARED_PYSCF_ext=$SHARED_APPS/pyscf_ext/pyscf-forge
export PYSCF_EXT_PATH=$SHARED_PYSCF_ext
export PYTHONPATH=$MRH$PYSCF:$SHARED_PYSCF_ext:$PYTHONPATH

python input_pdft.py > fefe_run_pdft.out
