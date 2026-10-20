#!/bin/bash
#SBATCH -J online
#SBATCH -p pbatch
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -t 01:00:00
#SBATCH -A genddfem
#SBATCH --exclusive
#SBATCH --mem=2G

medium_type="single_null"
#medium_type="double_null"
#medium_type="magnetic_island"

order=2
n_FOM_x=256
Nx=32
num_mode=20
num_loc_basis=25

VENV_DIR=/p/lustre2/${USER}/PU-DDFEM-venv/
module load python/3.13.2
eval "$(conda shell.bash hook)"
conda activate ${VENV_DIR}

python3 online.py --root_dir fourier_mode_harmonic_extension --train_anisotropy_order ${order} --num_loc_bc_mode ${num_mode} --medium_type ${medium_type} --test_anisotropy_order ${order} --n_FOM_x ${n_FOM_x} --Nx ${Nx} --num_loc_basis ${num_loc_basis}
