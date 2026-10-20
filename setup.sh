#!/bin/bash -i

VENV_DIR=/p/lustre2/${USER}/PU-DDFEM-venv/
module load python/3.13.2

# create if it doesn't exist already, otherwise just activate it
if [ ! -d ${VENV_DIR} ]; then
  conda create --prefix ${VENV_DIR} python==3.13.2

  eval "$(conda shell.bash hook)"
  conda activate ${VENV_DIR}

  conda install -c conda-forge scikit-sparse
else
  eval "$(conda shell.bash hook)"
  conda activate ${VENV_DIR}
fi

python3 -m pip install matplotlib
