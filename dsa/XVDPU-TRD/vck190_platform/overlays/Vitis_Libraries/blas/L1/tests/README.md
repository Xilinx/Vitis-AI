This folder contains Vivado-HLS-based unit tests for kernel primitives.

To test l1 primitives, please follow the steps below:
  1. navigate to xf_blas/L1/tests
  2. source ./set_env.sh
  3. run testbench via python scripts and profile. For example: 
    python ./sw/python/run_test.py ./hw/amax/profile.json 
    python ./sw/python/run_test.py ./hw/asum/profile.json 
    python ./sw/python/run_test.py ./hw/axpy/profile.json 
