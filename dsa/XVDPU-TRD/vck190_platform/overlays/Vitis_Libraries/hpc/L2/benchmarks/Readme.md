# Benchmark Overview

Here are benchmarks of the Vitis HPC Library using the Vitis environment and comparing results on
several FPGA and CPU platforms. It supports software and hardware emulation as well as running hardware accelerators on the Alveo U250, U280 or U50.

## Prerequisites

### Vitis HPC Library

- Vitis BLAS library is required to build any projects in Vitis HPC library
-  According to the benchmark application, Alveo U250, U280 or U50 need to be installed and correctly configured.
- Xilinx runtime (XRT) installed
- Xilinx Vitis 2021.1 installed and configured

## Building

Here is an example to indicate how to build the application and kernel with the command line Makefile flow.

### Download code

These benchmarks can be downloaded from [vitis libraries](https://github.com/Xilinx/Vitis_Libraries.git) ``master`` branch.

```
   git clone https://github.com/Xilinx/Vitis_Libraries.git
   cd Vitis_Libraries
   git checkout master
   cd hpc
```

### Setup environment

Specifying the corresponding Vitis, XRT, and path to the platform repository by running following commands.

```
   source <intstall_path>/installs/lin64/Vitis/2020.2/settings64.sh
   source /opt/xilinx/xrt/setup.sh
   export PLATFORM_REPO_PATHS=/opt/xilinx/platforms
```

### Python 3.6+
Follow the steps as per https://xilinx.github.io/Vitis_Libraries/blas/2021.1/user_guide/L1/pyenvguide.html to set up Python3 environment.

### Datasets
- For dense matrix conjugate gradient solver, data are randomly generated during benchmark
- For sparse matrix conjugate gradient solver, datasets from https://sparse.tamu.edu/ will be downloaded during benchmarking process
- Matrix files: binary data files or Matrix Market (.mtx) files 


## Benchmarks

### Conjugate Gradient Method

Conjugate Gradient (CG) Method is widely adopted in industry to solve linear system Ax=b, where the matrix A is symmetric and positive definite. Here are the benchmarks of CG solver for two types of linear system with the Jacobi preconditioners on Alveo FPGAs.

#### GEMV-Based Linear System

For dense matrix, the General matrix-vector multiplication is implemented in the Solver.  
Here is the benchmark for a randomly generated matrix with size 2048x2048. 

| Vector Dim    | Time per Iteration on U50 [ms] |   GFLOPS/W   |
|---------------|-----------------------------------|--------------|
|2048 			| 0.2559 	| 0.766	|

More details and benchmark results could be found in the following directory [cg_gemv_jacobi](./cg_gemv_jacobi/). 

#### SPMV-Based Linear System

For sparse matrix, the Sparse matrix-vector multiplication is implemented in the Solver. 
Here is the benchmark for the sparse matrix named nasa2910. 

|   Matrix name |    Rows/Cols  |    NNZs   |    No. Iters  |   Total run time [sec]    |    Time per Iter [ms] |
|   --------------- |   --------------- |   --------------- |   --------------- |   --------------- |   --------------- |
|   nasa2910    |   2910    |   174296  |   1340    |   0.0684992   |   0.0511188   |

More details and benchmark results could be found in the following directory [cg_spmv_jacobi](./cg_spmv_jacobi/)
