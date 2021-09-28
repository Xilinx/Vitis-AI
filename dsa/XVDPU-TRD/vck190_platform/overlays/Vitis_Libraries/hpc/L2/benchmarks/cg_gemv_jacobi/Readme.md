# GEMV-based Conjugate Gradient Solver with Jacobi Preconditioner

## Introduction

CG solver is widely adopted to solve linear system Ax=b, where the matrix A is symmetric and positive definite. 
Here is the benchmark for GEMV-based CG solver with the Jacobi preconditioner on Xilinx FPGA Alveo U50. 

## Executable Usage

### Environment Setup (Step 1)
Please follow the page [Benchmark Overview](../) to correctly setup the environment first.  

### Build Kernel (Step 2)

With the following commands, kernel bitstream `cgSolver.xclbin ` is built under the directory `./build_dir.hw.xilinx_u50_gen3x16_xdma_201920_3`

```
$ make build TARGET=hw DEVICE=xilinx_u50_gen3x16_xdma_201920_3
```

### Prepare Data (Step 3)
To benchmark the kernel, there two ways to prepare the data. 

#### Randomly-Generated Data (Optional)
You could safely skip this step as it is integrated with the one in the next step if you choose to use random data for the benchmark. 
Here states the principle of how it works.  With the following commands with given vector size e.g. 1024,  three data files are generated under directory `./build_dir.hw.xilinx_u50_gen3x16_xdma_201920_3/data/`.  

1.	It generates a random **SPD** matrix of size `NxN` with data type FP64 and then stores the data in a row-major to file `A.mat`.
2.	It generates a random FP64 vector of size `N` and then compute vector `b = Ax`.
3.	The two vectors are stored into files `x.mat` and `b.mat` respectively. 

Matrix `A` and vector `b` are used as inputs for the solver, and vector `x` is used as the golden reference. 

```
$ make data_gen TARGET=hw DEVICE=xilinx_u50_gen3x16_xdma_201920_3 N=1024
```
where `N` is the vector size and must be multiple of 16.

#### Users` data

Users could prepare their own data for benchmark. 
1.	Please prepare a **SPD** matrix with double precision floating point data type.
2.	Please prepare golden reference vector and result vector which is the product of the matrix and the golden reference.
3.	Please make sure the matrix size is `NxN` and vector size is `N`
4.	Please make sure`N` is multiple of 16.
5.	Please store the matrix, golden reference vector and result vector to binary files named  `A.mat`, `x.mat` and `b.mat` respectively, and place them into a directory.

### Run on FPGA with Example Data (Step 4)

#### Check Device

If you followed the guide and correctly setup the environment, you are able to run the following
command line. You could check whether the target device is prepared and find out the device ID. 
```
$ xbutil scan
```
#### Benchmark Random Dataset

If you decide to use randomly generated data for benchmark in step 3, you could skip that step and run the following command with given vector size `N`, e.g. 1024 and maximum number of iterations for the solver e.g. 100. 

```
$ make run TARGET=hw DEVICE=xilinx_u50_gen3x16_xdma_201920_3 N=1024 maxIter=100 
```
Here lists the configurable parameters with the `make` command for the benchmark. 

| Parameter Name | Default Value | Notes |
|-------------------| -------------- | ---------------|
| N | 1024 | Vector size, must be multiple of 16 |
| maxIter | 100 | Maximum No. iterations for the solver <= 2000|
| tol | 1e-8 | Fault tolerence |
| deviceID | 0 | Alveo U50 Card ID |
| condition_num | 128 | Conditioner number for matrix generated |



#### Usage
For users` own data, follow the usage specified bellow. 
```
    Usage: host.exe <XCLBIN File> <Max Iteration> <Vector Size> <DATA PATH> [device id]
                <XCLBIN File>       path to the xclbin file
                <Max Iteration>     maximum number of iterations
                <Tolerence>         Fault tolerence
                <Vector Size>       size of vector, matrix size N x N
                <DATA PATH>         path to the matrix and vector binary files
                <device id>         Device id given
```


## Resource Utilization
| Name                       | LUT              | LUTAsMem         | REG               | BRAM           | URAM          | DSP            |
|----------------------------|------------------|------------------|-------------------|----------------|---------------|----------------|
| User Budget                | 699619 [100.00%] | 369603 [100.00%] | 1447189 [100.00%] | 1112 [100.00%] | 640 [100.00%] | 5936 [100.00%] |
|    Used Resources          | 186448 [ 26.65%] |  17334 [  4.69%] |  325149 [ 22.47%] |  128 [ 11.51%] |   0 [  0.00%] | 1262 [ 21.26%] |

## Benchmark Results on Alveo U50 FPGA

CPU Hardware information
-   Model name: Intel(R) Xeon(R) CPU E5-2667 v4 @ 3.20GHz
-   Total threads: 32, Threads/Core: 2, Cores/Socket: 8, Total sockets: 2, Total Cores:16

FPGA Hardware Information
- Device name:  Xilinx Alveo U50
- Fmax: 333MHz
- Idle power 24W

| Vector Size |  U50 Performance [GFLOPS] | CPU Performance [GFLOPS] | U50 Energy Efficiency [GFLOPS/W]   |
|---------------|---------------|--------------|--------------|
|	1024	|	26.938	|	12.996	|	0.723	|
|	2048	|	30.658	|	27.469	|	0.766	|
|	4096	|	34.018	|	7.776	|	0.812	|
|	8192	|	36.742	|	8.226	|	0.839	|


### Power Consumption on FPGA
Power data could be obtained by 

```
$ xbutil top -d <DEVICE ID>
```
