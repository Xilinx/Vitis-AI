memKernel
=========

This example resides in ``L3/benchmarks/gemm/memKernel`` directory. The tutorial provides a step-by-step guide that covers commands for building and running kernel. This benchmark performs the matrix-matrix multiplication (A * B = C), M is number of rows of matrix A/C, K is number of columns of matrix A/number of rows of matrix B, N is number of columns of matrix B/C

Executable Usage
-----------------

* **Work Directory(Step 1)**

The steps for library download and environment setup can be found in [here](https://github.com/Xilinx/Vitis_Libraries/tree/master/blas/L2/benchmarks#building). For getting the design,

```
   cd L3/benchmarks/gemm/memKernel
```

* **Build kernel(Step 2)** 

Run the following make command to build your XCLBIN and host binary targeting a specific device. Please be noticed that this process will take a long time, maybe couple of hours.

```
    make run TARGET=hw PLATFORM_REPO_PATHS=/opt/xilinx/platforms DEVICE=xilinx_u250_xdma_201830_2
```

* **Run kernel(Step 3)**

To get the benchmark results, please run the following command.

Input Arguments:

```
    <host application> <xclbin> <config_info.dat>
```

For example:

```
    build_dir.hw.xilinx_u250_xdma_201830_2/gemm_bench.exe build_dir.hw.xilinx_u250_xdma_201830_2/blas.xclbin build_dir.hw.xilinx_u250_xdma_201830_2/config_info.dat
```

* **Example output(Step 4)** 

```
xfblasCreate  276.965961 msec
copyToFpga  0.237744 msec
copyFromFpga  0.753792 msec
Api time is 0.991536 msec
DATA_CSV:,Freq,M,K,N,TimeApiMs,EffApiPct,PerfApiTops
DATA_CSV:,242.000000,64,64,64,0.991536,0.426753,0.000541
>> Kernel #0 << Test passed!
```

* **Use script to run benchmark**

Use mkl to generate dataset, usage of this script is: ./run_gemm_mkl.sh number_of_thread datatype g(generate)/b(benchmark)
Then use run_gemm_bench.sh to run benchmark
```
    cd ../gemm_mkl
    ./run_gemm_mkl.sh 16 float g
    ./run_gemm_bench.sh build_dir.hw.xilinx_u250_xdma_201830_2/blas.xclbin build_dir.hw.xilinx_u250_xdma_201830_2/config_info.dat
```

Profiling
----------

The xclbin could be built in 242 MHz
The hardware resource utilization and benchmark results are shown in the two tables below.

##### Table 1 Hardware resources

|    Name    |   LUT    |  BRAM  |  URAM |   DSP  |    FF   |
|------------|----------|--------|-------|--------|---------|
| blasKernel | 250679   | 94     | 24    | 1224   | 430512  |


##### Table 2 Benchmark results

|  M   |  N   |  K   |  api execution time [ms]   | api Eff [%]  |  PerfApiTops  |
|------|------|------|----------------------------|--------------|---------------|
| 256  | 256  | 256  | 2.295277                   | 11.798572    | 0.058818      |
| 512  | 512  | 512  | 7.185994                   | 30.148638    | 0.149859      |
| 1024 | 1024 | 1024 | 33.357721                  | 51.957490    | 0.257887      |
| 2048 | 2048 | 2048 | 218.662946                 | 63.410230    | 0.314501      |
| 4096 | 4096 | 4096 | 1594.648667                | 69.559988    | 0.344877      |
| 8192 | 8192 | 8192 | 12695.637510               | 69.897233    | 0.346485      |
