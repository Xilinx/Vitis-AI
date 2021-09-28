streamingKernel
================

This example resides in ``L3/benchmarks/gemm/streamingKernel`` directory. The tutorial provides a step-by-step guide that covers commands for building and running kernel. This benchmark performs the matrix-matrix multiplication (A * B = C), M is number of rows of matrix A/C, K is number of columns of matrix A/number of rows of matrix B, N is number of columns of matrix B/C

Executable Usage
-----------------

* **Work Directory(Step 1)**

The steps for library download and environment setup can be found in [here](https://github.com/Xilinx/Vitis_Libraries/tree/master/blas/L2/benchmarks#building). For getting the design,

```
   cd L3/benchmarks/gemm/streamingKernel
```

* **Build kernel(Step 2)** 

Run the following make command to build your XCLBIN and host binary targeting a specific device. Please be noticed that this process will take a long time, maybe couple of hours.

```
    make run TARGET=hw PLATFORM_REPO_PATHS=/opt/xilinx/platforms DEVICE=xilinx_u250_gen3x16_xdma_3_1_202020_1
```

* **Run kernel(Step 3)**

To get the benchmark results, please run the following command.

Input Arguments:

```
    <host application> <xclbin> <config_info.dat>
```

For example:

```
    build_dir.hw.xilinx_u250_gen3x16_xdma_3_1_202020_1/gemm_bench.exe build_dir.hw.xilinx_u250_gen3x16_xdma_3_1_202020_1/blas.xclbin build_dir.hw.xilinx_u250_gen3x16_xdma_3_1_202020_1/config_info.dat
```

* **Example output(Step 4)** 

```
xfblasCreate  249.914832 msec
copyToFpga  0.243765 msec
copyFromFpga  0.437556 msec
Api time is 0.681321 msec
DATA_CSV:,Freq,M,K,N,TimeApiMs,EffApiPct,PerfApiTops
DATA_CSV:,250.000000,64,64,64,0.681321,0.601185,0.000788
>> Kernel #0 << Test passed!
```

* **Use script to run benchmark**

Use mkl to generate dataset, usage of this script is: ./run_gemm_mkl.sh number_of_thread datatype g(generate)/b(benchmark)
Then use run_gemm_bench.sh to run benchmark
```
    cd ../gemm_mkl
    ./run_gemm_mkl.sh 16 float g
    ./run_gemm_bench.sh build_dir.hw.xilinx_u250_gen3x16_xdma_3_1_202020_1/blas.xclbin build_dir.hw.xilinx_u250_gen3x16_xdma_3_1_202020_1/config_info.dat
```

Profiling
----------

The xclbin could be built in 250 MHz
The hardware resource utilization and benchmark results are shown in the two tables below.

##### Table 1 Hardware resources

|    Name                 |      LUT     |    BRAM   |   URAM   |   DSP  |      REG   |
|-------------------------|--------------|-----------|----------|--------|------------|
| gemmAddsKernel          | 101988       | 0         | 0        | 384    | 192516     |
| gemmCPlusXKernel        | 8529         | 24        | 0        | 66     | 20358      |
| gemmLoadStoreKernel     | 7126         | 23        | 0        | 16     | 19457      |
| gemmMergeKernel         | 8342         | 0         | 0        | 0      | 25219      |
| gemmMulsKernel          | 50640        | 0         | 0        | 768    | 98013      |
| gemmSystolicArrayKernel | 2541         | 0         | 0        | 0      | 240        |
| gemmTagsKernel          | 20203        | 15        | 0        | 8      | 34678      |
| gemmTimerKernel         | 32           | 0         | 0        | 0      | 115        |


##### Table 2 Benchmark results

|  M   |  N   |  K   |  api execution time [ms]   | api Eff [%]  |  PerfApiTops  |
|------|------|------|----------------------------|--------------|---------------|
| 256  | 256  | 256  | 1.370527                   | 19.127241    | 0.024626      |
| 512  | 512  | 512  | 4.517989                   | 46.417820    | 0.059589      |
| 1024 | 1024 | 1024 | 29.500145                  | 56.871639    | 0.072902      |
| 2048 | 2048 | 2048 | 217.555482                 | 61.693563    | 0.079026      |
| 4096 | 4096 | 4096 | 1685.337895                | 63.710774    | 0.081580      |
