gemvStreamCh16
==============

This example resides in ``L2/benchmarks/streamingKernel/gemvStreamCh16`` directory. The tutorial provides a step-by-step guide that covers commands for building and running kernel. This benchmark performs the matrix-vecotr multiplication, M is number of rows of matrix, N is number of columns of matrix

Executable Usage
-----------------

* **Work Directory(Step 1)**

The steps for library download and environment setup can be found in [here](https://github.com/Xilinx/Vitis_Libraries/tree/master/blas/L2/benchmarks#building). For getting the design,

```
   cd L2/benchmarks/streamingKernel/gemvStreamCh16
```

* **Build kernel(Step 2)** 

Run the following make command to build your XCLBIN and host binary targeting a specific device. Please be noticed that this process will take a long time, maybe couple of hours.

```
    make run TARGET=hw PLATFORM_REPO_PATHS=/opt/xilinx/platforms DEVICE=xilinx_u280_xdma_201920_1
```

* **Run kernel(Step 3)**

To get the benchmark results, please run the following command.

gemvStreamCh16 Input Arguments:

```
    <host application> <xclbin> <m> <n> <path_to_data> device_id
```

For example:

```
    build_dir.hw.xilinx_u280_xdma_201920_1/host.exe build_dir.hw.xilinx_u280_xdma_201920_1/gemv.xclbin 512 256 build_dir.hw.xilinx_u280_xdma_201920_1/data/ 0
```

* **Example output(Step 4)** 

```
Found Platform
Platform Name: Xilinx
INFO: Importing gemv.xclbin
Loading: 'gemv.xclbin'
Software-measured execution time 0.000292705s.
Software-measured HW efficiency 2.09904%.
Execution clock cycles is: 4759
Efficiency is: 43.0343%.
Results verified.
```

Profiling for u280
-------------------

The xclbin could be built in 319 MHz
The hardware resource utilization and benchmark results are shown in the two tables below.

##### Table 1 Hardware resources

| Name                | LUT               | LUTAsMem         | REG               | BRAM           | URAM          | DSP            |
|---------------------|-------------------|------------------|-------------------|----------------|---------------|----------------|
| krnl_gemv           |  122248 [ 10.48%] |  11010 [  1.90%] |  215381 [  9.02%] |   72 [  3.97%] |   0 [  0.00%] |  966 [ 10.71%] |
| streamTimer         |     195 [  0.02%] |      0 [  0.00%] |     291 [  0.01%] |    0 [  0.00%] |   0 [  0.00%] |    0 [  0.00%] |


##### Table 2 Benchmark results 

|  M    |  N    | hw execution time (s) | api execution time (s)  |  execution clock cycles  |  efficiency  |
|-------|-------|-----------------------|-------------------------|--------------------------|--------------|
| 512   | 256   | 1.4316e-05            | 0.00330468              | 4772                     | 42.9173%     |
| 512   | 512   | 1.9998e-05            | 0.00337302              | 6666                     | 61.4461%     | 
| 1024  | 1024  | 6.5904e-05            | 0.0035207               | 21968                    | 74.5812%     |
| 2048  | 2048  | 0.000235251           | 0.00365028              | 78417                    | 83.5737%     |
| 4096  | 4096  | 0.000939699           | 0.00452506              | 313233                   | 83.6898%     |
| 8192  | 8192  | 0.00332612            | 0.0105467               | 1108708                  | 94.5764%     |

Profiling for u50
-------------------

The xclbin could be built in 333 MHz
The hardware resource utilization and benchmark results are shown in the two tables below.

##### Table 1 Hardware resources

| Name                | LUT              | LUTAsMem         | REG               | BRAM           | URAM          | DSP            |
|---------------------|------------------|------------------|-------------------|----------------|---------------|----------------|
| krnl_gemv           | 121535 [ 16.26%] |  11002 [  2.85%] |  215897 [ 13.72%] |   72 [  6.19%] |   0 [  0.00%] |  966 [ 16.27%] |
| streamTimer         |    195 [  0.03%] |      0 [  0.00%] |     291 [  0.02%] |    0 [  0.00%] |   0 [  0.00%] |    0 [  0.00%] |

##### Table 2 Benchmark results 

|  M    |  N    | hw execution time (s) | cold api execution time (s)  | hot api execution time (s) |  execution clock cycles  |  efficiency  |
|-------|-------|-----------------------|------------------------------|----------------------------|--------------------------|--------------|
| 512   | 256   | 1.4481e-05            | 0.000241345                  | 0.00014245                 | 4827                     | 42.428%      |
| 512   | 512   | 2.0853e-05            | 0.000428344                  | 0.000136975                | 6951                     | 58.9268%     |
| 1024  | 1024  | 6.6462e-05            | 0.000439357                  | 0.00017869                 | 22154                    | 73.955%      |
| 2048  | 2048  | 0.000248076           | 0.000637851                  | 0.000367888                | 82692                    | 79.2531%     |
| 4096  | 4096  | 0.000898929           | 0.00156095                   | 0.00101729                 | 299643                   | 87.4854%     |
| 8192  | 8192  | 0.00332855            | 0.00478017                   | 0.00365307                 | 1109516                  | 94.5075%     |

