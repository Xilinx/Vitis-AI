Strongly Connected Component
============================

Strongly Connected Component example resides in ``L2/benchmarks/strongly_connected_component`` directory. The tutorial provides a step-by-step guide that covers commands for building and running kernel. 

Executable Usage
---------------

* **Work Directory(Step 1)**

The steps for library download and environment setup can be found in [here](https://github.com/Xilinx/Vitis_Libraries/tree/master/graph/L2/benchmarks#building). For getting the design,

```
   cd L2/benchmarks/strongly_connected_component
```   
  
* **Build kernel(Step 2)**

Run the following make command to build your XCLBIN and host binary targeting a specific device. Please be noticed that this process will take a long time, maybe couple of hours.

```
   make run TARGET=hw DEVICE=xilinx_u250_xdma_201830_2
```   

* **Run kernel(Step 3)**

To get the benchmark results, please run the following command.

```
   ./build_dir.hw.xilinx_u250_xdma_201830_2/host.exe -xclbin build_dir.hw.xilinx_u250_xdma_201830_2/scc_kernel.xclbin -o data/test_offset.csr -c data/test_column.csr -g data/test_golden.mtx
```   

Strongly Connected Component Input Arguments:

```
   Usage: host.exe -[-xclbin -o -c -g]
         -xclbin     strongly connected component binary
         -o          offset file of input graph in CSR format
         -c          edge file of input graph in CSR format
         -g          golden reference file for validatation
```          

Note: Default arguments are set in Makefile, you can use other [datasets](https://github.com/Xilinx/Vitis_Libraries/tree/master/graph/L2/benchmarks#datasets) listed in the table.

* **Example output(Step 4)**

```
   ---------------------SCC Test----------------
   Found Platform
   Platform Name: Xilinx
   Found Device=xilinx_u250_xdma_201830_2
   INFO: Importing build_dir.hw.xilinx_u250_xdma_201830_2/scc_kernel.xclbi
   Loading: 'build_dir.hw.xilinx_u250_xdma_201830_2/scc_kernel.xclbin'
   kernel has been created
   kernel start------
   Input: numVertex=13, numEdges=19
   kernel end------
   Execution time 53.929ms
   Write DDR Execution time 0.115905ms
   Kernel Execution time 53.37ms
   Read DDR Execution time 0.039641ms
   Total Execution time 53.5255ms
   HW components:6
   The number of components:6
   Check Passed.
```

Profiling
----------

The kernel is built by Vivado tools and benchmard in U250 FPGA card at 275MHz. The hardware resource utilization and benchmark performance are listed in the table below.

##### Table 1 Hardware resources

|    Name    |      LUT     |     BRAM    |   URAM   |   DSP   |
|------------|--------------|-------------|----------|---------|
|  Platform  |    104112    |     165     |     0    |    4    |
| scc_kernel |    164311    |    523.5    |    110   |    6    |
|    Total   | 268423 (16%) | 688.5 (26%) | 110 (9%) | 10 (0%) |

##### Table 2 Comparison between spark on CPU and FPGA

|    Datasets      |  Vertex |  Edges  | Number of SCC | Iteration number in Spark | FPGA Time | Spark Time(4 threads) | Speed up | Spark Time(8 threads) | Speed up | Spark Time(16 threads) | Speed up | Spark Time(32 threads) | Speed up |
|------------------|---------|----------|---------|-----------|-----------|------------|----------|------------|----------|------------|----------|------------|----------|
|    cit-Patents   | 3774768 | 16518948 | 3774768 |     6     |   20711   |    52137   |   2.52   |    60517   |   2.92   |    51390   |   2.48   |    39939   |   1.93   |
|     hollywood    | 1139905 | 57515616 | 1139905 |     6     |    9780   |    75681   |   7.74   |    45935   |   4.70   |    39595   |   4.05   |    29665   |   3.03   |
| soc-LiveJournal1 | 4847571 | 68993773 | 971232  |     6     |   39952   |   424444   |   10.62  |   304755   |   7.63   |   244916   |   6.13   |   231465   |   5.79   |
|   ljournal-2008  | 5363260 | 79023142 | 1119171 |     16    |   34840   |   540199   |   15.51  |   458633   |   13.16  |   378304   |   10.86  |   402120   |   11.54  |
|      GEOMEAN     |         |          |         |           |   23043   |   173431   |   7.53X  |   140397   |   6.09X  |   117178   |   5.09X  |   102476   |   4.45X  |

##### Note
```
    1. Spark running on platform with Intel(R) Xeon(R) CPU E5-2690 v4 @2.600GHz, 56 Threads (2 Sockets, 14 Core(s) per socket, 2 Thread(s) per core)
    2. Time unit: ms.
```
