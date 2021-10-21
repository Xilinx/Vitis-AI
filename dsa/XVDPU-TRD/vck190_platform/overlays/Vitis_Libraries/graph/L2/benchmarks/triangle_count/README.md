Triangle Counting
=================

Triangle Counting example resides in ``L2/benchmarks/triangle_count`` directory. The tutorial provides a step-by-step guide that covers commands for building and running kernel.

Executable Usage
----------------

* **Work Directory(Step 1)**

The steps for library download and environment setup can be found in [here](https://github.com/Xilinx/Vitis_Libraries/tree/master/graph/L2/benchmarks#building). For getting the design,

```
   cd L2/benchmarks/triangle_count
```   

* **Build kernel(Step 2)**

Run the following make command to build your XCLBIN and host binary targeting a specific device. Please be noticed that this process will take a long time, maybe couple of hours.

```
   make run TARGET=hw DEVICE=xilinx_u250_xdma_201830_2
```

* **Run kernel(Step 3)**

To get the benchmark results, please run the following command.

```
   ./build_dir.hw.xilinx_u250_xdma_201830_2/host.exe -xclbin build_dir.hw.xilinx_u250_xdma_201830_2/TC_kernel.xclbin -o data/csr_offsets.txt -i data/csr_columns.txt 
```   

Triangle Counting Input Arguments:

```
   Usage: host.exe -[-xclbin -o -i]
         -xclbin   triangle count binary
         -o        offset file of input graph in CSR format
         -i        edge file of input graph in CSR format
```          

Note: Default arguments are set in Makefile, you can use other [datasets](https://github.com/Xilinx/Vitis_Libraries/tree/master/graph/L2/benchmarks#datasets) listed in the table.

* **Example output(Step 4)**        
        
```
   ---------------------Triangle Count-----------------
   vertex 2,max_diff=4
   Found Platform
   Platform Name: Xilinx
   Found Device=xilinx_u250_xdma_201830_2
   INFO: Importing build_dir.hw.xilinx_u250_xdma_201830_2/TC_kernel.xclbin
   Loading: 'build_dir.hw.xilinx_u250_xdma_201830_2/TC_kernel.xclbin'
   kernel has been created
   ...

   kernel start------

   ...
   kernel end------
   Execution time 69.292ms
   Write DDR Execution time 62.7083 ms
   Kernel Execution time 6.32296 ms
   Read DDR Execution time 0.098528 ms
   Total Execution time 69.1972 ms
   INFO: case pass!
```

Profiling
---------

The hardware resource utilizations are listed in the following table.

###### Table 1 Hardware resources

|  Kernel       |   BRAM   |   URAM   |    DSP   |   LUT   | Frequency(MHz)  |
|---------------|----------|----------|----------|---------|-----------------|
|  TC_Kernel    |    62    |    16    |    0     |  21001  |      300        |

The performance is shown in the table below.

##### Table 2 Comparison between CPU and FPGA

| Datasets         | Vertex   | Edges    | FPGA time | Spark time(4 threads) |  speedup | Spark time(8 threads) |  speedup | Spark time(16 threads) |  speedup | Spark time(32 threads) |  speedup |
|------------------|----------|----------|-----------|------------|----------|------------|----------|------------|----------|------------|----------|
| as-Skitter       | 1694616  | 11094209 |  53.05    |  46.5      |   0.88   |  31.30     |   0.59   |  25.66     |   0.48   |  26.60     |   0.50   |
| coPapersDBLP     | 540486   | 15245729 |   4.37    |  68.0      |  15.55   |  42.08     |   9.63   |  29.55     |   6.76   |  33.15     |   7.59   |
| coPapersCiteseer | 434102   | 16036720 |   6.80    |  74.4      |  10.94   |  38.74     |   5.70   |  37.42     |   5.50   |  33.87     |   4.98   |
| cit-Patents      | 3774768  | 16518948 |   0.80    |  75.8      |  95.10   |  57.20     |  71.50   |  44.87     |  56.09   |  39.61     |  49.51   |
| europe_osm       | 50912018 | 54054660 |   1.08    |  577.1     | 534.07   | 295.57     | 273.68   | 221.86     | 205.43   | 144.68     | 133.96   |
| hollywood        | 1139905  | 57515616 | 113.48    |  395.0     |   3.49   | 246.42     |   2.17   | 220.90     |   1.95   |    --      |    --    |
| soc-LiveJournal1 | 4847571  | 68993773 |  21.17    |  194.3     |   9.18   | 121.15     |   5.72   | 104.64     |   4.94   | 149.34     |   7.05   |
| ljournal-2008    | 5363260  | 79023142 |  19.73    |  223.5     |  11.33   | 146.63     |   7.43   | 171.35     |   8.68   |    --      |    --    |
| GEOMEAN          |          |          |   9.47    |  143.2     |  15.1X   |  88.54     |   9.4X   |  76.05     |   8.0X   |  54.27     |   9.8X   |

##### Note
```
    1. Spark time is the execution time of funciton "TriangleCount.runPreCanonicalized".
    2. Spark running on platform with Intel(R) Xeon(R) CPU E5-2690 v4 @2.600GHz, 56 Threads (2 Sockets, 14 Core(s) per socket, 2 Thread(s) per core).
    3. time unit: second.
    4. "-" Indicates that the result could not be obtained due to insufficient memory.
```
