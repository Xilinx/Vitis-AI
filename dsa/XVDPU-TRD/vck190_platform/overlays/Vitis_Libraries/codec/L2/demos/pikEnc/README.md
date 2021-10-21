PIK Encoder
===============

PIK Encoder example resides in ``L2/demos/pikEnc`` directory. The tutorial provides a step-by-step guide that covers commands for building and running kernel.

Executable Usage
----------------

* **Work Directory(Step 1)**

The steps for library download and environment setup can be found in :ref:`l2_vitis_codec`. For getting the design,

```
   cd L2/demos/pikEnc
```

* **Build kernel(Step 2)**

Run the following make command to build your XCLBIN and host binary targeting a specific device. Please be noticed that this process will take a long time, maybe couple of hours.

```
   make run TARGET=hw DEVICE=xilinx_u200_xdma_201830_2
```   

* **Run kernel(Step 3)**

To get the benchmark results, please run the following command.

```
   ./build_dir.hw.xilinx_u200_xdma_201830_2/host.exe --xclbin build_dir.hw.xilinx_u200_xdma_201830_2/pikEnc.xclbin PNGFilePath PIKFilePath --fast
```   

PIK Encoder Input Arguments:

```
   Usage: host.exe -[-xclbin]
          --xclbin:         the kernel name
          --fast:           the encoding mode
          PNGFilePath:      the path to the input *.PNG
          PIKFilePath:  the path to the output *.pik
```          

Note: Default arguments are set in Makefile, you can use other :ref:`pictures` listed in the table.

* **Example output(Step 4)** 

```
   Found Platform
   Platform Name: Xilinx
   INFO: Found Device=xilinx_u200_xdma_201830_2
   INFO: Importing build_dir.hw.xilinx_u200_xdma_201830_2/pikEnc.xclbin
   Loading: 'build_dir.hw.xilinx_u200_xdma_201830_2/pikEnc.xclbin'
   INFO: Kernel has been created
   INFO: Finish kernel setup
   ...

   INFO: Finish kernel execution
   INFO: Finish E2E execution
   INFO: Data transfer from host to device: 100 us
   INFO: Data transfer from device to host: 20 us
   INFO: Average kernel execution per run: 600 ms
```

Profiling
---------

The hardware resource utilizations are listed in the following table.
Different tool versions may result slightly different resource.


##### Table 1 IP resources for PIK encoder 

|      IP       |   BRAM   |   URAM   |    DSP   |    FF    |   LUT   |
|---------------|----------|----------|----------|----------|---------|
|    Kernel1    |    25    |    93    |    568   |   125920 |  97441  |
|    Kernel2    |    411   |    252   |    1614  |   309222 |  262543 |
|    Kernel3    |    178   |    128   |    216   |   114845 |  90011  |


##### Table 2 PIK Encoder Performance
      
|   Size\Time(ms)  |  Kernel1  |  Kernel2  |  Kernel3  |
|------------------|-----------|-----------|-----------|
|     512x512      |    16     |    14     |     7     |
|    1024x1024     |    52     |    48     |    24     |
|    2048x2048     |    191    |    180    |    86     |


