JPEG Decoder
============

Jpeg Decoder example resides in ``L2/demos/jpegDec`` directory. The tutorial provides a step-by-step guide that covers commands for building and running kernel.

Executable Usage
----------------

* **Work Directory(Step 1)**

The steps for library download and environment setup can be found in :ref:`l2_vitis_codec`. For getting the design,

```
   cd L2/demos/jpegDec
```

* **Build kernel(Step 2)**

Run the following make command to build your XCLBIN and host binary targeting a specific device. Please be noticed that this process will take a long time, maybe couple of hours.

```
   make run TARGET=hw DEVICE=xilinx_u250_xdma_201830_2
```   

* **Run kernel(Step 3)**

To get the benchmark results, please run the following command.

```
   ./build_dir.hw.xilinx_u250_xdma_201830_2/host.exe -xclbin build_dir.hw.xilinx_u250_xdma_201830_2/jpegDecoder.xclbin -JPEGFile android.jpg
```   

JPEG Decoder Input Arguments:

```
   Usage: host.exe -[-xclbin -dataSetDir -refDir]
          -xclbin:    the kernel name
          -JPEGFile:  the path point to input *.jpg
```          

Note: Default arguments are set in Makefile, you can use other :ref:`pictures` listed in the table.

* **Example output(Step 4)** 

```
   Found Platform
   Platform Name: Xilinx
   INFO: Found Device=xilinx_u250_xdma_201830_2
   INFO: Importing build_dir.hw.xilinx_u250_xdma_201830_2/jpegDecoder.xclbin
   Loading: 'build_dir.hw.xilinx_u250_xdma_201830_2/jpegDecoder.xclbin'
   INFO: Kernel has been created
   INFO: Finish kernel setup
   ...

   INFO: Finish kernel execution
   INFO: Finish E2E execution
   INFO: Data transfer from host to device: 40 us
   INFO: Data transfer from device to host: 6 us
   INFO: Average kernel execution per run: 988 us
   ...

   INFO: android.yuv will be generated from the jpeg decoder's output   oINFO: android.yuv is generated correctly
   INFO: android.yuv is generated correctly
```   

Profiling
---------

The hardware resource utilizations are listed in the following table.
Different tool versions may result slightly different resource.

##### Table 1 IP resources for jpegDecoder with huffman decoder(L1 IP)

|           IP          |   BRAM   |   URAM   |    DSP   |    FF    |   LUT   | Frequency(MHz)  |
|-----------------------|----------|----------|----------|----------|---------|-----------------|
|     huffman_decoder   |     5    |     0    |    12    |    6963  |   7344  |       286       |

##### Table 2 IP resources for jpegDecoder with jfif parser and huffman decoder(L1 IP)

|           IP          |   BRAM   |   URAM   |    DSP   |    FF    |   LUT   | Frequency(MHz)  |
|-----------------------|----------|----------|----------|----------|---------|-----------------|
| kernel_parser_decoder |     5    |     0    |    12    |    7615  |   8382  |       257       |

##### Table 3 Hardware resources for jpegDecoder with jfif parser, huffman, iq and idct (L2 kernel)   

|        Kernel         |   BRAM   |   URAM   |    DSP   |    FF    |   LUT   | Frequency(MHz)  |
|-----------------------|----------|----------|----------|----------|---------|-----------------|
|      jpegDecoder      |     7    |     0    |    39    |   12298  |  13417  |       257       |

Result
------

To check the output yuv file, download https://sourceforge.net/projects/raw-yuvplayer/ . 
Then upload the rebuild_image.yuv, set the right sample radio and custom size on the software, and check the yuv file.

Table 1 : Jpeg Decoder profiling

![Table 1 : Jpeg Decoder profiling](../../../docs/images/jpegDecoderpofile.png)

##### Note      
```      
    | 1. MAX_DEC_PIX is for benchmark. If testcase image is larger than 20M, the value of MAX_DEC_PIX should be enlarged following the size of image.   
    | 2. MAXCMP_BC is for benchmark. If testcase image is larger than 20M, the value of MAXCMP_BC should be enlarged following the size of image.   
```
