.. 
   Copyright 2019 Xilinx, Inc.
  
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
  
       http://www.apache.org/licenses/LICENSE-2.0
  
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

.. _l2_manual_jpeg_decoder:

============
JPEG Decoder
============

Jpeg Decoder example resides in ``L2/demos/jpegDec`` directory. The tutorial provides a step-by-step guide that covers commands for building and running kernel.

Executable Usage
===============

* **Work Directory(Step 1)**

The steps for library download and environment setup can be found in :ref:`l2_vitis_codec`. For getting the design,

.. code-block:: bash

   cd L2/demos/jpegDec

* **Build kernel(Step 2)**

Run the following make command to build your XCLBIN and host binary targeting a specific device. Please be noticed that this process will take a long time, maybe couple of hours.

.. code-block:: bash

   make run TARGET=hw DEVICE=xilinx_u50_gen3x16_xdma_201920_3

* **Run kernel(Step 3)**

To get the benchmark results, please run the following command.

.. code-block:: bash

   ./build_dir.hw.xilinx_u50_gen3x16_xdma_201920_3/host.exe -xclbin build_dir.hw.xilinx_u50_gen3x16_xdma_201920_3/jpegDecoder.xclbin -JPEGFile t0.jpg

JPEG Decoder Input Arguments:

.. code-block:: bash

   Usage: host.exe -[-xclbin -JPEGFile]
          -xclbin:    the kernel name
          -JPEGFile:  the path point to input *.jpg

Note: Default arguments are set in Makefile, you can use other :ref:`pictures` listed in the table.

* **Example output(Step 4)** 

.. code-block:: bash

   Found Platform
   Platform Name: Xilinx
   INFO: Found Device=xilinx_u50_gen3x16_xdma_201920_3
   INFO: Importing kernelJpegDecoder.xclbin
   Loading: 'kernelJpegDecoder.xclbin'   
   INFO: Kernel has been created
   INFO: Finish kernel setup
   ...

   INFO: Finish kernel execution
   INFO: Finish E2E execution
   INFO: Data transfer from host to device: 108 us
   INFO: Data transfer from device to host: 726 us
   INFO: Average kernel execution per run: 1515 us
   ...

   INFO: android.yuv will be generated from the jpeg decoder's output
   INFO: android.yuv is generated correctly

Profiling
=========

The hardware resource utilizations are listed in the following table.
Different tool versions may result slightly different resource.


.. table:: Table 1 Hardware resources for jpegDecoder with jfif parser, huffman decoder, IQ&IDCT
    :align: center

    +-----------------------+----------+----------+----------+----------+---------+-----------------+
    |        Kernel         |   BRAM   |   URAM   |    DSP   |    FF    |   LUT   | Frequency(MHz)  |
    +-----------------------+----------+----------+----------+----------+---------+-----------------+
    |      jpegDecoder      |    28    |     0    |    39    |   23653  |  24591  |       243       |
    +-----------------------+----------+----------+----------+----------+---------+-----------------+

To check the output yuv file, download https://sourceforge.net/projects/raw-yuvplayer/ . 
Then upload the rebuild_image.yuv, set the right sample radio and custom size on the software, and check the yuv file.

.. table:: Table 2 Profiling for jpegDecoder kernel
    :align: center

    +------------------+--------------+
    |     Picture      |  Latency(ms) |
    +==================+==============+
    |   android.jpg    |    1.515     |
    +------------------+--------------+
    |    t0.jpg        |    0.790     |
    +------------------+--------------+

.. note::
    | 1. MAX_DEC_PIX is for benchmark. If testcase image is larger than 20M, the value of MAX_DEC_PIX should be enlarged following the size of image.   
    | 2. MAXCMP_BC is for benchmark. If testcase image is larger than 20M, the value of MAXCMP_BC should be enlarged following the size of image.   

.. toctree::
   :maxdepth: 1

