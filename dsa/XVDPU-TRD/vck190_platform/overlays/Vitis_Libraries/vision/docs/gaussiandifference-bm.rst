.. 
   Copyright 2021 Xilinx, Inc.
  
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
  
       http://www.apache.org/licenses/LICENSE-2.0
  
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.


.. _l3_gaussiandifference:

====================
Gaussian Difference
====================

Gaussian Difference example resides in ``L3/examples/gaussiandifference`` directory.

This benchmark tests the performance of `gaussiandifference` function. The Difference of Gaussian Filter function can be implemented by applying Gaussian Filter on the original source image, and that Gaussian blurred image is duplicated as two images. The Gaussian blur function is applied to one of the duplicated images, whereas the other one is stored as it is. Later, perform the Subtraction function on, two times Gaussian applied image and one of the duplicated image.

The tutorial provides a step-by-step guide that covers commands for building and running kernel.

Executable Usage
================

* **Work Directory(Step 1)**

The steps for library download and environment setup can be found in README file of L3 folder. For getting the design,

.. code-block:: bash

   cd L3/example/gaussiandifference

* **Build kernel(Step 2)**

Run the following make command to build your XCLBIN and host binary targeting a specific device. Please be noticed that this process will take a long time, maybe couple of hours.

.. code-block:: bash

   export OPENCV_INCLUDE=< path-to-opencv-include-folder >
   export OPENCV_LIB=< path-to-opencv-lib-folder >
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:< path-to-opencv-lib-folder > 
   export DEVICE=< path-to-platform-directory >/< platform >.xpfm
   make host xclbin TARGET=hw


* **Run kernel(Step 3)**

To get the benchmark results, please run the following command.

.. code-block:: bash

   make run TARGET=hw 


* **Example output(Step 4)** 

.. code-block:: bash
   
   -----------Guassian Difference Design---------------
   INFO: Running OpenCL section.
   Found Platform
   Platform Name: Xilinx
   INFO: Device found - xilinx_u200_xdma_201830_2
   XCLBIN File Name: krnl_gaussiandifference
   INFO: Importing Vitis_Libraries/vision/L3/examples/gaussiandifference/Xilinx_Gaussiandifference_L3_Test_vitis_hw_u200/build_dir.hw.xilinx_u200_xdma_201830_2/krnl_gaussiandifference.xclbin
   Loading: 'Vitis_Libraries/vision/L3/examples/gaussiandifference/Xilinx_Gaussiandifference_L3_Test_vitis_hw_u200/build_dir.hw.xilinx_u200_xdma_201830_2/krnl_gaussiandifference.xclbin'
   Test Passed 
   ------------------------------------------------------------

Profiling 
=========

The Gaussian Difference design is validated on Alveo U200 board at 300 MHz frequency. 
The hardware resource utilizations are listed in the following table.

.. table:: Table 1 Hardware resources for Gaussian Difference
    :align: center

    +------------------------------------------+-----------------+------------+------------+------------+
    |            Dataset                       |      LUT        |    BRAM    |     FF     |    DSP     |
    +------------+------+----------------------+                 |            |            |            |
    | Resolution | NPPC | other params         |                 |            |            |            |
    +============+======+======================+=================+============+============+============+
    |     4K     |   1  |      Filter - 3x3    |     27235       |     35     |    37426   |     281    |
    +------------+------+----------------------+-----------------+------------+------------+------------+
    |     FHD    |   1  |      Filter - 3x3    |     15146       |     19     |    22032   |     193    |
    +------------+------+----------------------+-----------------+------------+------------+------------+


The performance is shown below

.. table:: Table 2 Performance numbers in terms of FPS (Frames Per Second) for Gaussian Difference
    :align: center
	
    +----------------------+--------------+--------------+
    |       Dataset        |   FPS(CPU)   |   FPS(FPGA)  |
    +======================+==============+==============+
    |     4k (3840x2160)   |     126      |     289      |
    +----------------------+--------------+--------------+
    |   Full HD(1920x1080) |     500      |     1100     |
    +----------------------+--------------+--------------+


.. toctree::
    :maxdepth: 1
