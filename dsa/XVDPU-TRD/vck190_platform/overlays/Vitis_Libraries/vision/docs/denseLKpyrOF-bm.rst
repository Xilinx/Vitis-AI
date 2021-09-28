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


.. _l1_pyroptflow:

===============================
Dense Pyramidal LK Optical Flow
===============================

Dense Pyramidal LK Optical Flow example resides in ``L2/examples/lkdensepyrof`` directory.

This benchmark tests the performance of `lkdensepyrof` function with a pair of images. Optical flow is the pattern of apparent motion of image objects between two consecutive frames, caused by the movement of object or camera. It is a 2D vector field, where each vector is a displacement vector showing the movement of points from first frame to second.

The tutorial provides a step-by-step guide that covers commands for building and running kernel.

Executable Usage
================

* **Work Directory(Step 1)**

The steps for library download and environment setup can be found in README of L2 folder. For getting the design,

.. code-block:: bash

   cd L2/examples/lkdensepyrof

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
   
   -----------Optical Flow Design---------------
   Found Platform
   Platform Name: Xilinx
   XCLBIN File Name: krnl_pyr_dense_optical_flow
   INFO: Importing vision/L2/examples/lkdensepyrof/Xilinx_Lkdensepyrof_L2_Test_vitis_hw_u200/build_dir.hw.xilinx_u200_xdma_201830_2/krnl_pyr_dense_optical_flow.xclbin
   Loading: 'vision/L2/examples/lkdensepyrof/Xilinx_Lkdensepyrof_L2_Test_vitis_hw_u200/build_dir.hw.xilinx_u200_xdma_201830_2/krnl_pyr_dense_optical_flow.xclbin'

    *********Pyr Down Execution*********

    CL buffer created

    data copied to host

    Kernel args set
   opencv

   0 image  0 level pyrdown done

    CL buffer created

    data copied to host

    Kernel args set
   opencv

   0 image  1 level pyrdown done

    CL buffer created

    data copied to host

    Kernel args set
   opencv

   0 image  2 level pyrdown done

    CL buffer created

    data copied to host

    Kernel args set
   opencv

   0 image  3 level pyrdown done

    One image done

    CL buffer created

    data copied to host

    Kernel args set
   opencv

   1 image  0 level pyrdown done

    CL buffer created

    data copied to host

    Kernel args set
   opencv

   1 image  1 level pyrdown done

    CL buffer created

    data copied to host

    Kernel args set
   opencv

   1 image  2 level pyrdown done

    CL buffer created

    data copied to host

    Kernel args set
   opencv

   1 image  3 level pyrdown done

    One image done

    *********Pyr Down Done*********

    *********Starting OF Computation*********

   Buffers created

    *********OF Computation Level = 4*********

    *********OF Computation iteration = 0*********

   Data copied from host to device

   kernel args set

   4 level 0 calls done

    *********OF Computation iteration = 1*********

   Data copied from host to device

   kernel args set

   4 level 1 calls done

    *********OF Computation iteration = 2*********

   Data copied from host to device

   kernel args set

   4 level 2 calls done

    *********OF Computation iteration = 3*********

   Data copied from host to device

   kernel args set

   4 level 3 calls done

    *********OF Computation iteration = 4*********

   Data copied from host to device

   kernel args set

   4 level 4 calls done

   Buffers created

    *********OF Computation Level = 3*********

    *********OF Computation iteration = 0*********

   Data copied from host to device

   kernel args set

   3 level 0 calls done

    *********OF Computation iteration = 1*********

   Data copied from host to device

   kernel args set

   3 level 1 calls done

    *********OF Computation iteration = 2*********

   Data copied from host to device

   kernel args set

   3 level 2 calls done

    *********OF Computation iteration = 3*********

   Data copied from host to device

   kernel args set

   3 level 3 calls done

    *********OF Computation iteration = 4*********

   Data copied from host to device

   kernel args set

   3 level 4 calls done

   Buffers created

    *********OF Computation Level = 2*********

    *********OF Computation iteration = 0*********

   Data copied from host to device

   kernel args set

   2 level 0 calls done

    *********OF Computation iteration = 1*********

   Data copied from host to device

   kernel args set

   2 level 1 calls done

    *********OF Computation iteration = 2*********

   Data copied from host to device

   kernel args set

   2 level 2 calls done

    *********OF Computation iteration = 3*********

   Data copied from host to device

   kernel args set

   2 level 3 calls done

    *********OF Computation iteration = 4*********

   Data copied from host to device

   kernel args set

   2 level 4 calls done

   Buffers created

    *********OF Computation Level = 1*********

    *********OF Computation iteration = 0*********

   Data copied from host to device

   kernel args set

   1 level 0 calls done

    *********OF Computation iteration = 1*********

   Data copied from host to device

   kernel args set

   1 level 1 calls done

    *********OF Computation iteration = 2*********

   Data copied from host to device

   kernel args set

   1 level 2 calls done

    *********OF Computation iteration = 3*********

   Data copied from host to device

   kernel args set

   1 level 3 calls done

    *********OF Computation iteration = 4*********

   Data copied from host to device

   kernel args set

   1 level 4 calls done

   Buffers created

    *********OF Computation Level = 0*********

    *********OF Computation iteration = 0*********

   Data copied from host to device

   kernel args set

   0 level 0 calls done

    *********OF Computation iteration = 1*********

   Data copied from host to device

   kernel args set

   0 level 1 calls done

    *********OF Computation iteration = 2*********

   Data copied from host to device

   kernel args set

   0 level 2 calls done

    *********OF Computation iteration = 3*********

   Data copied from host to device

   kernel args set

   0 level 3 calls done

    *********OF Computation iteration = 4*********

   Data copied from host to device

   kernel args set

   0 level 4 calls done
   ------------------------------------------------------------

Profiling 
==========

The lkdensepyrof design is validated on Alveo U200 board at 300 MHz frequency. 
The hardware resource utilizations are listed in the following table.

.. table:: Table 1 Hardware resources for LK Dense Pyramidal Optical Flow
    :align: center

    +------------------------------------------+-----------------+------------+------------+------------+
    |            Dataset                       |      LUT        |    BRAM    |     FF     |    DSP     |
    +------------+------+----------------------+                 |            |            |            |
    | Resolution | NPPC | other params         |                 |            |            |            |
    +============+======+======================+=================+============+============+============+
    |     4K     |   1  | 5 iterations,        |     30781       |    182     |    26169   |     83     |
    |            |      | 5 levels             |                 |            |            |            |
    +------------+------+----------------------+-----------------+------------+------------+------------+
    |     FHD    |   1  | 5 iterations,        |     30839       |    107     |    25714   |     83     |
    |            |      | 5 levels             |                 |            |            |            |
    +------------+------+----------------------+-----------------+------------+------------+------------+


The performance is shown below

.. table:: Table 2 Performance numbers in terms of FPS (Frames Per Second) for 2 consecutive frames for LK Dense Pyramidal Optical Flow
    :align: center
	
    +----------------------+--------------+--------------+
    |       Dataset        |   FPS(CPU)   |   FPS(FPGA)  |
    +======================+==============+==============+
    |     4k (3840x2160)   |     0.15     |    3         |
    +----------------------+--------------+--------------+
    |   Full HD(1920x1080) |     0.63     |    12        |
    +----------------------+--------------+--------------+

.. toctree::
    :maxdepth: 1
