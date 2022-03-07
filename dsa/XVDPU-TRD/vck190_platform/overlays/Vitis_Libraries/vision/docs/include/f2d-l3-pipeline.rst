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


.. _l1_f2daiel3:

===============================
Filter2D Pipeline on Multiple AIE Cores
===============================

This example demonstrates how a function/pipeline of functions can run on multiple AIE cores to achieve higher throughput. Back-to-back Filter2D pipeline running on three AIE cores is demonstrated in this example. The source files can be found in ``L3/tests/aie/Filter2D_multicore/16bit_aie_8bit_pl`` directory.

This example tests the performance of back-to-back Filter2D pipeline with three images being parallely processed on three AIE cores. Each AIE core is being fed by one instance of Tiler and Stitcher PL kernels. 

The tutorial provides a step-by-step guide that covers commands for building and running the pipeline.

Executable Usage
================

* **Work Directory(Step 1)**

The steps for library download and environment setup can be found in README of L3 folder. Please refer :ref:`Getting Started with Vitis Vision AIEngine Library Functions` for more details. For getting the design,

.. code-block:: bash

   cd L3/tests/aie/Filter2D_multicore/16bit_aie_8bit_pl

* **Build kernel(Step 2)**

Run the following make command to build your XCLBIN and host binary targeting a specific device. Please be noticed that this process will take a long time, maybe couple of hours.

.. code-block:: bash

   export DEVICE=< path-to-platform-directory >/< platform >.xpfm
   make all TARGET=hw

* **Run kernel(Step 3)**

To get the benchmark results, please run the following command.

.. code-block:: bash

   make run TARGET=hw

* **Running on HW**

After the build for hardware target completes, sd_card.img file will be generated in the build directory. 

1. Use a software like Etcher to flash the sd_card.img file on to a SD Card. 
2. After flashing is complete, insert the SD card in the SD card slot on board and power on the board.
3. Use Teraterm to connect to COM port and wait for the system to boot up.
4. After the boot up is done, goto /media/sd-mmcblk0p1 directory and run the executable file.
  
Performance
==========

The performance is shown below

.. table:: Table 1 Performance numbers in terms of FPS (Frames Per Second) for full HD images
    :align: center
	
    +----------------------+--------------+
    |       Dataset        |   FPS        |
    +======================+==============+
    |   Full HD(1920x1080) |   555        |
    +----------------------+--------------+

.. toctree::
    :maxdepth: 1