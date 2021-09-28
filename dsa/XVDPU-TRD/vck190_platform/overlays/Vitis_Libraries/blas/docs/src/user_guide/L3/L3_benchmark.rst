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

.. meta::
   :keywords: BLAS, Library, Vitis BLAS Library, L3, level 3
   :description: Vitis BLAS library level 3 appliction programming interface benchmark.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

.. _benchmark_l3:

=====================
L3 API benchmark
=====================

.. toctree::
   :maxdepth: 3
   
   L3_benchmark_gemm.rst


Benchmark Test Overview
============================

Here are benchmarks of the Vitis BLAS library using the Vitis environment and comparing with Intel Math Kernel Library. It supports hardware emulation as well as running hardware accelerators on the Alveo U250.

1.1 Prerequisites
--------------------

1.1.1 Vitis BLAS Library
^^^^^^^^^^^^^^^^^^^^^^^^^^

- Alveo U250 installed and configured as per https://www.xilinx.com/products/boards-and-kits/alveo/u250.html#gettingStarted (when running hardware)
- Xilinx runtime (XRT) installed
- Xilinx Vitis 2021.1 installed and configured

1.1.2 Interl MK Library
^^^^^^^^^^^^^^^^^^^^^^^^^

- Downloaded and installed MKL from https://software.intel.com/en-us/mkl/choose-download/linux 

1.2  Benchmark with CPU
-------------------------

1.2.1 Step 1 :
^^^^^^^^^^^^^^^^^

Setup and build envrionment using MKL scripts:

.. code-block:: bash 

    source <INTEL_MKL_INSTALL_DIR>/bin/mklvars.sh intel64


1.2.2 Step 2 :
^^^^^^^^^^^^^^^^^

Run Makefile command. For example:

.. code-block:: bash 

    cd gemm_mkl
    ./run_gemm_mkl.sh 16 float b


1.3 Building and run the kernel with Vitis
--------------------------------------------

Taken gemm/memKernel as an example to indicate how to build the application and kernel with the command line Makefile flow.

1.3.1 Download code
^^^^^^^^^^^^^^^^^^^^^

These blas benchmarks can be downloaded from [vitis libraries](https://github.com/Xilinx/Vitis_Libraries.git) ``master`` branch.

.. code-block:: bash 

   git clone https://github.com/Xilinx/Vitis_Libraries.git
   cd Vitis_Libraries
   git checkout master
   cd blas


1.3.2 Setup environment
^^^^^^^^^^^^^^^^^^^^^^^^^

Setup and build envrionment using the Vitis and XRT scripts:

.. code-block:: bash 

    source <install path>/Vitis/2021.1/settings64.sh
    source /opt/xilinx/xrt/setup.sh

1.3.3 Build and run the kernel
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Run Makefile command. For example:

.. code-block:: bash 

    make run TARGET=hw PLATFORM_REPO_PATHS=/opt/xilinx/platforms DEVICE=xilinx_u250_xdma_201830_2


The Makefile supports various build target including hw emulation and hardware (hw_emu, hw)

The host application could be run manually using the following pattern:

.. code-block:: bash 

    <host application> <xclbin> <argv>


For example:

.. code-block:: bash 

    build_dir.hw.xilinx_u250_xdma_201830_2/host.exe build_dir.hw.xilinx_u250_xdma_201830_2/blas.xclbin build_dir.hw.xilinx_u250_xdma_201830_2/config_info.dat

    

