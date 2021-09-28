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


.. Project documentation master file, created by
   sphinx-quickstart on Thu Jun 20 14:04:09 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==========
Benchmark 
==========
    
.. _datasets:

Datasets
-----------

Dataset is selected based on maximum resolution of the image to be processed, number of pixels processed per clock cycle (NPPC) and few other function specific parameters.

Performance
------------

Resource utilization numbers along with the achieved Frames Per Second (FPS) are represented for standalone kernel for both Full HD(1920x1080) and 4K(3840x2160) image resolutions. CPU numbers are an average of 100 runs and are calculated on Intel(R) Xeon(R) CPU E5-2680 @ 2.70GHz.

.. note:: Some of the reference functions against which benchmark is done, although follow similar algorithm as Vitis Vision function, doesnot match functionally with the corresponding Vitis Vision function because of different output format. In such cases there are two reference functions, one, a standard function, used to calculate the performance which can be enabled or disabled through the macro "__XF_BENCHMARK" defined in the config file. The other is internal reference function used for verifying functional correctness against the Vitis Vision function.

.. table:: Table 1 Performance on FPGA
    :align: center

    +-------------------------------+------------------------------------------+--------------+----------+-----------------+------------+------------+------------+
    |        Algorithm              |            Dataset                       |    FPS(CPU)  | FPS(FPGA)|      LUT        |    BRAM    |     FF     |    DSP     |
    |                               +------------+------+----------------------+              |          |                 |            |            |            |
    |                               | Resolution | NPPC | other params         |              |          |                 |            |            |            |
    +===============================+============+======+======================+==============+==========+=================+============+============+============+
    |                               |     4K     |   8  | L1 Norm,             |     9        |    95    |     31408       |    132     |    19148   |     96     |
    |    Canny Edge tracing         |            |      | Filter - 3x3         |              |          |                 |            |            |            |
    |                               +------------+------+----------------------+--------------+----------+-----------------+------------+------------+------------+
    |                               |     FHD    |   8  | L1 Norm,             |     25       |    333   |     17451       |    65      |    11256   |     63     |
    |                               |            |      | Filter - 3x3         |              |          |                 |            |            |            |
    +-------------------------------+------------+------+----------------------+--------------+----------+-----------------+------------+------------+------------+  
    |  Harris Corner Detection      |     4K     |   8  | L1 Norm,             |     22       |    289   |     40460       |    227     |    27236   |     208    |
    |                               |            |      | Filter - 3x3         |              |          |                 |            |            |            |
    |                               +------------+------+----------------------+--------------+----------+-----------------+------------+------------+------------+
    |                               |     FHD    |   8  | L1 Norm,             |     62       |    1100  |     22478       |    113     |    16021   |     138    |
    |                               |            |      | Filter - 3x3         |              |          |                 |            |            |            |
    +-------------------------------+------------+------+----------------------+--------------+----------+-----------------+------------+------------+------------+
    |  Fast Corner Detection        |     4K     |   8  |      NA              |     79       |    289   |     21171       |     10     |    13396   |     0      |
    |                               +------------+------+----------------------+--------------+----------+-----------------+------------+------------+------------+
    |                               |     FHD    |   8  |      NA              |     186      |    1100  |     20437       |     10     |    14322   |     0      |
    +-------------------------------+------------+------+----------------------+--------------+----------+-----------------+------------+------------+------------+
    |  Kalman filter                |     NA     |   1  | SV - 16x16x16        | 6.57ms       | 0.55 ms  |     59342       |     98     |    90762   |     361    |
    |                               |            |      |                      | (latency)    | (latency)|                 |            |            |            | 
    +-------------------------------+------------+------+----------------------+--------------+----------+-----------------+------------+------------+------------+
    |  Lk Dense Pyramidal Optical   |     4K     |   1  | 5 iterations,        |     0.15     |    3     |     30781       |    182     |    26169   |     83     |
    |  Flow(2 frames)               |            |      | 5 levels             |              |          |                 |            |            |            |
    |                               +------------+------+----------------------+--------------+----------+-----------------+------------+------------+------------+ 
    |                               |     FHD    |   1  | 5 iterations,        |     0.63     |    12    |     30839       |    107     |    25714   |     83     |
    |                               |            |      | 5 levels             |              |          |                 |            |            |            |
    +-------------------------------+------------+------+----------------------+--------------+----------+-----------------+------------+------------+------------+
    |  Corner Tracker               |     4K     |   1  | MAXCORNERS-10K       |     0.53     |    3     |     37547       |    225     |    35089   |    115     |
    |  (2 frames)                   +------------+------+----------------------+--------------+----------+-----------------+------------+------------+------------+
    |                               |     FHD    |   1  | MAXCORNERS-10K       |      2       |    11    |     34624       |    129     |    34198   |    115     |
    +-------------------------------+------------+------+----------------------+--------------+----------+-----------------+------------+------------+------------+
    |  Colordetect                  |     4K     |   1  | Filter - 3x3         |     15       |    289   |     16523       |    176     |    9145    |      3     |
    |                               +------------+------+----------------------+--------------+----------+-----------------+------------+------------+------------+
    |                               |     FHD    |   1  | Filter - 3x3         |     28       |    1100  |      9961       |     91     |    5432    |      1     |
    +-------------------------------+------------+------+----------------------+--------------+----------+-----------------+------------+------------+------------+
    |  Gaussian difference          |     4K     |   1  | Filter - 3x3         |     126      |    289   |     27235       |     35     |    37426   |     281    |
    |                               +------------+------+----------------------+--------------+----------+-----------------+------------+------------+------------+
    |                               |     FHD    |   1  | Filter - 3x3         |     500      |    1100  |     15146       |     19     |    22032   |     193    |
    +-------------------------------+------------+------+----------------------+--------------+----------+-----------------+------------+------------+------------+
    |  Bilateral filter             |     4K     |   8  | Filter - 3x3         |     72       |    289   |     22523       |     35     |    22862   |     65     |
    |                               +------------+------+----------------------+--------------+----------+-----------------+------------+------------+------------+
    |                               |     FHD    |   8  | Filter - 3x3         |     200      |    1100  |     12513       |     18     |    13451   |     44     |
    +-------------------------------+------------+------+----------------------+--------------+----------+-----------------+------------+------------+------------+
    |  Stereo LBM                   |     4K     |   1  | PARALLEL Units - 32  |     13       |    34    |     19005       |     26     |    21113   |     7      |
    |                               |            |      | Disparity - 32       |              |          |                 |            |            |            |
    |                               +------------+------+----------------------+--------------+----------+-----------------+------------+------------+------------+
    |                               |     FHD    |   1  | PARALLEL Units - 32  |     35       |    135   |     19380       |     13     |    20670   |     7      |
    |                               |            |      | Disparity - 32       |              |          |                 |            |            |            |
    +-------------------------------+------------+------+----------------------+--------------+----------+-----------------+------------+------------+------------+
    |  ISP Pipeline                 |     4K     |   4  | 16-bit               |     0.11     |    135   |     18987       |     24     |    17713   |     91     |
    |                               +------------+------+----------------------+--------------+----------+-----------------+------------+------------+------------+
    |                               |     FHD    |   4  | 16-bit               |     0.44     |    520   |     19254       |     19     |    17534   |     88     |
    +-------------------------------+------------+------+----------------------+--------------+----------+-----------------+------------+------------+------------+
	
Below are the links to the individual benchmark result files.

.. toctree::
   :maxdepth: 1

   canny-bm.rst
   harris-bm.rst
   fast-bm.rst
   kalmanfilter-bm.rst
   denseLKpyrOF-bm.rst
   cornerTracker-bm.rst
   colordetect-bm.rst
   gaussiandifference-bm.rst
   bilateralfilter-bm.rst
   stereolbm-bm.rst
   ISPpipeline-bm.rst


.. _l2_vitis_vision:

Vitis Vision Library
~~~~~~~~~~~~~~~~~~~~

* **Download code**

These Vitis Vision benchmarks can be downloaded from `vitis libraries <https://github.com/Xilinx/Vitis_Libraries.git>`_ ``master`` branch.

.. code-block:: bash

   git clone https://github.com/Xilinx/Vitis_Libraries.git 
   cd Vitis_Libraries
   git checkout master
   cd vision

* **Setup environment**

Specify the corresponding Vitis, XRT, FPGA device, and path to the OpenCV libs by running following commands.

.. code-block:: bash

   source <intstall_path>/installs/lin64/Vitis/2021.1_released/settings64.sh
   source <path-to-xrt-installation>/setup.sh
   export DEVICE=< path-to-platform-directory >/< platform >.xpfm
   export OPENCV_INCLUDE=< path-to-opencv-include-folder >
   export OPENCV_LIB=< path-to-opencv-lib-folder >
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:< path-to-opencv-lib-folder > 
