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


.. Project documentation master file, created by
   sphinx-quickstart on Thu Jun 20 14:04:09 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

==========
Benchmark 
==========
    
.. _pictures:

Pictures
-----------

The data is used by benchmarks, our commonly used pictures are listed in table 1. 

.. table:: Table 1 Pictures for benchmark
    :align: center

    +--------------------+----------+-------------+
    |   Pictures         |  Format  |    Size     |
    +====================+==========+=============+
    |  android.jpg       |    420   |  960*1280   |
    +--------------------+----------+-------------+
    |  offset.jpg        |    422   |  5184*3456  |
    +--------------------+----------+-------------+
    |  hq.jpg            |    444   |  5760*3840  |
    +--------------------+----------+-------------+
    |  iphone.jpg        |    420   |  3264*2448  |
    +--------------------+----------+-------------+
    |  lena_c_512.png    |    444   |   512*512   |
    +--------------------+----------+-------------+

Performance
-----------

For representing the resource utilization in each benchmark, we separate the overall utilization into 2 parts, where P stands for the resource usage in
platform, that is those instantiated in static region of the FPGA card, as well as K represents those used in kernels (dynamic region). The input is
png, jpg, pik, e.g. format, and the target device is set to Alveo U200.

.. table:: Table 2 Performance for processing pictures on FPGA
    :align: center

    +---------------------+------------------+--------------+----------+----------------+-------------+------------+------------+
    |    Architecture     |     Picture      |  Latency(ms) |  Timing  |    LUT(P/K)    |  BRAM(P/K)  |  URAM(P/K) |  DSP(P/K)  |
    +=====================+==================+==============+==========+================+=============+============+============+
    | JPEG Huffman Decoder|   android.jpg    |    0.889     |  270MHz  |  108.1K/7.9K   |   178/5     |    0/0     |     4/12   |
    +---------------------+------------------+--------------+----------+----------------+-------------+------------+------------+
    |  JPEG Decoder       |   android.jpg    |    1.515     |  243MHz  |  108.1K/23.1K  |   178/28    |    0/0     |     4/39   |
    +---------------------+------------------+--------------+----------+----------------+-------------+------------+------------+
    |  PIK                |  lena_c_512.png  |     16.0     |  300MHz  |  150.9K/439.4K |    338/62   |    0/16    |     7/0    |
    +---------------------+------------------+--------------+----------+----------------+-------------+------------+------------+

These are details for benchmark result and usage steps.

.. toctree::
   :maxdepth: 1

   benchmark/jpegHuffmanDecoderIP.rst
   benchmark/jpegDecoder.rst
   benchmark/pikEnc.rst
   

Test Overview
--------------

Here are benchmarks of the Vitis Codec Library using the Vitis environment and comparing with cpu(). 


.. _l2_vitis_codec:

Vitis Codec Library
~~~~~~~~~~~~~~~~~~~

* **Download code**

These graph benchmarks can be downloaded from `vitis libraries <https://github.com/Xilinx/Vitis_Libraries.git>`_ ``master`` branch.

.. code-block:: bash

   git clone https://github.com/Xilinx/Vitis_Libraries.git 
   cd Vitis_Libraries
   git checkout master
   cd codec 

* **Setup environment**

Specifying the corresponding Vitis, XRT, and path to the platform repository by running following commands.

.. code-block:: bash

   source <intstall_path>/installs/lin64/Vitis/2021.1/settings64.sh
   source /opt/xilinx/xrt/setup.sh
   export PLATFORM_REPO_PATHS=/opt/xilinx/platforms
