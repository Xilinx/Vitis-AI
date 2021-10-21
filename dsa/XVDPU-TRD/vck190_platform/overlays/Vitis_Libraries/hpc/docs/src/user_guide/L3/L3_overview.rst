.. 
   Copyright 2019 - 2021 Xilinx, Inc.
  
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
  
       http://www.apache.org/licenses/LICENSE-2.0
  
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

.. _user_guide_overview_l3:

***************************
Introduction of L3 APIs
***************************

2D-RTM L3 APIs are provided to allow users to experiment the
shot parallelism, that is simultaneously running the backward propagation path in the previous shot
and the forward propagation path in the current shot. *Figure 1* shows 
the details of the shot parallelism and how it's used to improve the system throughput.

.. figure:: /images/rtm2DShotPar.png
    :align: center
    :alt: 2D-RTM shot parallelism
    
    Figure 1. 2D-RTM shot parallelism implemented by L3 APIs 

The detailed implementation of shot parallelism can be found in the ``execute`` defined in
``L3/tests/rtm/main.cpp``, which also provies an usage example of 2D-RTM L3 APIs. The
``execute`` function depends on a set of other L3 APIs defined by class ``FPGA``. More detailed
information about ``FPGA`` class can be found in ``L3/include/sw/fpga_xrt.hpp``




