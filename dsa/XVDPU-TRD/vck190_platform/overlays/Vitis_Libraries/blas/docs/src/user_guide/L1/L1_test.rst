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
   :keywords: BLAS, Library, Vitis BLAS Library, primitives, L1 test
   :description: Vitis BLAS library L1 primitive implementations have been tested against numpy functions. That is, a python based testing environment has been developed to generate random test inputs for each primitive and its corresponding function in numpy, compute the golden reference via the numpy function call, and finally compare the golden reference with the csim and cosim outputs of the primitive to verify the correctness of the implementation.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

.. _user_guide_test_l1:

*******************************
L1 Test
*******************************

All L1 primitive implementations have been tested against numpy functions. That is, a python based testing environment has been developed to generate random test inputs for each primitive and its corresponding function in numpy, compute the golden reference via the numpy function call, and finally compare the golden reference with the csim and cosim outputs of the primitive to verify the correctness of the implementation.
To run the testing process of L1 primitives, please follow the steps below.

1. Set up Python environment
=============================
Please follow the instructions described in :doc:`Python environment setup guide <pyenvguide>` to install anaconda3 and setup xf_blas environment. All testing should be run under xf_blas environment. Please deactivate xf_blas environment after testing.

2. Set up Vivado_hls environment
=================================
Please navigate to directory L1/tests, and change the setting of environment variable **TA_PATH** to point to the installation path of your Vitis 2020.2, and run following command to set up Vivado_hls environment.

.. code-block:: bash

   export XILINX_VITIS=${TA_PATH}/Vitis/2020.2
   export XILINX_VIVADO=${TA_PATH}/Vivado/2020.2
   source ${XILINX_VIVADO}/settings64.sh

3. Test L1 primitives
==============================
The L1 primitives can be tested individually or as a group. To launch the testing process, please navigate to the directory **L1/tests**, and enter the following command.

.. code-block:: bash

   $ python ./run_test.py --operator amax amin asum axpy copy dot nrm2 scal swap gemv gbmv sbmvLo sbmvUp tbmvLo tbmvUp trmvLo trmvUp symvLo symvUp spmvUp spmvLo tpmvLo tpmvUp

The above command will test and verify all L1 primitives' implementation in both csim and cosim modes. Hence, it can take a very long time. The following commands show examples for quickly testing some primitives in pure csim or cosim mode.

.. code-block:: bash

   $ python ./run_test.py --operator amax amin --csim
   $ python ./run_test.py --operator copy dot --cosim

By default, the testing process only runs in a single thread mode. To speed up the process, users can run the testing with multiple thread via **--parallel** option. For example,

.. code-block:: bash

   $ python ./run_test --operator gemv gbmv --parallel 4

4. Test configuration
==========================
For each primitive, a test configuration file **profile.json** has been provided to specify the test inputs range, the size of the input vector or matrix, the template parameter value used for instantiating the primitive and the simulation mode (csim or cosim) used for testing. Users can find the profile.json file under directory **L1/include/hw/primitive_name**. For example, the profile.json file under **L1/include/hw** contains the following code.

.. code-block:: bash

   {
    "b_csim": true,
    "b_synth": true,
    "b_cosim": true,
    "dataTypes": [
      "float64",
      "uint16",
      "int32"
    ],
    "retTypes": [
      "int32",
      "int32",
      "int32"
    ],
    "op": "amax",
    "logParEntries": 2,
    "vectorDims": [
      1024,
      4096,
      8192
    ],
    "valueRange": [
      -1024,
      1024
    ],
    "numSimulation": 2
  }

The configuration file will instruct our testing infrastructure to generate the tests.

5. Test outputs
==================
At the end of the testing process, users will find a file called **statistics.rpt** that summarizes the test results, **Passed** or **Failed** for each primitive under test. For each primitive, there is also a file called **report.rpt** in the primitive's folder under directory **out_test/**. This file summarizes the quality of the implementation, namely the resource usage and the efficiency of the implementation. Where efficiency is calculated by equation **theoretical_cycles / measured_cosim_cycles**. The higher the efficiency, the better the performance the implementation will provide.
