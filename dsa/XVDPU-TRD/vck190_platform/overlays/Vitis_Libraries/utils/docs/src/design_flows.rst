.. 
   Copyright 2019-2021 Xilinx, Inc.
  
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
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

.. toctree::
   :hidden:

.. _design_flows:

Design Flows
------------

The common tool and library pre-requisites that apply across all design flows are documented
in the requirements section above.

Recommended design flows are described as follows:

Shell Environment
~~~~~~~~~~~~~~~~~

Setup the build environment using the Vitis and XRT scripts.

.. code-block:: shell

    source /opt/xilinx/Vitis/2021.1/settings64.sh
    source /opt/xilinx/xrt/setup.sh
    export PLATFORM_REPO_PATHS=/opt/xilinx/platforms


For ``csh`` users, please look for corresponding scripts with ``.csh`` suffix and adjust the variable setting command
accordingly.

Setting `PLATFORM_REPO_PATHS` to the installation folder of platform files can enable makefiles
in this library to use `DEVICE` variable as a pattern.
Otherwise, full path to .xpfm file needs to be provided via `DEVICE` variable.

HLS Cases Command Line Flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

    cd L1/tests/hls_case_folder
    
    make run CSIM=1 CSYNTH=0 COSIM=0 VIVADO_SYN=0 VIVADO_IMPL=0 \
        DEVICE=/path/to/xilinx_u280_xdma_201920_3.xpfm

Test control variables are:

* ``CSIM`` for high level simulation.
* ``CSYNTH`` for high level synthesis to RTL.
* ``COSIM`` for co-simulation between software test bench and generated RTL.
* ``VIVADO_SYN`` for synthesis by Vivado.
* ``VIVADO_IMPL`` for implementation by Vivado.

For all these variables, setting to ``1`` indicates execution while ``0`` for skipping.
The default value of all these control variables are ``0``, so they can be omitted from command line
if the corresponding step is not wanted.

