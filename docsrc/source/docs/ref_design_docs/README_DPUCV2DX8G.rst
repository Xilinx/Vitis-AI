:orphan:

VEK280 DPUCV2DX8G Reference Design
==================================


.. note:: Until the release of Versal |trade| AI Edge production speed files (currently targeted with release of Vivado 2023.2.1), PDI generation for Versal AI Edge will require an early enablement license that can be requested via the Versal AI Edge Errata Secure Site.  Also, the reference design archive does include a pre-compiled AIE object ``libadf.a`` for this specific DPU configuration, however, if the user wishes to reconfigure and recompile the DPUCV2DX8G, access to an AIE-ML Compiler early enablement license is required and can be obtained from the AIE Compiler Early Access Lounge.  Finally, the user will also want to have access to documentation such as the VEK280 schematics, user guide and BSPs which are provided in the VEK280 Early Access Lounge.  Please contact your local AMD sales or FAE contact to request access.


.. note:: This design is based on the B01 version of the VEK280 evaluation board. It can also be leveraged with the A01 version of the board with specific limitations that are outlined below.



The reference design associated with this document `is found
here <https://www.xilinx.com/bin/public/openDownload?filename=DPUCV2DX8G_VAI_v3.5.tar.gz>`__.

Table of Contents
-----------------

-  `1 Revision History <#1-revision-history>`__
-  `2 Overview <#2-overview>`__
-  `3 Software Tools and System
   Requirements <#3-software-tools-and-system-requirements>`__

   -  `3.1 Hardware <#31-hardware>`__
   -  `3.2 Software <#32-software>`__

-  `4 Design Files <#4-design-files>`__

   -  `4.1 Design Components <#41-design-components>`__

-  `5 Tutorials <#5-tutorials>`__

   -  `5.1 Board Setup <#51-board-setup>`__
   -  `5.2 Build and Run The Reference
      Design <#52-build-and-run-the-reference-design>`__

      -  `5.2.1 Build the DPU <#521-build-the-dpu>`__
      -  `5.2.2 Get Json File <#522-get-json-file>`__
      -  `5.2.3 Run ResNet50 Example <#523-run-resnet50-example>`__

   -  `5.3 Change the Configuration <#53-change-the-configuration>`__

-  `6 Instructions for Changing the
   Platform <#6-instructions-for-changing-the-platform>`__

   -  `6.1 DPUCV2DX8G Ports <#61-dpucv2dx8g-ports>`__
   -  `6.2 Changing the Platform <#62-changing-the-platform>`__

-  `7 Instructions for Adding Other
   Kernels <#7-instructions-for-adding-other-kernels>`__

   -  `7.1 RTL Kernel <#71-rtl-kernel>`__

-  `8 Known Issues <#8-known-issues>`__

1 Revision History
------------------

Vitis AI 3.5 change log: - Update platform to B01 board with ES silicon,
and support Vitis 2023.1. - Support multi-batch setting

VitisAI 3.0 change log: - Initial early access version

--------------

2 Overview
----------

The Xilinx Versal Deep Learning Processing Unit (DPUCV2DX8G) is a
computation engine optimized for convolutional neural networks. It
includes a set of highly optimized instructions, and supports most
convolutional neural networks, such as VGG, ResNet, GoogLeNet, YOLO,
SSD, MobileNet, and others.

This tutorial contains information about:

-  How to setup the VEK280 evaluation board.

-  How to build and run the DPUCV2DX8G reference design with VEK280
   platform in Vitis environment.

-  .. rubric:: How to modify the platform.
      :name: how-to-modify-the-platform.

3 Software Tools and System Requirements
----------------------------------------

3.1 Hardware
~~~~~~~~~~~~

Required:

-  Revision B01 VEK280 evaluation board

-  USB type-C cable, connected to a PC for the terminal emulator

-  SD card

.. note::  If the target is an A01 board, a USB Ethernet Adapter is also required.

3.2 Software
~~~~~~~~~~~~

Required:

- Vitis 2023.1 
- Python (version 2.7.5 or 3.6.8)
- csh

.. note::  ``bash`` is used during the build but some ``csh`` scripts are used.


4 Design Files
--------------

4.1 Design Components
~~~~~~~~~~~~~~~~~~~~~

The top-level directory structure shows the the major design components.

::

   ├── app
   ├── README.md
   ├── vek280_platform                        # VEK280 platform folder
   │   ├── LICENSE            
   │   ├── Makefile 
   │   ├── hw 
   │   ├── sw 
   │   ├── platform
   │   ├── platform.mk
   │   └── README.md 
   ├── vitis_prj                              # Vitis project folder
   │   ├── Makefile
   │   ├── scripts
   │   ├── xv2dpu
   │   └── xv2dpu_config.mk
   └── xv2dpu_ip                              # DPUCV2DX8G IP folder
       ├── aie
       └── rtl

--------------

5 Tutorials
-----------

5.1 Board Setup
~~~~~~~~~~~~~~~

Board jumper and switch settings:
                                 

Configure the Versal Boot Mode switch SW1 to boot from SD Card:

-  SW1[1:4]- [ON,OFF,OFF,OFF].

5.2 Build and Run The Reference Design
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following tutorials assume that the `$TRD_HOME` environment variable
is set as shown below.

::

   % export TRD_HOME =<Vitis AI path>/reference_design/DPUCV2DX8G-TRD

**Step1:** Build VEK280 platform

First, build the VEK280 platform in the folder `$TRD_HOME/vek280_platform`, more details refer to the instructions in `$TRD_HOME/vek280_platform/README.md`.

::

   % source <Vitis_install_path>/Vitis/2023.1/settings64.sh

   % source <PetaLinux_install_path>/settings.sh

   % make all

**Step2:** Setup the environment for building the DPUCV2DX8G IP and
kernel

When platform is ready, set the Vitis environment variable as given below.

Open a linux terminal. Set the linux as Bash mode.

::

   % source <vitis install path>/Vitis/2023.1/settings64.sh

5.2.1 Build the DPU
^^^^^^^^^^^^^^^^^^^

The default architecture of DPUCV2DX8G is C20B1 (`CU_N=1`,
`BATCH_SingleCU=1`, 16 AIE-ML cores for Convolution, 4 AIE-ML cores for
Non-Convolution), PL clock frequency is 300 MHz. This version of the
reference design only supports `CU_N=1`, but can support `BATCH_SingleCU` to
1~14. You can modify the file `$TRD_HOME/vitis_prj/xv2dpu_config.mk` to
change these parameters.

Execute the following command to build the project:

::

   % cd $TRD_HOME/vitis_prj

   % make all

Upon completion, you will find the generated SD card image here

`$TRD_HOME/vitis_prj/package_out/sd_card.img.gz` and the implemented Vivado project here `$TRD_HOME/vitis_prj/hw/binary_container_1/link/vivado/vpl/prj/prj.xpr`

.. note::  You can execute `make help` to see more detailed information.

.. note::  The implementation strategy may be changed by editing the file `$TRD_HOME/vitis_prj/scripts/system.cfg`. The default strategy is ``prop=run.impl_1.strategy=Performance_ExploreWithRemap``.

.. note::  If you are not modifying the configuration of the DPUCV2DX8G the compiled AIE Engine archive `libadf.a` can be reused. If you wish to skip compilation, comment out the last line of `$TRD_HOME/vitis_prj/Makefile`, which will save time when re-building the hardware design.

::

   # -@rm -rf aie

5.2.2 Get Json File
^^^^^^^^^^^^^^^^^^^

The `arch.json` file is an important file required by Vitis AI. It works
together with the Vitis AI compiler to support model compilation with
various DPUCV2DX8G configurations. The `arch.json` file will be
generated by Vitis during the compilation of DPUCV2DX8G reference
design, it can be found in `$TRD_HOME/vitis_prj/package_out/sd_card`.

It can also be found in the following path:

::

   $TRD_HOME/vitis_prj/hw/binary_container_1/link/vivado/vpl/prj/prj.gen/sources_1/bd/*/ip/*_DPUCV2DX8G_*/arch.json

5.2.3 Run ResNet50 Example
^^^^^^^^^^^^^^^^^^^^^^^^^^

The reference design project has generated the matching model file in
`$TRD_HOME/app` path, pre-configured with default settings. If the
configuration of the DPUCV2DX8G is modified, the model needs to be
compiled with the new fingerprint file, `arch.json`.

In this section, we will execute this example.

Use the balenaEtcher tool to flash
`$TRD_HOME/vitis_prj/package_out/sd_card.img.gz` into SD card, insert the SD card with the image into the destination board and power up the board. After Linux boots, copy the folder `$TRD_HOME/app` in this reference design to the target folder ``~/``, and run the following commands:

::

   % cd ~/app/model/

   % xdputil benchmark resnet50.xmodel 1

A typical output would appear as shown below:

::

   I1123 04:08:22.475286  1127 test_dpu_runner_mt.cpp:474] shuffle results for batch...
   I1123 04:08:22.476413  1127 performance_test.hpp:73] 0% ...
   I1123 04:08:28.476716  1127 performance_test.hpp:76] 10% ...
   .
   .
   .
   I1123 04:09:22.478189  1127 performance_test.hpp:76] 100% ...
   I1123 04:09:22.478253  1127 performance_test.hpp:79] stop and waiting for all threads terminated....
   I1123 04:09:22.478495  1127 performance_test.hpp:85] thread-0 processes 20225 frames
   I1123 04:09:22.478528  1127 performance_test.hpp:93] it takes 2299 us for shutdown
   I1123 04:09:22.478543  1127 performance_test.hpp:94] FPS= 337.061 number_of_frames=20225 time= 60.0039 seconds.
   I1123 04:09:22.478579  1127 performance_test.hpp:96] BYEBYE 

.. note::  For running other networks, refer to the `Vitis AI Github <https://github.com/Xilinx/Vitis-AI>`__ and `Vitis AI User Guide <https://docs.xilinx.com/r/en-US/ug1414-vitis-ai>`__.

5.3 Change the Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The DPUCV2DX8G IP provides some user-configurable parameters, refer to
the document `PG425 <https://docs.xilinx.com/r/en-US/pg425-dpu>`__. 

In this reference design, user-configurable parameters are in the file `$TRD_HOME/vitis_prj/xv2dpu_config.mk`. 

They are: 

- ``CU_N`` – Compute Unit (CU) number (only a value of 1 is supported in the current IP). 
- ``CPB_N`` – number of AI Engine cores for Convolution per batch handler (only a value of 16 is supported for this design). 
- ``BATCH_SingleCU`` – number of batch engine integrated in DPUCV2DX8G IP for CU_N=1. Values supported are 1 through 14.

After changing `$TRD_HOME/vitis_prj/xv2dpu_config.mk`, execute ``make all`` to build the design.

--------------

6 Instructions for Changing the Platform
----------------------------------------

6.1 DPUCV2DX8G Ports
~~~~~~~~~~~~~~~~~~~~

The DPUCV2DX8G ports are listed as below.

+------------------------------------------------+---------------------+
| Ports                                          | Descriptions        |
+================================================+=====================+
| m*_<wgt|img|instr>_axi                         | Master AXI          |
|                                                | interfaces,         |
|                                                | connected with NOC  |
|                                                | to access DDR       |
|                                                | (cips_noc in this   |
|                                                | reference design    |
|                                                | platform)           |
+------------------------------------------------+---------------------+
| m*_<data|ctrl>_axis                            | Master AXI-stream   |
|                                                | interface,          |
|                                                | connected with AI   |
|                                                | Engine              |
|                                                | (ai_engine_0).      |
+------------------------------------------------+---------------------+
| s*_<data|done>_axis                            | Slave AXI-stream    |
|                                                | interface,          |
|                                                | connected with AI   |
|                                                | Engine              |
|                                                | (ai_engine_0).      |
+------------------------------------------------+---------------------+
| m_axi_clk                                      | Input clock used    |
|                                                | for DPUCV2DX8G      |
|                                                | general logic, AXI  |
|                                                | and AXI-stream      |
|                                                | interface. Default  |
|                                                | frequency is 300M   |
|                                                | Hz in this          |
|                                                | reference design.   |
+------------------------------------------------+---------------------+
| m_axi_aresetn                                  | Active-Low reset    |
|                                                | for DPUCV2DX8G      |
|                                                | general logic.      |
+------------------------------------------------+---------------------+
| s_axi_control                                  | AXI lite interface  |
|                                                | for controlling     |
|                                                | DPUCV2DX8G          |
|                                                | registers,          |
|                                                | connected with CIPS |
|                                                | through             |
|                                                | A                   |
|                                                | XI_Smartconnect_IP. |
+------------------------------------------------+---------------------+
| s_axi_aclk                                     | Input clock for     |
|                                                | S_AXI_CONTROL.      |
|                                                | Default frequency   |
|                                                | is 150M Hz in this  |
|                                                | reference design.   |
+------------------------------------------------+---------------------+
| s_axi_aresetn                                  | Active-Low reset    |
|                                                | for S_AXI_CONTROL.  |
+------------------------------------------------+---------------------+
| interrupt                                      | Interrupt signal    |
|                                                | generated by        |
|                                                | DPUCV2DX8G.         |
+------------------------------------------------+---------------------+

DPUCV2DX8G’s connection with AI Engine array and NOC are all defined in
the `$TRD_HOME/vitis_prj/scripts/xv2dpu_aie_noc.cfg` (generated by `xv2dpu_aie_noc.py`).

For the clock design, make sure that: 

- s_axi_aclk for `s_axi_control` should use clock with lower frequency (e.g. 150MHz) to get better timing. 
- `AI Engine Core Frequency` should be 4 times of DPUCV2DX8G’s `m_axi_clk`, or the maximum AI Engine frequency. In this reference, it is 1250MHz (the maximum AI Engine frequency of XCVE2802-2MP device on the VEK280 board). The value of `AI Engine Core Frequency` can be set in the platform design files or `vitis_prj/scripts/postlink.tcl`.

6.2 Changing the Platform
~~~~~~~~~~~~~~~~~~~~~~~~~

Changing platform needs to modify 1 files: `vitis_prj/Makefile`.

.. note::   This target platform is based on ES device.

1) `vitis_prj/Makefile`:

-  Change the path of `xpfm` file for the varibale `PLATFORM`

::

     PLATFORM           = */*.xpfm

-  Change the path of `rootfs.exts` and `Image` in the package section
   (at the bottom of Makefile)

::

     --package.rootfs     */rootfs.ext4 \
     --package.sd_file    */Image \

--------------

7 Instructions for Adding Other Kernels
---------------------------------------

Vitis kernels developed for Versal devices, could be RTL kernel (only
use PL resouces), AIE kernel (only uses AI Engine tiles), or kernel
including both PL and AIE. The basic instructions for adding other
kernels in this reference design are shown below.

7.1 RTL Kernel
~~~~~~~~~~~~~~

Package the RTL kernel as XO file. Then modify 2 files:
`vitis_prj/Makefile`, and `vitis_prj/scripts/xv2dpu_aie_noc.py`,

1) `vitis_prj/Makefile`:

-  Add the name of XO files in the parameters `BINARY_CONTAINER_1_OBJS`
   by adding following command

::

   BINARY_CONTAINER_1_OBJS   += xxx.xo

-  In the v++ linking command line, specify the clock frequency for the
   clock soure of RTL kernel.

::

   --clock.freqHz <freqHz>:<kernelName.clk_name>

2) `vitis_prj/scripts/xvdpu_aie_noc.py`:

-  Create instance for the RTL kernel, and map kernel ports to memory
   (NOC)

::

   result += "nk=<kernel name>:<number>:<cu_name>.<cu_name>...\n" 

.. note::  For support with adding AI Engine kernel or RTL + AI Engine kernels to this design, please reach out to us for support.

8 Known Issues
--------------

1) This reference design has updated to support rev-B ES vek280 board,
   if you want to use it on rev-A board, Ethernet will not work, however
   you can use a USB Ethernet Adapter to workaround this issue.

2) This version of the reference design supports only a subset of the
   Vitis AI Model Zoo models.

3) The app/model/resnet50.xmodel only support the default arch
   setting(BATCH_SingleCU=1). To enable alternative batch settings, it
   is necessary to compile the corresponding xmodel with a new arch.json
   file.

4) It is suggested to add the following line to your tcl scripts
   `$HOME/.Xilinx/Vivado/Vivado_init.tcl`. For details about
   `Vivado_init.tcl`, please refer to the link page
   `https://docs.xilinx.com/r/en-US/ug894-vivado-tcl-scripting/Initializing-Tcl-Scripts`.
   This setting can help to optimize the ddr r/w performance by
   preplacing the NoC netlist.

::

   set_param place.preplaceNOC true

5) If your OS is Ubuntu, during AIE compilation step, you may get the error
   like "[AIE ERROR] XAieSim_GetStackRange():522: Invalid Map file, 2: No 
   such file or directory", the reason should be that your Ubuntu does not 
   install the "rename" function, you can install it manually.

.. raw:: html

   <!--
                                                                            
   * Copyright 2019 Xilinx Inc.                                               
   *                                                                          
   * Licensed under the Apache License, Version 2.0 (the "License");          
   * you may not use this file except in compliance with the License.         
   * You may obtain a copy of the License at                                  
   *                                                                          
   *    http://www.apache.org/licenses/LICENSE-2.0                            
   *                                                                          
   * Unless required by applicable law or agreed to in writing, software      
   * distributed under the License is distributed on an "AS IS" BASIS,        
   * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
   * See the License for the specific language governing permissions and      
   * limitations under the License.                                           

   -->


.. |trade|  unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim:
.. |reg|    unicode:: U+000AE .. REGISTERED TRADEMARK SIGN
   :ltrim:
