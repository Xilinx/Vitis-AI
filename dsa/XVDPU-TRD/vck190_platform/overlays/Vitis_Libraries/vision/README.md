# Vitis Vision Library
The Vitis Vision library is a set of 90+ kernels, optimized for Xilinx FPGAs, AI Engine and SoCs, based on the OpenCV computer vision library. The kernels in the Vitis Vision library are optimized and supported in the Xilinx Vitis Tool Suite.

# DESIGN FILE HIERARCHY
The library is organized into the following folders -

| Folder Name | Contents |
| :------------- | :------------- |
| L1 | Examples that evaluate the Vitis Vision kernels, and demonstrate the kernels' use model in HLS flow |
| L2 | Examples that evaluate the Vitis Vision kernels, and demonstrate the kernels' use model in Vitis flow.  |
| L3 | Applications formed by stitching a pipeline of Vitis Vision functions |
| ext | Utility functions used in the opencl host code |
| data | Input images required to run examples and tests |

The organization of contents in each folder is described in the readmes of the respective folders.

## HARDWARE and SOFTWARE REQUIREMENTS
The Vitis Vision library is designed to work with Zynq, Zynq Ultrascale+, VCK190, and Alveo FPGAs. The library has been verified on zcu102, zcu104, vck190, U50, and U200 boards.

### Prerequisites

* Valid installation of [Vitis™ 2021.2](https://www.xilinx.com/html_docs/xilinx2021_1/vitis_doc/acceleration_installation.html#dhg1543555360045__ae364401) or later version and the corresponding licenses.
* Xilinx® Runtime ([XRT](https://www.xilinx.com/html_docs/xilinx2021_1/vitis_doc/acceleration_installation.html#dhg1543555360045__ae364401)) must be installed. XRT provides software interface to Xilinx FPGAs.
* Install/compile [OpenCV-4.4.0]((https://github.com/opencv/opencv/tree/4.4.0)) libraries(with compatible libjpeg.so). Appropriate version (X86/aarch32/aarch64) of compiler must be used based on the available processor for the target board.
* libOpenCL.so must be installed if not present.
* [Install the card](https://www.xilinx.com/support/documentation/boards_and_kits/accelerator-cards/1_9/ug1301-getting-started-guide-alveo-accelerator-cards.pdf) for which the platform is supported in Vitis 2021.2 or later versions.
* If targeting an [embedded platform](https://www.xilinx.com/html_docs/xilinx2021_1/vitis_doc/acceleration_installation.html#dhg1543555360045__ae364401) , set up the [evaluation board](https://xilinx-wiki.atlassian.net/wiki/spaces/A/pages/444006775/Zynq+UltraScale+MPSoC).

##### OpenCV Installation Guidance:

It is recommended to do a fresh installation of OpenCV 4.4.0 and not use existing libs of your system, as they may or may not work with Vitis environment.

**Please make sure you update and upgrade the packages and OS libraries of your system and
have cmake version>3.5 installed before proceeding.**

The below steps can help install the basic libs required to compile and link the OpenCV calls in Vitis Vision host code.

1. create a directory "source" and clone [opencv-4.4.0](https://github.com/opencv/opencv/tree/4.4.0) into it.
2. create a directory "source_contrib" and clone [opencv-4.4.0-contrib](https://github.com/opencv/opencv_contrib/tree/4.4.0) into it.
3. create 2 more directories: *build* , *install*
4. open a bash terminal and *cd* to *build* directory
5. Run the command: *export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/*
6. Run the command: *cmake -D CMAKE_BUILD_TYPE=RELEASE
  -D CMAKE_INSTALL_PREFIX=< path-to-install-directory>
  -D CMAKE_CXX_COMPILER=< path-to-Vitis-installation-directory>/tps/lnx64/gcc-6.2.0/bin/g++
  -D OPENCV_EXTRA_MODULES_PATH=< path-to-source_contrib-directory>/modules/
  -D WITH_V4L=ON -DBUILD_TESTS=OFF -DBUILD_ZLIB=ON
  -DBUILD_JPEG=ON -DWITH_JPEG=ON -DWITH_PNG=ON
  -DBUILD_EXAMPLES=OFF -DINSTALL_C_EXAMPLES=OFF
  -DINSTALL_PYTHON_EXAMPLES=OFF -DWITH_OPENEXR=OFF
  -DBUILD_OPENEXR=OFF <path-to-source-directory>*
7. Run the command: *make all -j8*
8. Run the command: *make install*

The OpenCV includes and libs will be in the *install* directory

##### Vitis HLS flow

L1 functions are targeted for Vitis HLS flow, where C-Simulation, Synthesis, Co-Simulation and RTL implementation can be performed. Vitis and OpenCV-4.4.0 x86 version libs need to be installed before hand. Rest of the prerequisites are optional to use this flow.

Please refer to L1 readme on how to setup the environment and run the functions

##### Vitis Flow

L2/L3 functions are targeted for Vitis flow, where software-emulation, hardware-emulation, and hardware build (to generate FPGA binaries) can be performed.

All the prerequisites need to be installed before starting with Vitis flow. For embedded devices, platforms and common images have to downloaded separately from the Xilinx official [download center](https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/embedded-platforms.html).

All the limitations and constraints of Vitis (OS support, compatibility etc.) are also applicable to Vitis Vison library.

Please refer to L2/L3 readme on how to setup the environment and run the functions

## OTHER INFORMATION
Full User Guide for Vitis Vision and using OpenCV on Xilinx devices Check here:
[Xilinx Vitis Vision User Guide](https://xilinx.github.io/Vitis_Libraries/vision/2021.2/index.html)

## SUPPORT
For questions and to get help on this project or your own projects, visit the [Xilinx Forums](https://forums.xilinx.com/t5/Vitis-Acceleration-SDAccel-SDSoC/bd-p/tools_v)

## LICENSE AND CONTRIBUTING TO THE REPOSITORY
The source for this project is licensed under the [Apache License](http://www.apache.org/licenses/LICENSE-2.0)

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

To contribute to this project, follow the guidelines in the [Repository Contribution README][] (link yet to be updated)

## ACKNOWLEDGEMENTS
This library is written by developers at
- [Xilinx](http://www.xilinx.com)

## Changelog:

**Versal AI Engine additions** :

    • Back to back filter2D with batch size three support  
		application showcasing increasing throughput of single filter2D kernel,
        by doing 3, back-2-back filter2D achieving 555 FPS with PL datamovers.

**New Programmable Logic (PL) functions and features**

    • ISP pipeline and functions:
		• End to End Mono Image Processing (ISP) pipeline with CLAHE TMO
			useful for ISP pipelines with monochrome sensors
		• RGB-IR along-with RGB-IR Image Processing (ISP) pipeline
			useful for ISP pipelines with IR sensors
		• Global Tone Mapping (GTM) along with an ISP pipeline using GTM
			adding to growing TMO (tone-mapping-operators) in the library for different quality
            and area tradeoff purposes: CLAHE, Local Tone Mapping, Quantization and Dithering


**Known issues**

  * Vitis GUI projects on RHEL83 and CEntOS82 may fail because of a lib conflict in the
     LD_LIBRARY_PATH setting. User needs to remove ${env_var:LD_LIBRARY_PATH} from the project
      environment settings for the function to build successfully.
  * SVM L2 function fails emulation with 2021.2 Vitis.
  * rgbir2bayer, isppipeline_rgbir functions are not supported on 2021.2 Vitis. Please use 2021.1
    Vitis for these 2 functions.
