# Vitis Vision Library
The Vitis Vision library is a set of 90+ kernels, optimized for Xilinx FPGAs and SoCs, based on the OpenCV computer vision library. The kernels in the Vitis Vision library are optimized and supported in the Xilinx Vitis Tool Suite.

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

Vitis 2021.1 Development Environment is required to work with the library.

**Vitis Flow:**

U200 platform, available in the Vitis tool, is required to build and run the library functions on U200 PCIe board. Same applies for U50. VCK190 and Zynq based platforms have to dowmloaded separately from the Xilinx official download centre.

## OTHER INFORMATION
Full User Guide for Vitis Vision and using OpenCV on Xilinx devices Check here:
[Xilinx Vitis Vision User Guide](https://xilinx.github.io/Vitis_Libraries/vision/2021.1/index.html)

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

**First release of selected vision functions for Versal AI Engines** 

**Functions available:** 

    •Filter2D

    •absdiff

    •accumulate

    •accumulate_weighted

    •addweighted

    •blobFromImage

    •colorconversion

    •convertscaleabs

    •erode

    •gaincontrol

    •gaussian

    •laplacian

    •pixelwise_mul

    •threshold

    •zero

**xfcvDataMovers** : Utility datamovers to facilitate easy tiling of high resolution images and transfer to local memory of AI Engines cores. Two flavors

    • Using PL kernel : higher throughput at the expense of additional PL resources. 
    • Using GMIO : lower throughput than PL kernel version but uses Versal NOC (Network on chip) and no PL resources. 
 

**New Programmable Logic (PL) functions and features**

The below functions and pipelines are newly added into the library.

*ISP pipeline and functions*


    • Updated 2020.2 Non-HDR Pipeline : 

		    • Support to change few of the ISP parameters at runtime : Gain parameters for red and blue channels, AWB enable/disable option, gamma tables for R,G,B, %pixels ignored to compute min&max for awb normalization.
		    • Gamma Correction and Color Space conversion (RGB2YUYV) made part of the pipeline.
		
    • New 2021.1 HDR Pipeline : 2020.2 Pipeline + HDR support

		     • HDR merge : merges the 2 exposures which supports sensors with digital overlap between short exposure frame and long exposure frame.
		     • Four Bayer patterns supported : RGGB,BGGR,GRBG,GBRG
		     • HDR merge + isp pipeline with runtime configurations, which returns RGB output.
		     • Extraction function : HDR extraction function is preprocessing function, which takes single digital overlapped stream as input and returns the 2 output exposure frames(SEF,LEF). 
		 
*3DLUT*

	3DLUT provides input-output mapping to control complex color operators, such as hue, saturation, and luminance.
 
*CLAHE*


	Contrast Limited Adaptive Histogram Equalization is a method which limits the contrast while performing adaptive histogram equalization so that it does not over amplify the contrast in the near constant regions. This it also reduces the problem of noise amplification.

*Flip*


	Flips the image along horizontal and vertical line.

*Custom CCA*


	Custom version of Connected Component Analysis Algorithm for defect detection in fruits. Apart from computing defected portion of fruits , it computes defected-pixels as well as total-fruit-pixels

*Other updates*


    • Canny updates : Canny function now supports any image resolution.
    • Gamma correction : Gamma function changed
    • AWB optimization : AWB modules optimized.


**Library Related Changes**

    • All tests have been upgraded from using OpenCV 3.4.2 to OpenCV 4.4.0   
    • Added support for vck190 and aws-vu9p-f1 platforms.
    • A new benchmarking section with benchmarking collateral for selected pipeline/functions published.

**Known issues**

    • Vitis GUI projects on RHEL83 and CEntOS82 may fail because of a lib conflict in the LD_LIBRARY_PATH setting.
    • Windows OS has path length limitations, kernel names must be smaller than 25 characters.
