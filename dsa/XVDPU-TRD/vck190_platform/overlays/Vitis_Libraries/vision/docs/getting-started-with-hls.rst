
.. meta::
   :keywords: Vision, Library, Vitis Vision Library, HLS, Getting Started, C-simulation, C-synthesis, co-simulation, cv, Tcl
   :description: Describes the methodology to create a kernel, corresponding host code and a suitable makefile to compile an Vitis Vision kernel for any of the supported platforms in Vitis.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

Getting Started with HLS
#########################

The Vitis vision library can be used to build applications in Vivado® HLS as well as Vitis HLS.
This section of the document provides steps on how to run a single library component through
the Vivado HLS or Vitis HLS 2020.2 flow which includes, C-simulation,
C-synthesis, C/RTL co-simulation, and exporting the RTL as an IP.

All the functions under L1 folder of the Vitis Vision library can be built through Vitis HLS flow
in the following two modes:

#. Tcl Script Mode
#. GUI Mode


Tcl Script Mode
----------------

Each configuration of all functions in L1 are provided with TCL script which can be run through the
available Makefile.

Open a terminal and run the following commands to set the environment and build :

.. code:: c

   source < path-to-Vitis-installation-directory >/settings64.sh

   source < part-to-XRT-installation-directory >/setup.sh

   export DEVICE=< path-to-platform-directory >/< platform >.xpfm

   export OPENCV_INCLUDE=< path-to-opencv-include-folder >

   export OPENCV_LIB=< path-to-opencv-lib-folder >

   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:< path-to-opencv-lib-folder >

   make run CSIM=1 CSYNTH=1 COSIM=1 VIVADO_IMPL=1

GUI Mode
--------

Use the following steps to operate the HLS Standalone Mode using GUI:

#. Open a terminal and update the LD_LIBRARY_PATH to point to OpenCV lib folder.
#. From the same terminal, open Vivado® HLS or Vitis HLS in GUI mode and create a new project
#. Specify the name of the project. For example - Dilation.
#. Click Browse to enter a workspace folder used to store your projects.
#. Click Next.
#. Under the source files section, add the accel.cpp file which can be
   found in the examples folder. Also, fill the top function name (here
   it is dilation_accel).
#. Click Next.
#. Under the test bench section add tb.cpp.
#. Click Next.
#. Select the clock period to the required value (10ns in example).
#. Select the suitable part. For example, ``xczu9eg-ffvb1156-2-i``.
#. Click Finish.
#. Right click on the created project and select Project Settings.
#. In the opened tab, select Simulation.
#. Files added under the Test Bench section will be displayed. Select a
   file and click Edit CFLAGS.
#. Enter
   ``-I<path-to-L1-include-directory> -std=c++0x
   -I<path-to-opencv-include-folder>``.
   
   Note: When using Vivado HLS in the Windows operating system, make
   sure to provide the ``-std=c++0x`` flag only for C-Sim and Co-Sim. Do
   not include the flag when performing synthesis.
#. In the Linker Flags section, enter the opencv libs and path to the opencv libs
   ``-L<path-to-opencv-lib-folder> -lopencv_core -lopencv_imgcodecs -lopencv_imgproc``
#. Select Synthesis and repeat the above step for all the displayed
   files. Do not add opencv include path here.
#. Click OK.
#. Run the C Simulation, select Clean Build and specify the required
   input arguments.
#. Click OK.
#. All the generated output files/images will be present in the
   solution1->csim->build.
#. Run C synthesis.
#. Run co-simulation by specifying the proper input arguments.
#. The status of co-simulation can be observed on the console.


Constraints for Co-simulation
------------------------------

There are few limitations in performing co-simulation of the Vitis vision
functions. They are:

#. Functions with multiple accelerators are not supported.
#. Compiler and simulator are default in HLS (gcc, xsim).
#. Since HLS does not support multi-kernel integration, the current flow
   also does not support multi-kernel integration. Hence, the Pyramidal
   Optical flow and Canny Edge Detection functions and examples are not
   supported in this flow.
#. The maximum image size (HEIGHT and WIDTH) set in config.h file should
   be equal to the actual input image size.


AXI Video Interface Functions
=============================

Vitis vision has functions that will transform the xf::cv::Mat into Xilinx®
Video Streaming interface and vice-versa. ``xf::cv::AXIvideo2xfMat()`` and
``xf::cv::xfMat2AXIVideo()`` act as video interfaces to the IPs of the
Vitis vision functions in the Vivado® IP integrator.
``cvMat2AXIvideoxf<NPC>`` and ``AXIvideo2cvMatxf<NPC>``
are used on the host side.

An example function, 'axiconv', depicting the usage of these functions
is provided in the L1/examples directory.

.. table:: Table. AXI Video Interface Functions

   +----------------------------+-----------------------------------------+
   | Video Library Function     | Description                             |
   +============================+=========================================+
   | AXIvideo2xfMat             | Converts data from an AXI4 video stream |
   |                            | representation to xf::cv::Mat format.   |
   +----------------------------+-----------------------------------------+
   | xfMat2AXIvideo             | Converts data stored as xf::cv::Mat     |
   |                            | format to an AXI4 video stream.         |
   +----------------------------+-----------------------------------------+
   | cvMat2AXIvideoxf           | Converts data stored as cv::Mat format  |
   |                            | to an AXI4 video stream                 |
   +----------------------------+-----------------------------------------+
   | AXIvideo2cvMatxf           | Converts data from an AXI4 video stream |
   |                            | representation to cv::Mat format.       |
   +----------------------------+-----------------------------------------+


AXIvideo2xfMat
--------------

The ``AXIvideo2xfMat`` function receives a sequence of images using the
AXI4 Streaming Video and produces an ``xf::cv::Mat`` representation.

.. rubric:: API Syntax


.. code:: c

   template<int W,int T,int ROWS, int COLS,int NPC>
   int AXIvideo2xfMat(hls::stream< ap_axiu<W,1,1,1> >& AXI_video_strm, xf::cv::Mat<T,ROWS, COLS, NPC>& img)

.. rubric:: Parameter Descriptions


The following table describes the template and the function parameters.

.. table:: Table. AXIvideo2cvMatxf Function Parameter Description

   +-----------------------------------+-----------------------------------+
   | Parameter                         | Description                       |
   +===================================+===================================+
   | W                                 | Data width of AXI4-Stream.        |
   |                                   | Recommended value is pixel depth. |
   +-----------------------------------+-----------------------------------+
   | T                                 | Pixel type of the image. 1        |
   |                                   | channel (XF_8UC1). Data width of  |
   |                                   | pixel must be no greater than W.  |
   +-----------------------------------+-----------------------------------+
   | ROWS                              | Maximum height of input image.    |
   +-----------------------------------+-----------------------------------+
   | COLS                              | Maximum width of input image.     |
   +-----------------------------------+-----------------------------------+
   | NPC                               | Number of pixels to be processed  |
   |                                   | per cycle. Possible options are   |
   |                                   | XF_NPPC1 and XF_NPPC8 for 1-pixel |
   |                                   | and 8-pixel operations            |
   |                                   | respectively.                     |
   +-----------------------------------+-----------------------------------+
   | AXI_video_strm                    | HLS stream of ap_axiu (axi        |
   |                                   | protocol) type.                   |
   +-----------------------------------+-----------------------------------+
   | img                               | Input image.                      |
   +-----------------------------------+-----------------------------------+

This function will return bit error of ERROR_IO_EOL_EARLY( 1 ) or
ERROR_IO_EOL_LATE( 2 ) to indicate an unexpected line length, by
detecting TLAST input.

For more information about AXI interface see UG761.


xfMat2AXIvideo
--------------

The ``Mat2AXI`` video function receives an xf::cv::Mat representation of a
sequence of images and encodes it correctly using the AXI4 Streaming
video protocol.

.. rubric:: API Syntax


.. code:: c

   template<int W, int T, int ROWS, int COLS,int NPC>
   int xfMat2AXIvideo(xf::cv::Mat<T,ROWS, COLS,NPC>& img,hls::stream<ap_axiu<W,1,1,1> >& AXI_video_strm)

.. rubric:: Parameter Descriptions


The following table describes the template and the function parameters.

.. table:: Table. xfMat2AXIvideo Function Parameter Description

   +-----------------------------------+-----------------------------------+
   | Parameter                         | Description                       |
   +===================================+===================================+
   | W                                 | Data width of AXI4-Stream.        |
   |                                   | Recommended value is pixel depth. |
   +-----------------------------------+-----------------------------------+
   | T                                 | Pixel type of the image. 1        |
   |                                   | channel (XF_8UC1). Data width of  |
   |                                   | pixel must be no greater than W.  |
   +-----------------------------------+-----------------------------------+
   | ROWS                              | Maximum height of input image.    |
   +-----------------------------------+-----------------------------------+
   | COLS                              | Maximum width of input image.     |
   +-----------------------------------+-----------------------------------+
   | NPC                               | Number of pixels to be processed  |
   |                                   | per cycle. Possible options are   |
   |                                   | XF_NPPC1 and XF_NPPC8 for 1-pixel |
   |                                   | and 8-pixel operations            |
   |                                   | respectively.                     |
   +-----------------------------------+-----------------------------------+
   | AXI_video_strm                    | HLS stream of ap_axiu (axi        |
   |                                   | protocol) type.                   |
   +-----------------------------------+-----------------------------------+
   | img                               | Output image.                     |
   +-----------------------------------+-----------------------------------+

This function returns the value 0.

Note: The NPC values across all the functions in a data flow must follow
the same value. If there is mismatch it throws a compilation error in
HLS.


cvMat2AXIvideoxf
----------------

The ``cvMat2Axivideoxf`` function receives image as cv::Mat
representation and produces the AXI4 streaming video of image.

.. rubric:: API Syntax


.. code:: c

   template<int NPC,int W>
   void cvMat2AXIvideoxf(cv::Mat& cv_mat, hls::stream<ap_axiu<W,1,1,1> >& AXI_video_strm)


.. rubric:: Parameter Descriptions


The following table describes the template and the function parameters.

.. table:: Table. AXIvideo2cvMatxf Function Parameter Description

   +-----------------------------------+-----------------------------------+
   | Parameter                         | Description                       |
   +===================================+===================================+
   | W                                 | Data width of AXI4-Stream.        |
   |                                   | Recommended value is pixel depth. |
   +-----------------------------------+-----------------------------------+
   | NPC                               | Number of pixels to be processed  |
   |                                   | per cycle. Possible options are   |
   |                                   | XF_NPPC1 and XF_NPPC8 for 1-pixel |
   |                                   | and 8-pixel operations            |
   |                                   | respectively.                     |
   +-----------------------------------+-----------------------------------+
   | AXI_video_strm                    | HLS stream of ap_axiu (axi        |
   |                                   | protocol) type.                   |
   +-----------------------------------+-----------------------------------+
   | cv_mat                            | Input image.                      |
   +-----------------------------------+-----------------------------------+


AXIvideo2cvMatxf
----------------

The ``Axivideo2cvMatxf`` function receives image as AXI4 streaming video
and produces the cv::Mat representation of image

.. rubric:: API Syntax


.. code:: c

   template<int NPC,int W>
   void AXIvideo2cvMatxf(hls::stream<ap_axiu<W,1,1,1> >& AXI_video_strm, cv::Mat& cv_mat) 

.. rubric:: Parameter Descriptions

The following table describes the template and the function parameters.

.. table:: Table. AXIvideo2cvMatxf Function Parameter Description

   +-----------------------------------+-----------------------------------+
   | Parameter                         | Description                       |
   +===================================+===================================+
   | W                                 | Data width of AXI4-Stream.        |
   |                                   | Recommended value is pixel depth. |
   +-----------------------------------+-----------------------------------+
   | NPC                               | Number of pixels to be processed  |
   |                                   | per cycle. Possible options are   |
   |                                   | XF_NPPC1 and XF_NPPC8 for 1-pixel |
   |                                   | and 8-pixel operations            |
   |                                   | respectively.                     |
   +-----------------------------------+-----------------------------------+
   | AXI_video_strm                    | HLS stream of ap_axiu (axi        |
   |                                   | protocol) type.                   |
   +-----------------------------------+-----------------------------------+
   | cv_mat                            | Output image.                     |
   +-----------------------------------+-----------------------------------+
