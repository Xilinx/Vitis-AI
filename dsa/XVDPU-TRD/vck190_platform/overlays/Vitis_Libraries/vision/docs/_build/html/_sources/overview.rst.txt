.. meta::
   :keywords: Vision, Library, Vitis Vision Library, overview, features, kernel
   :description: Using the Vitis vision library.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

.. _overview:

Overview
########

The Vitis vision library has been designed to work in the Vitis development environment, and provides a software
interface for computer vision functions accelerated on an FPGA device.
Vitis vision library functions are mostly similar in functionality to their
OpenCV equivalent. Any deviations, if present, are documented.


.. seealso:: For more information on the Vitis vision library prerequisites, see Prerequisites_. 
To familiarize yourself with the steps required to use the Vitis vision library
functions, see the `Using the Vitis vision
Library <using-the-vitis-vision-library.html>`__.

.. _basic-features:

Basic Features
===============

All Vitis vision library functions follow a common format. The following
properties hold true for all the functions.

-  All the functions are designed as templates and all arguments that
   are images, must be provided as ``xf::cv::Mat``.
-  All functions are defined in the ``xf::cv`` namespace.
-  Some of the major template arguments are:

   -  Maximum size of the image to be processed
   -  Datatype defining the properties of each pixel
   -  Number of pixels to be processed per clock cycle
   -  Other compile-time arguments relevent to the functionality.

The Vitis vision library contains enumerated datatypes which enables you to
configure ``xf::cv::Mat``. For more details on ``xf::cv::Mat``, see the `xf::cv::Mat
Image Container Class <api-reference.html>`__.

.. _xfopencv-kernel:

Vitis Vision Kernel on Vitis
============================

The Vitis vision library is designed to be used with the Vitis development
environment. 

The OpenCL host code is written in the testbench file, whereas the calls to Vitis 
Vision functions are done from the accel file.
The image containers for Vitis vision library functions are ``xf::cv::Mat``
objects. For more information, see the `xf::cv::Mat Image Container
Class <api-reference.html>`__.

.. _xfopencv-lib-contents:

Vitis Vision Library Contents
==============================

The following table lists the contents of the Vitis vision library.

.. table::  Vitis Vision Library Contents

	+-----------------------------------+-----------------------------------+
	| Folder                            | Details                           |
	+===================================+===================================+
	| L1/examples                       | Contains the sample testbench code|
	|                                   | to facilitate running unit tests  |
	|                                   | on Vitis/Vivado HLS. The examples/|
	|                                   | has folders with algorithm names. |
	|                                   | Each algorithm folder contains    |
	|                                   | testbench, accel, config, Makefile|
	|                                   | , Json file and a 'build' folder. |
	+-----------------------------------+-----------------------------------+
	| L1/include/common                 | Contains the common library       |
	|                                   | infrastructure headers, such as   |
	|                                   | types specific to the library.    |
	+-----------------------------------+-----------------------------------+
	| L1/include/core                   | Contains the core library         |
	|                                   | functionality headers, such as    |
	|                                   | the ``math`` functions.           |
	+-----------------------------------+-----------------------------------+
	| L1/include/features               | Contains the feature extraction   |
	|                                   | kernel function definitions. For  |
	|                                   | example, ``Harris``.              |
	+-----------------------------------+-----------------------------------+
	| L1/include/imgproc                | Contains all the kernel function  |
	|                                   | definitions related to image proce|
	|                                   | ssing definitions.                |
	+-----------------------------------+-----------------------------------+
	| L1/include/video                  | Contains all the kernel function  |
	|                                   | definitions, related to video proc|
	|                                   | essing functions.eg:Optical flow  |
	+-----------------------------------+-----------------------------------+
	| L1/include/dnn                    | Contains all the kernel function  |
	|                                   | definitions, related to deep lea  |
	|                                   | rning preprocessing.              |
	+-----------------------------------+-----------------------------------+
	| L1/tests                          | Contains all test folders to run  |
	|                                   | simulations, synthesis and export |
	|                                   | RTL.The tests folder contains the |
	|                                   | folders with algorithm names.Each |
	|                                   | algorithm folder further contains |
	|                                   | configuration folders, that has   |
	|                                   | makefile and tcl files to run     |
	|                                   | tests.                            |
	+-----------------------------------+-----------------------------------+
	| L1/examples/build                 | Contains xf_config_params.h file, |
	|                                   | which has configurable macros and |
	|                                   | varibales related to the particula|
	|                                   | r example.                        |
	+-----------------------------------+-----------------------------------+
	| L2/examples                       | Contains the sample testbench code|
	|                                   | to facilitate running unit tests  |
	|                                   | on Vitis. The examples/ contains  |
	|                                   | the folders with algorithm names. |
	|                                   | Each algorithm folder contains    |
	|                                   | testbench, accel, config, Makefile|
	|                                   | , Json file and a 'build' folder. |
	+-----------------------------------+-----------------------------------+
	| L2/tests                          | Contains all test folders to run  |
	|                                   | software, hardware emulations     |
	|                                   | and hardware build. The tests cont|
	|                                   | ains folders with algorithm names.|
	|                                   | Each algorithm folder further cont|
	|                                   | ains configuration folders, that  |
	|                                   | has makefile and tcl files to run |
	|                                   | tests.                            |
	+-----------------------------------+-----------------------------------+
	| L2/examples/build                 | Contains xf_config_params.h file, |
	|                                   | which has configurable macros and |
	|                                   | varibales related to the particula|
	|                                   | r example.                        |
	+-----------------------------------+-----------------------------------+
	| L3/examples                       | Contains the sample testbench code|
	|                                   | to build pipeline functions       |
	|                                   | on Vitis. The examples/ contains  |
	|                                   | the folders with algorithm names. |
	|                                   | Each algorithm folder contains    |
	|                                   | testbench, accel, config, Makefile|
	|                                   | , Json file and a 'build' folder. |
	+-----------------------------------+-----------------------------------+
	| L3/tests                          | Contains all test folders to run  |
	|                                   | software, hardware emulations     |
	|                                   | and hardware build.The tests cont |
	|                                   | ains folders with algorithm names.|
	|                                   | Each algorithm name folder contai |
	|                                   | ns the configuration folders,     |
	|                                   | inside configuration folders      |
	|                                   | makefile is present to run tests. |
	+-----------------------------------+-----------------------------------+
	| L3/examples/build                 | Contains xf_config_params.h file, |
	|                                   | which has configurable macros and |
	|                                   | varibales related to the particula|
	|                                   | r example.                        |
	+-----------------------------------+-----------------------------------+
	| L3/benchmarks                     | Contains benchmark examples to    |
	|                                   | compare the software              |
	|                                   | implementation versus FPGA        |
	|                                   | implementation using Vitis vision |
	|                                   | library.                          |
	+-----------------------------------+-----------------------------------+
	| ext                               | Contains the utility functions    |
	|                                   | related to opencl hostcode.       |
	+-----------------------------------+-----------------------------------+


.. include:: getting-started-with-vitis-vision.rst 
.. include:: using-the-vitis-vision-library.rst
.. include:: getting-started-with-hls.rst
.. include:: migrating-hls-video-library-to-vitis-vision.rst 
.. include:: design-examples.rst 
