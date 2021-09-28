.. meta::
   :keywords: Vision, Library, Vitis Vision AIE Library, overview, features, kernel
   :description: Using the Vitis Vision AIE library.
   :xlnxdocumentclass: Document
   :xlnxdocumenttype: Tutorials

.. _overview_aie:

Overview 
########
 
In the 2021.1 release, Vitis Vision library has added few functions which are implemented on `AI Engine™` of Xilinx ACAP Versal devices and validated on `VCK190`_ boards. These implementations exploit the VLIW, SIMD vector processing capabilities of `AI Engine™`_ .  

.. _AI Engine™ : https://www.xilinx.com/products/technology/ai-engine.html
.. _VCK190: https://www.xilinx.com/products/boards-and-kits/vck190.html

.. _basic-features-aie:

Basic Features
===============
To process high resolution images, xfcvDataMovers component is also provided which divides the image into tiled-units and uses efficient data-movers to manage the transfer of tiled-units to and from AIEngine™ cores. You can find more information on the types of data-movers and their usage, in the :ref:`Getting Started with Vitis Vision AIEngine Library Functions <_aie_prerequisites>` section.  


Vitis Vision AIE Library Contents
==================================

Vitis Vision AIEngine™ files are organized into the following directories: 

.. table:: Table Vitis Vision AIE Library Contents

   +----------------------------------------------------------+---------------------------------------------------------------------------------------------+
   | Folder                                                   | Details                                                                                     |
   +==========================================================+=============================================================================================+
   | L1/include/aie/imgproc                                   | contains header files of vision AIEngine™ functions                                         |
   +----------------------------------------------------------+---------------------------------------------------------------------------------------------+
   | L1/include/aie/common                                    | contains header files of data-movers and other utility functions                            |
   +----------------------------------------------------------+---------------------------------------------------------------------------------------------+
   | L1/lib/sw                                                | contains the data-movers library object files                                               |
   +----------------------------------------------------------+---------------------------------------------------------------------------------------------+
   | L2/tests/aie                                             | contains the ADF graph code and host-code using data-movers and vision AIEngine™ functions  |
   |                                                          | from L1/include/aie                                                                         |
   +----------------------------------------------------------+---------------------------------------------------------------------------------------------+
   
   
.. include:: include/getting-started-with-vitis-vision-aie.rst
.. include:: include/vitis-aie-design-methodology.rst
.. include:: include/functionality-evaluation-aie.rst
.. include:: include/design-example-aie.rst
