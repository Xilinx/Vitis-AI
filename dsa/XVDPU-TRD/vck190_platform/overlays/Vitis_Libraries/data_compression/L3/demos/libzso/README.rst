====================
Zlib (SO) Demo
====================

Infrastructure
--------------

List below presents infrastructure required to build & deploy this demo.
Mandatory requirements are marked accordingly in order to get this demo working in
deployment environment. Vitis is required only for development.

    ``Vitis: 2020.2_released (Only for Developers)``
    
    ``XRT: 2020.2_PU1 (Mandatory)``
    
    ``XRM: 2020.2_released (Mandatory)``
    
    ``SHELL: u50_gen3x16_xdma_201920_3 (Mandatory)``
    
    
Application Usage
-----------------

**Compression**     -->  ``./xzlib input_file``

**Decompression**   -->  ``./xzlib -d input_file.zlib``

**Test Flow**       -->  ``./xzlib -t input_file`` 

**No Acceleration** -->  ``./xzlib -t input_file -n 0`` 

**Help**           -->  ``./xzlib -h``

**Regression**     --> Refer ``run.sh`` script to understand usage of various options provided with ``xzlib`` utility. 


Deployment
----------

**#1** ``source ./scripts/setup.csh <absolute path to xclbin>`` --> Setup of XRT, XRM, Environment Variables, XRM Daemon Start done

**#2**  ``./scripts/xrmxclbin.sh <number of devices>`` --> XRM loads XCLBIN to devices as per user input
 
**#3** Build libz.so (Shared Object file) - ``make lib ENABLE_INFLATE_XRM=yes`` --> Current directory contains libz.so.1.2.7

**#4** Build xzlib (Host Executable ) - ``make host `` --> ./build directory contains ``xzlib`` executable

Note: By default host executable is built for both compress/decompress on FPGA.

References
----------

**[XRM Build and Installation]**: https://xilinx.github.io/XRM/Build.html

**[XRM Test Instructions]**: https://xilinx.github.io/XRM/Test.html
