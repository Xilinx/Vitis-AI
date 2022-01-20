## Level 1: HLS Functions and Modules

The Level 1 APIs are presented as HLS C++ classes and functions.

This level of API is mainly provide for hardware-savvy HLS developers. The API description and design details of these modules can be found in Vitis Vision User Guide. 

'examples' folder contains the testbench and accel C++ files that demonstrate the call of Vitis Vision functions in HLS flow.

'build' folder inside 'examples' folder has the configuration file that would help modify the default configuration of the function.

'include' folder contains the definitions of all the functions in various hpp files

'tests' folder has sub-folders named according to the function and the configuration it would run. Each individual folder has Makefiles and config files that would perform C-Simulation, Synthesis, Co-Simulation etc., of the corresponding function in the example folder using standalone Vivado HLS.


### Commands to run:

source < path-to-Vitis-installation-directory >/settings64.sh

source < part-to-XRT-installation-directory >/setup.sh

export DEVICE=< path-to-platform-directory >/< platform >.xpfm

export OPENCV_INCLUDE=< path-to-opencv-include-folder >

export OPENCV_LIB=< path-to-opencv-lib-folder >

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:< path-to-opencv-lib-folder >

make run CSIM=1 CSYNTH=1 COSIM=0

Note : Please read "Getting started with HLS" section of [Vitis Vision documentation](https://xilinx.github.io/Vitis_Libraries/vision/2021.2/index.html) for special cases, constraints and other full details.
