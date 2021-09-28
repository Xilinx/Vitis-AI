# Graph Library L3

In order to use Graph L3, the sofware and hardware requirements should be met and the shared library (`libgraphL3.so`) should be built and linked in the users application.  

## Requirements
### Software Requirements
- CentOS/RHEL 7.8 and Ubuntu 16.04 LTS
- [Xilinx RunTime (XRT) 2020.1](https://github.com/Xilinx/XRT)
- [Xilinx FPGA Resource Manager (XRM) 2020.2](https://github.com/Xilinx/XRM)


### Hardware Requirements
- [Alveo U50](https://www.xilinx.com/products/boards-and-kits/alveo/u50.html)

## Folder contents:  
- `lib`: libgraphL3.so and makefile  
- `include`: header files. xf_graph_L3.hpp can be found in the `include` folder  
- `src`: source codes. xf_graph_L3.cpp can be found here, which is used for generating libgraphL3.so   
- `tests`: C++ testcases.   

## Getting Started
- [Build dynamic library](lib)  
- [Run testcase](tests)  
