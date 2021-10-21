# Benchmark Test Overview

Here are benchmarks of the Vitis BLAS library using the Vitis environment. It supports software and hardware emulation as well as running hardware accelerators on the Alveo U250.

## Prerequisites

### Vitis BLAS Library
- Alveo U250 installed and configured as per https://www.xilinx.com/products/boards-and-kits/alveo/u250.html#gettingStarted (when running hardware)
- Xilinx runtime (XRT) installed
- Xilinx Vitis 2021.1 installed and configured

## Building

Taken gemm_4CU as an example to indicate how to build the application and kernel with the command line Makefile flow.

### Download code

These blas benchmarks can be downloaded from [vitis libraries](https://github.com/Xilinx/Vitis_Libraries.git) ``master`` branch.

```
   git clone https://github.com/Xilinx/Vitis_Libraries.git
   cd Vitis_Libraries
   git checkout master
   cd blas
```

### Setup environment

Setup and build envrionment using the Vitis and XRT scripts:

```
    source <install path>/Vitis/2021.1/settings64.sh
    source /opt/xilinx/xrt/setup.sh
```

### Build and run the kernel :

Run Makefile command. For example:

```
    make run TARGET=hw PLATFORM_REPO_PATHS=/opt/xilinx/platforms DEVICE=xilinx_u250_xdma_201830_2
```

The Makefile supports various build target including software emulation, hw emulation and hardware (sw_emu, hw_emu, hw)

The host application could be run manually using the following pattern:

```
    <host application> <xclbin> <argv>
```

For example:

```
    build_dir.hw.xilinx_u250_xdma_201830_2/host.exe build_dir.hw.xilinx_u250_xdma_201830_2/blas.xclbin 64 64 64
```