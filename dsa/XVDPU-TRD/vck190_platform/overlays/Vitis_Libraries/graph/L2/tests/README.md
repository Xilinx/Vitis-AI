# Vitis Tests for Kernels

This folder contains basic test for each of Graph kernels. They are meant to discover simple regression errors.

**These kernels have only been tested on Alveo U200 and U250, the makefile does not support other devices.**

To run the test, execute the following command:

```
source <install path>/Vitis/2019.2/settings64.sh
source /opt/xilinx/xrt/setup.sh
export PLATFORM_REPO_PATHS=/opt/xilinx/platforms
make run TARGET=sw_emu DEVICE=xilinx_u250_xdma_201830_2
```

`TARGET` can also be `hw_emu` or `hw`.
