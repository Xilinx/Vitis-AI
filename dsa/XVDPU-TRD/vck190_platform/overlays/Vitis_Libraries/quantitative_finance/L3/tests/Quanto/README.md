
## Quanto Test

This test shows how to utilize the Quanto Model in the L3 framework.

## Prerequisites
- Alveo card (eg U200) installed and configured as per https://www.xilinx.com/products/boards-and-kits/alveo/u200.html#gettingStarted
- Xilinx runtime (XRT) installed
- Xilinx Vitis 2020.1 installed and configured

## Build Instuctions
The demonstration application and kernel is built using a command line Makefile flow.

### Step 1:
Setup the build environment using Vitis and XRT scripts:

            source <install path>/Vitis/2020.1/settings64.sh
            source /opt/xilinx/xrt/setup.sh

### Step 2 :
Call the Makefile. For example:

	make check TARGET=sw_emu DEVICE=xilinx_u200_xdma_201830_2

	make check TARGET=hw_emu DEVICE=xilinx_u200_xdma_201830_2

	make all TARGET=hw DEVICE=xilinx_u200_xdma_201830_2
        
	make run TARGET=hw DEVICE=xilinx_u200_xdma_201830_2

The Makefile supports software emulation, hardware emulation and hardware targets ('sw_emu', 'hw_emu' and 'hw', respectively).

If the make check or run option is used the demo will automatically be lauched.
