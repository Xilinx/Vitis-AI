## Numerical Integration Test
This is a demonstration of using the Numerical Integration library in the Heston Closed Form pricing engine.  It supports software and hardware emulation as well as running the hardware accelerator on supported Alveo cards.

It uses pre-canned test data generated from a host Heston Closed Form Model and compares the results. The kernel implements the hcf test (from the Vitis library) replacing the simple 'Trapezoidal Rule' of the original with the Romberg method.

## Prerequisites
- Alveo U200 installed and configured as per https://www.xilinx.com/products/boards-and-kits/alveo/u200.html#gettingStarted
- Xilinx runtime (XRT) installed
- Xilinx Vitis 2020.1 installed and configured

## Building the Test
The demonstration application and kernel is built using a command line Makefile flow.

### Step 1 :
Setup the build environment using the Vitis and XRT scripts:

            source <install path>/Vitis/2020.1/settings64.sh
            source /opt/xilinx/xrt/setup.sh

### Step 2 :
Call the Makefile. For example:

	make check TARGET=sw_emu DEVICES=xilinx_u200_xdma_201830_2
        

The Makefile supports software emulation, hardware emulation and hardware targets ('sw_emu', 'hw_emu' and 'hw', respectively).  




If the make check option is used the demo will automatically be lauched.

