## Quanto price and Greeks Demonstration
This is a demonstration of the Quanto price and Greeks built using the Vitis environment.  It supports software and hardware emulation as well as running the hardware accelerator on supported Alveo cards.

The demonstration takes a file of input parameter sets (one input parameter set consists of the underlying, volatility, domestic rate, foreign rate, time-to-maturity, strike price, and so on), passes them to the kernel and retrieves the pricing and associated Greeks.  These are then compared to a full precision model and the worst case difference between the kernel and model are displayed.

## Prerequisites

- Xilinx Vitis 2020.1 installed and configured
- Xilinx runtime (XRT) installed
- Supported Xilinx Board (e.g. Alveo U250) installed and configured as per https://www.xilinx.com/products/boards-and-kits/alveo/u250.html#gettingStarted


## Building the demonstration
The kernel and host application are built using a command line Makefile flow.

### Step 1 :
Setup the build environment using the Vitis and XRT scripts:

            source <install path>/Vitis/2020.1/settings64.sh
            source /opt/xilinx/xrt/setup.sh

### Step 2 :
Call the Makefile passing in the intended target and device. The Makefile supports software emulation, hardware emulation and hardware targets ('sw_emu', 'hw_emu' and 'hw', respectively). For example to build and run the test application:

            make run TARGET=sw_emu DEVICE=xilinx_u250_xdma_201830_2

For all Makefile targets, the host application and xclbin are delivered to named folders depending on the target and part selected.  For example, the command above will produce:

            ./bin_xilinx_u250_xdma_201830_2/quanto_test.exe
            ./xclbin_xilinx_u250_xdma_201830_2_sw_emu/quanto_kernel.xclbin

These output products can be used directly from the command line.  The application takes the xclbin as the first argument along followed by the number of test parameters to generate.

Assuming an Alveo U250 card with the XRT configured, the hardware build can be run in the same way.  Here a much large number of parameters should be used to fully exercise the DDR bandwidth. Another test vector file can be created with more entries:

            ./bin_xilinx_u250_xdma_201830_2/quanto_test.exe ./xclbin_xilinx_u250_xdma_201830_2_hw/quanto_kernel.xclbin <larger test vector file>.txt

To delete all xclbin and host binary, you may want to use:

			make cleanall

## Example Output
This is an example output from the demonstration using a sw_emu target.

			*********************
			Quanto call Demo v1.0
			*********************

			Generating reference results...
			Running CPU model...
			Connecting to device and loading kernel...
			Found Platform
			Platform Name: Xilinx
			INFO: Importing <project_root>/xclbin_xilinx_u250_xdma_201830_2_sw_emu/quanto_kernel.xclbin
			Loading: '<project_roow>/xclbin_xilinx_u250_xdma_201830_2_sw_emu/quanto_kernel.xclbin'
			Allocating buffers...
			Launching kernel...
			Kernel done!
			Comparing results...
			Processed 1 call options:
			Throughput = 9.82355e-05 Mega options/sec

			Largest host-kernel price difference = -3.15396e-07
			Largest host-kernel delta difference = -4.65582e-08
			Largest host-kernel gamma difference = 8.15404e-11
			Largest host-kernel vega difference  = 2.77749e-08
			Largest host-kernel theta difference = -1.65196e-09
		    Largest host-kernel rho difference   = -1.27437e-07
			CPU execution time                          = 31us
			FPGA time returned by profile API           = 10.1796 ms
			FPGA execution time (including mem transfer)= 11186us
			Total tests = 1
			Total fails = 0
			Tolerance   = 1e-05

There will be a difference seen between the kernel and host data.  This error arises from the use of floats for the internal kernel processing.
