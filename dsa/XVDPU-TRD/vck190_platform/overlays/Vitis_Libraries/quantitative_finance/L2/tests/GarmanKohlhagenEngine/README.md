## Garman-Kohlhagen Closed Form Demonstration
This is a demonstration of the Garman-Kohlhagen (GK) closed form solver built using the Vitis environment.  It supports software and hardware emulation as well as running the hardware accelerator on supported Alveo cards.

The demonstration takes a file of input parameter sets (one input parameter set consists of the underlying, volatility, domestic rate, foreign rate, time-to-maturity and strike price), passes them to the kernel and retrieves the pricing and associated Greeks.  These are then compared to a full precision model and the worst case difference between the kernel and model are displayed.

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

Alternatively use 'all' to build the output products without running the application:

            make all TARGET=sw_emu DEVICE=xilinx_u250_xdma_201830_2

For all Makefile targets, the host application and xclbin are delivered to named folders depending on the target and part selected.  For example, the command above will produce:

            ./bin_xilinx_u250_xdma_201830_2/gk_test.exe
            ./xclbin_xilinx_u250_xdma_201830_2_sw_emu/gk_kernel.xclbin

These output products can be used directly from the command line.  The application takes the xclbin as the first argument along followed by the number of test parameters to generate.  Due the parallel nature of the processing, the kernel processes input sets in multiples of 16 so the host will round up the number of parameters if required.


The software emulation can be run as follows:

            export XCL_EMULATION_MODE=sw_emu
            ./bin_xilinx_u250_xdma_201830_2/gk_test.exe ./xclbin_xilinx_u250_xdma_201830_2_sw_emu/gk_kernel.xclbin test_vectors_gk.txt

The hardware emulation can be run in a similar way, but a smaller number of parameters should be used as an RTL simulation is used under-the-hood:

            export XCL_EMULATION_MODE=hw_emu
            ./bin_xilinx_u250_xdma_201830_2/gk_test.exe ./xclbin_xilinx_u250_xdma_201830_2_hw_emu/gk_kernel.xclbin one.txt

Assuming an Alveo U250 card with the XRT configured, the hardware build can be run in the same way.  Here a much large number of parameters should be used to fully exercise the DDR bandwidth. Another test vector file can be created with more entries:

            unset XCL_EMULATION_MODE
            ./bin_xilinx_u250_xdma_201830_2/gk_test.exe ./xclbin_xilinx_u250_xdma_201830_2_hw/gk_kernel.xclbin <larger test vector file>.txt

## Example Output
This is an example output from the demonstration using a sw_emu target.


            *************
            BSM Demo v1.0
            *************

            Generating randomized data and reference results...
            Connecting to device and loading kernel...
            Found Platform
            Platform Name: Xilinx
            INFO: Importing <project_root>/xclbin_xilinx_u250_xdma_201830_2_sw_emu/gk_kernel.xclbin
            Loading: '<project_root>/xclbin_xilinx_u250_xdma_201830_2_sw_emu/gk_kernel.xclbin'
            Allocating buffers...
            Launching kernel...
              Duration returned by profile API is 16921.7 ms ****
            Kernel done!
            Comparing results...
            Processed 16384 call options:
            Throughput = 0.000968226 Mega options/sec

              Largest host-kernel price difference = -5.07147e-05
              Largest host-kernel delta difference = 3.3043e-07
              Largest host-kernel gamma difference = 3.56924e-08
              Largest host-kernel vega difference  = 3.19532e-07
              Largest host-kernel theta difference = -3.94425e-08
              Largest host-kernel rho difference   = -1.11218e-06


There will be a difference seen between the kernel and host data.  This error arises from the use of floats for the internal kernel processing and the approximation to erfc() which is required by the closed-form solution.
