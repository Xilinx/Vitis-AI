## Population Markov Chain Mote Carlo (MCMC) Demonstration
This is a demonstration of the Population based MCMC built using the Vitis environment.  It supports software and hardware emulation as well as running the hardware accelerator on supported Alveo cards.

The demonstration run the kernel to generate configurable number of Samples. Samples are saved to a csv file for further analysis.

## Prerequisites

- Alveo U200 installed and configured as per https://www.xilinx.com/products/boards-and-kits/alveo/u200.html#gettingStarted
- Xilinx runtime (XRT) installed
- Xilinx Vitis 2020.1 installed and configured

## Building the demonstration
The kernel and host application is built using a command line Makefile flow.

### Step 1 :
Setup the build environment using the Vitis and XRT scripts:

            source <install path>/Vitis/2020.1/settings64.sh
            source /opt/xilinx/xrt/setup.sh

### Step 2 :
Call the Makefile passing in the intended target and platform (.xpfm). For example:

            make all TARGET=sw_emu DEVICE=xilinx_u250_xdma_201830_2

 The Makefile supports software emulation, hardware emulation and hardware targets ('sw_emu', 'hw_emu' and 'hw', respectively). The host application (mcmc_test) is written to the root of this demo folder, and the xclbin is delivered into the xclbin* folder with a name identifying the card and target.  For example the U200 software emulation build produces:

            ./xclbin_xilinx_u250_xdma_201830_2_sw_emu/mcmc_kernel.xclbin'

The xclbin is passed as a parameter to the host code along with the number of Samples to generate and burn_in samples.
The software emulation can be run as follows:

            export XCL_EMULATION_MODE sw_emu
            ./mcmc_test ./xclbin/mcmc_kernel.sw_emu.u200.xclbin 500 50

The hardware emulation can be run in a similar way, but a smaller number of parameters should be used as an RTL simulation is used under-the-hood:

            export XCL_EMULATION_MODE hw_emu
            ./bin_xilinx_u250_xdma_201830_2/mcmc_test.exe xclbin_xilinx_u250_xdma_201830_2_sw_emu/mcmc_kernel.xclbin 500 50

Assuming an Alveo U200 card with the XRT configured the hardware build is run in the same way.  Here a much large number of parameters should be used to fully exercise the DDR bandwidth:

            unset XCL_EMULATION_MODE
            ./bin_xilinx_u250_xdma_201830_2/mcmc_test.exe xclbin_xilinx_u250_xdma_201830_2_hw/mcmc_kernel.xclbin 500 50

The hardware build can be run in similar way. Here it's just another way of building and running an application with just a one line command:

            make run TARGET=hw DEVICE=xilinx_u250_xdma_201830_2

## Example Output
This is an example output from the demonstration using a sw_emu target.

    *************
    MCMC Demo v1.0
    *************

    Connecting to device and loading kernel...
    CRITICAL WARNING: [HW-EM 08-0] Unable to find emconfig.json. Using default device "xilinx:pcie-hw-em:7v3:1.0"
    Found Platform
    Platform Name: Xilinx
    INFO: Importing xclbin_xilinx_u250_xdma_201830_2_sw_emu/mcmc_kernel.xclbin
    Loading: 'xclbin_xilinx_u250_xdma_201830_2_sw_emu/mcmc_kernel.xclbin'
    Allocating buffers...
    Launching kernel...
    Duration returned by profile API is 30507.4 ms ****
    Kernel done!
    Processed 500 samples with 10 chains
    Samples saved to vitis_samples_out.csv
    Use Python plot_hist.py to plot histogram
