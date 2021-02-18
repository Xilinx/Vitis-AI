# Overlaybins
This directory is used to hold hardware overlays for accelerating neural networks on different hardware platforms. 

## Background
Field Programmable Gate Arrays - [FPGA](https://www.xilinx.com/products/silicon-devices/fpga/what-is-an-fpga.html)s -
are semiconductor devices, which by design implement an array of logic blocks with a programmable interconnect.
Unlike "hardened" devices (i.e. CPU/GPU) FPGAs can be programmed to implement a hardware design that does specifically what the user wants.
After the design of the hardware system, the FPGA must be programmed using a binary file. This process is typically referred to as configuration.
Furthermore, in a usecase where there is fixed functionality, and dynamic functionality an FPGA can be partially reconfigured.
In a datacenter environment, the FPGA is always connected to the CPU via PCIe, and it is always connected to external off-chip memory.
Given the above assumption the FPGA binary can be partitioned into a static shell (Xilinx uses the term DSA), and a dynamic overlay (Xilinx uses the term xclbin).
The static shell must be loaded prior to the loading of any overlay. Xilinx's cloud partners have already loaded the shell.
If this explanation seems a bit confusing, don't worry too much about it. Software will automatically detect the platform, and load the appropriate xclbin.
