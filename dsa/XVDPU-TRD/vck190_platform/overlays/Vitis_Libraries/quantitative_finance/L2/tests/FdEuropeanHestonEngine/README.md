## Heston Equation Finite Difference Demonstration
This is a demonstration of the Heston Finite Difference solver using the Vitis environment.  It supports software and hardware emulation as well as running on a Xilinx Accelerator Card.

It uses a fixed set of test data produced via a Python model and a corresponding reference output, all stored in .csv format in the /data subdirectory.  The demo will take the input data, compute the pricing grid based using the kernel and then compare this to the reference data from the Python model.  The largest difference between the reference data and the kernel computed grid will be displayed.

## Prerequisites

- Xilinx Vitis 2020.1 installed and configured
- Xilinx runtime (XRT) installed
- Supported Xilinx Board (e.g. Alveo U250) installed and configured as per https://www.xilinx.com/products/boards-and-kits/alveo/u250.html#gettingStarted

## Building the Finite Difference demonstration
The demonstration application and kernel is built using a command line Makefile flow.

### Step 1 :
Setup the build environment using the Vitis and XRT scripts:

            source <install path>/Vitis/2020.1/settings64.sh
            source /opt/xilinx/xrt/setup.sh

### Step 2 :
Call the Makefile passing in the intended target and device. The Makefile supports software emulation, hardware emulation and hardware targets ('sw_emu', 'hw_emu' and 'hw', respectively). For example to build and run the test application:

            make check TARGET=sw_emu DEVICE=xilinx_u250_xdma_201830_2

Alternatively use 'all' to build the output products without running the application:

            make all TARGET=sw_emu DEVICE=xilinx_u250_xdma_201830_2

Several parameters can be passed to the Makefile in order to modify the configuration of the host and/or kernel. Most have restricted acceptable values and several must be consistent with others for the build to succeed.

| Makefile parameter | Default | Kernel/Host | Decription                                   | Valid Values                                |
|--------------------|---------|-------------|----------------------------------------------|---------------------------------------------|
|FD_DATATYPE         | double  | Kernel      | Type used to build kernel                    | float,   double                             |
|FD_DATAEQINTTYPE    | int64_t | Kernel      | Equivalent integer size to FD_DATATYPE       | int32_t, int64_t                            |
|FD_MEMWIDTH         | 8       | Kernel      | Number of data types that fit in a DDR word  | 8, 16 (equal to 512 / bits in FD_DATATYPE)  |
|FD_MSIZE            | 8192    | Kernel      | Matrix size M                                | Must be equal to M1 x M2 below              |
|FD_M1               | 128     | Host        | Matrix dimension in S direction              | Must be power-of-two                        |
|FD_M2               | 64      | Host        | Matrix dimension in V direction              | Must be power-of-two                        |
|FD_N                | 200     | Host        | Number of iterations of FD engine            |                                             |

In the case of the software and hardware emulations, the Makefile will build and launch the host code as part of the run.  These can be rerun manually using the following pattern:

            <host application> <xclbin> <data location> <M1> <M2> <N>

For example example to run a prebuilt software emulation output (assuming the standard build directories):

            export XCL_EMULATION_MODE=sw_emu
            ./bin_xilinx_u250_xdma_201830_2/fd_test.exe ./xclbin_xilinx_u250_xdma_201830_2_sw_emu/fd_heston_kernel_u250_m8192_double.xclbin ./data 128 64 200

Assuming an Alveo U250 card with the XRT configured, the hardware build is run as follows:

            unset XCL_EMULATION_MODE
            ./bin_xilinx_u250_xdma_201830_2/fd_test.exe ./xclbin_xilinx_u250_xdma_201830_2_hw/fd_heston_kernel_u250_m8192_double.xclbin ./data 128 64 200

## Example Output
The testbench will load precomputed data, process it via the FD engine and compare to the expected result, displaying the worst case difference. For example, the following is from the software emulation:

            Loading precomputed data from Python reference model...
            Opened ./data/ref_128x64_N200/A.csv OK
              - File has 73171 non-zeroes
              Padded data to 73176
              Padded row/column indices to 73184
            Opened ./data/ref_128x64_N200/A1.csv OK
            Opened ./data/ref_128x64_N200/A2.csv OK
            Opened ./data/ref_128x64_N200/X1.csv OK
            Opened ./data/ref_128x64_N200/X2.csv OK
            Opened ./data/ref_128x64_N200/b.csv OK
            Opened ./data/ref_128x64_N200/u0.csv OK
            Opened ./data/ref_128x64_N200/ref.csv OK
            Found Platform
            Platform Name: Xilinx
            INFO: Importing ./xclbin_xilinx_u250_xdma_201830_2_sw_emu/fd_heston_kernel_u250_m8192_double.xclbin
            Loading: './xclbin_xilinx_u250_xdma_201830_2_sw_emu/fd_heston_kernel_u250_m8192_double.xclbin'
            Maximum difference is 5.57066e-12, found at array index 2301

There will be a slight difference of the order 10^-12 for a double precision engine.  This is due to the ordering of floating point operations between the hardware implementation and the Python based reference model.
