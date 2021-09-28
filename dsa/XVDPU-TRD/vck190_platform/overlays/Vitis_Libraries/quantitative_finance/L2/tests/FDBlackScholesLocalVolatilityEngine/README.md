## Heston Equation Finite Difference Demonstration
This is a demonstration of a 1D Local Volatility Finite Difference solver using the Vitis environment.  It supports software and hardware emulation as well as running on a Xilinx Accelerator Card.

It uses a fixed set of test data produced via a Python model and a corresponding reference output, all stored in .csv format in the /data subdirectory.  The demo will take the input data, compute the pricing grid based using the kernel and then compare this to the reference data from the Python model.  The largest difference between the reference data and the kernel computed grid will be displayed.

## Prerequisites

- Xilinx Vitis 2020.1 installed and configured
- Xilinx runtime (XRT) installed
- Supported Xilinx Board (e.g. Alveo U200) installed and configured as per https://www.xilinx.com/products/boards-and-kits/alveo/u250.html#gettingStarted

## Building the Local Volatility demonstration
The demonstration application and kernel is built using a command line Makefile flow.

### Step 1 :
Setup the build environment using the Vitis and XRT scripts:

            source <install path>/Vitis/2020.1/settings64.sh
            source /opt/xilinx/xrt/setup.sh

### Step 2 :
Call the Makefile passing in the intended target and device. The Makefile supports software emulation, hardware emulation and hardware targets ('sw_emu', 'hw_emu' and 'hw', respectively). For example to build and run the test application:

            make run TARGET=sw_emu DEVICE=xilinx_u200_xdma_201830_2

Alternatively use 'all' to build the output products without running the application:

            make all TARGET=sw_emu DEVICE=xilinx_u200_xdma_201830_2

Several parameters can be passed to the Makefile in order to modify the configuration of the host and/or kernel. Most have restricted acceptable values and several must be consistent with others for the build to succeed.

| Makefile parameter | Default | Kernel/Host | Decription                                     | Valid Values                                |
|--------------------|---------|-------------|------------------------------------------------|---------------------------------------------|
|FD_DATA_TYPE        | float   | Kernel      | Type used to build kernel                      | float,   double                             |
|FD_DATA_EQ_TYPE     | int32_t | Kernel      | Equivalent integer size to FD_DATATYPE         | int32_t, int64_t                            |
|FD_N_SIZE           | 128     | Kernel      | Size of discretized grid in spatial dimension  | Must be power of two                        |
|FD_M_SIZE           | 128     | Kernel      | Size of discretized grid in temporal dimension | Max size, can pass smaller runtime value    |
|FD_NUM_PCR          | 2       | Kernel      | Number of units used by PCR solver             | Must be power of two                        |
|FD_LOG2_N_SIZE      | 8       | Kernel      | log2(FD_N)                                     | Must be consistent with FD_N_SIZE           |

In the case of the software and hardware emulations, the Makefile will build and launch the host code as part of the run.  These can be rerun manually using the following pattern:

            <host application> <xclbin> <testcase name> 

For example example to run a prebuilt software emulation output (assuming the standard build directories):

            ./bin_xilinx_u200_xdma_201830_2/fd_bs_lv_test.exe ./xclbin_xilinx_u200_xdma_201830_2_sw_emu/fd_bs_lv_kernel_N128_M256.xclbin case0

The data for a particular test is stored in ./data/<casename> in .csv format.  A parameters.csv file stores the information about the test.  For example, the case0 run above results in the following output:

            Loading testcase...
            Opened ./data/case0/parameters.csv OK
                N     = 128
                M     = 256
                theta = 0.5
                s     = 50
                k     = 60
            Loading precomputed data from Python reference model...
            Opened ./data/case0/xGrid.csv OK
            Opened ./data/case0/tGrid.csv OK
            Opened ./data/case0/sigma.csv OK
            Opened ./data/case0/rate.csv OK
            Opened ./data/case0/initialCondition.csv OK
            Opened ./data/case0/reference.csv OK
            Found Platform
            Platform Name: Xilinx
            INFO: Importing ./xclbin_xilinx_u200_xdma_201830_2_sw_emu/fd_bs_lv_kernel_N128_M256.xclbin
            Loading: './xclbin_xilinx_u200_xdma_201830_2_sw_emu/fd_bs_lv_kernel_N128_M256.xclbin'
            Launching kernel...
              Duration returned by profile API is 171.97 ms **** 
            Maximum difference is -0.00238037, found at array index 120

            solution
            [0.000000000e+00, 1.191890876e-28, 5.002537067e-28, 1.968369351e-27, 
            7.638963079e-27, 2.932134355e-26, 1.113217583e-25, 4.180017151e-25, 
            1.552128204e-24, 5.698726446e-24, 2.068601553e-23, 7.422880089e-23, 
                                            ...
            1.426361084e+02, 1.482684326e+02, 1.538988190e+02, 1.594927673e+02, 
            1.650125122e+02, 1.704188232e+02, 1.756721802e+02, 1.807335968e+02, ]

            reference
            [0.000000000e+00, 1.191910737e-28, 5.002622290e-28, 1.968404403e-27, 
            7.639101746e-27, 2.932188897e-26, 1.113239030e-25, 4.180098502e-25, 
                                            ...
            1.426384888e+02, 1.482708130e+02, 1.539011230e+02, 1.594947510e+02, 
            1.650141907e+02, 1.704201508e+02, 1.756728973e+02, 1.807335968e+02, ]

            difference
            [0.000000000e+00, -1.986115255e-33, -8.522240004e-33, -3.505192499e-32, 
            -1.386669560e-31, -5.454233603e-31, -2.144715586e-30, -8.135128085e-30, 
                                              ...
            -2.380371094e-03, -2.380371094e-03, -2.304077148e-03, -1.983642578e-03, 
            -1.678466797e-03, -1.327514648e-03, -7.171630859e-04, 0.000000000e+00, ]

(Full results display truncated here for clarity)

There will be a slight difference of the order 10^-3 for a single precision engine.  This is due to the ordering of floating point operations between the hardware implementation and the Python based reference model.
