## Merton Jump Diffusion Call Demo
This is a demonstration of the Merton Jump Diffusion solver using the Vitis environment.  It supports software and hardware emulation as well as running the hardware accelerator on supported Alveo cards.

The demo runs a test file containing test cases. The test case has the format:

            S=<value>, K=<value>, r=<value>, sigma=<value>, T=<value>, lambda=<value>, kappa=<value>, delta=<value>, N=<value>, exp=<value>

            S - current stock price
            K - strike price
            r - interest rate
            sigma - volatility
            T - time to vest (years)
            lambda - mean jump per unit time
            kappa - Expected[Y-1] Y is the random variable
            delta - delta^2 = variance of ln(Y)
            N - number of terms in the finite sum approximation
            exp - the expected result

The demo splits the test data into batches of 2048 to send to the FPGA.
The demo can compare FPGA generated results with C Model generated results, or with the expected values in the test data. It can also compare the C Model results with the expected values.
The demo can also do a performance comparison between the FPGA solution and an un-optimised C Model solution.

## Prerequisites
- Alveo card (eg U200) installed and configured as per https://www.xilinx.com/products/boards-and-kits/alveo/u200.html#gettingStarted
- Xilinx runtime (XRT) installed
- Xilinx Vitis 2020.1 installed and configured

## Building the demonstration
The demonstration application and kernel is built using a command line Makefile flow.

### Step 1.
Setup the build environment using the Vitis and XRT scripts:

            source <install path>/Vitis/2020.1/settings64.sh
            source /opt/xilinx/xrt/setup.sh

### Step 2.
Call the makefile:
The Makefile is set up to generate a 'float' FPGA solution
To build a 'double' FPGA, edit Makefile and edit DT=double

            make all TARGET=hw
                Generates the m76_demo application and the FPGA xclbin file to run on a xilinx_u200_xdma_201830_2 (default)

            make all TARGET=hw DEVICE=xilinx_u250_xdma_201830_1
                Generates the m76_demo application and the FPGA xclbin file to run on a xilinx_u250_xdma_201830_1 card

            make all TARGET=sw_emu
                Generates the m76_demo application to run a sofware simulation of the FPGA

## Running the demo
### Example 1:

        ./m76_demo -f example_test_data.csv -c
Runs all the test cases in the file example_test_data.csv
Compares the calculated results with the expected values using the default tolerance of 0.001
Displays differences
The output looks like:

            CPU:
            FPGA:
            Found Platform
            Platform Name: Xilinx
            INFO: Importing xclbin/m76_hw_u200_float.xclbin
            Loading: 'xclbin/m76_hw_u200_float.xclbin'
            FAIL (37): S=100.248, K=100, sigma=2.0798, T=3, r=0, lambda=2, kappa=-0.2, delta=0.8, exp=96.024
                  expected(96.024) got(96.0251) diff(0.00112915)

            FAIL (42): S=120.263, K=100, sigma=0.7752, T=3, r=0, lambda=2, kappa=-0.2, delta=0.8, exp=92.9625
                  expected(92.9625) got(92.9636) diff(0.00111389)

            FAIL (43): S=120.263, K=100, sigma=1.202, T=3, r=0, lambda=2, kappa=-0.2, delta=0.8, exp=102.62
                  expected(102.62) got(102.621) diff(0.00120544)

            FAIL (44): S=120.263, K=100, sigma=2.0798, T=3, r=0, lambda=2, kappa=-0.2, delta=0.8, exp=115.646
                  expected(115.646) got(115.647) diff(0.00138092)

            FAIL (49): S=202.555, K=100, sigma=0.7752, T=3, r=0, lambda=2, kappa=-0.2, delta=0.8, exp=168.514
                  expected(168.514) got(168.516) diff(0.00198364)

            FAIL (50): S=202.555, K=100, sigma=1.202, T=3, r=0, lambda=2, kappa=-0.2, delta=0.8, exp=180.282
                  expected(180.282) got(180.285) diff(0.00212097)

            FAIL (51): S=202.555, K=100, sigma=2.0798, T=3, r=0, lambda=2, kappa=-0.2, delta=0.8, exp=196.646
                  expected(196.646) got(196.649) diff(0.00234985)

            Total tests  = 49
            Total passes = 42
            Total fails  = 7


### Example 2:

        ./m76_demo -f example_test_data.csv -t 0.01 -c

Runs all the test cases in the file example_test_data.csv

Compares the calculated results with the expected values using a tolerance of 0.01

Displays differences

### Example 3:

        ./m76_demo -f example_test_data.csv -x

Runs all the test cases in the file example_test_data.csv

Compares the calculated results with a C model using the default tolerance of 0.001

Displays differences

### Example 4:

        ./m76_demo -f example_test_data.csv -C

Runs all the test cases in the file example_test_data.csv

Compares the C model calculated results with the expected values using the default tolerance of 0.001

Displays differences

### Example 5:

        ./m76_demo -f 10004_entry_test_data_file.csv -s

Runs the test file data through the FPGA and C Model and compares their speeds of execution. Note that due to the overhead of transferring input data to the FPGA and retrieving results from the FPGA, the FPGA will provide increasing performance advantages when larger and larger amounts of data are processed.

The output looks like the following. It can be seen that the test data file has been split into 4 batches of 2048 plus a final slightly smaller batch containing the rest. The first batch to the FPGA takes slightly longer as XRT is being setup.

            FPGA:
            Found Platform
            Platform Name: Xilinx
            INFO: Importing xclbin/m76_hw_u200_float.xclbin
            Loading: 'xclbin/m76_hw_u200_float.xclbin'
            Parse file time    = 148411us
            Import binary time = 156758us
            FPGA
                1259us
                835us
                873us
                811us
                790us
                mean = 913
            CPU
                106420us
                106373us
                106383us
                106347us
                94089us
                mean = 103922

## Example Test Data file
This file is part of a large internal test suite. The expected values have been generated using Quantlib configured with a Merton76Process, a JumpDiffusionEngine and an Actual365Fixed day counter. The Quantlib results are in places slightly different from the C Model and FPGA. Running the above demo will bring these out. Explanation TODO!

# Demo Explanation
Merton Jump Diffusion basically sums a number of Black Scholes calls with varying parameters in order to come up with a single call price. In order to optimise kernel performance, the BS calculations and the summing are both done from the engine wrapper. This lets both calls be pipelined. Having the summing done within the main engine call leads to much larger latencys and a less efficient solution.

## L2/demos/M76Engine/src/m76_kernel.cpp
This is the kernel wrapper which accepts an array of structures of Jump Diffusion parameters; each element in the array contains the parameters for a single Jump Diffusion call price solution. Each element is passed to the Jump Diffusion engine which returns an array of the individual Black Scholes prices which have to be summed. The wrapper then calls the summing function. In this example the engine call is unrolled by a factor of 8 and the arrays are partitioned with a factor of 8 in order to allow concurrent access by 8 parallel engines.

## L2/include/xf_fintech/m76_engine.h
This is the engine which contains the call to produce a number of Black Ssholes call process for varying parameters. These individual call prices are placed in an array a passed back to the caller to be summed later with the engine summing function.
