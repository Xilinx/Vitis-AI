## Heston Closed Form Call Demo
This is a demonstration of the Heston Closed Form solver using the Vitis environment.  It supports software and hardware emulation as well as running the hardware accelerator on supported Alveo cards.

There are 2 demos described in more detail below:
- Run a series of tests and compare the results with pre-computed Heston Closed Form QuantLib results.
- Run a series of tests and compare the outputs and timings with a CPU C++ model.

In all cases the following test parameters can be changed:
- dw (the integration integral, default 0.5)
- w_max (the upper integration limit, default 200)
- tolerance (the accuracy required for a PASS in the comparison tests, default 0.0001)

## Prerequisites
- Alveo U200 installed and configured as per https://www.xilinx.com/products/boards-and-kits/alveo/u200.html#gettingStarted
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

            make all TARGET=hw DEVICE=xilinx_u200_xdma_201830_1
                Generates the hcf_host application and the FPGA xclbin file to run on a xilinx_u200_xdma_201830_2 
            make all TARGET=sw_emu DEVICE=xilinx_u200_xdma_201830_1
                Generates the hcf_host application to run a sofware simulation of the FPGA


## Running the demo
./compare_with_ql.sh data/sub_grid/test_data_11_1.txt 
Runs all the test cases in the file data/sub_grid/test_data_11_1.txt
Compares the calculated results with QuantLib generated results
Writes the results to results/res_test_data_11_1.txt

./compare_with_ql.sh -d0.25 data/sub_grid/test_data_11_1.txt
As above but uses a dw=0.025 in the integration

./compare_with_ql.sh -s data/sub_grid/test_data_11_1.txt
As above and displays timings

./hcf_host -fone.txt
Runs the test case specified in the file one.txt and displays the timings
The maximum number of test cases in a file is 1024

./hcf_host -fone.txt -v
Runs the test case specified in the file one.txt and displays the results and timings

./hcf_host -fone.txt -v -d0.25
Runs the test case specified in the file one.txt and displays the results and timings
The integration uses a dw = 0.25 and w_max = 200 (default)

./hcf_host -fone.txt -v -d0.25 -w400
Runs the test case specified in the file one.txt and displays the results and timings
The integration uses a dw = 0.5 (default) and w_max = 400

./hcf_host -fone.txt -c -v
Runs the test case specified in the file one.txt and compares the results of the FPGA with a CPU only implementation


## DATA FILES
data/original
tc_*_*.txt files contain model parameters for a series of tests used in the Xilinx Heston Finite Difference investigation.
There are 7 test suites 9, 11, 12, 13, 14, 15, 16
Each suite has 4 sub-suites where one of the model parameters is changed, keeping the others constant
Each sub-suite then has a fixed set of model parameters and varies s0 and v0 over a grid
s0 (initial stock price) varies from 0 - 800
vo (initial volatility) varies from 0 - 5
Each sub-suite therefore contains 8320 individual test cases

ql_*_*.csv files contain the QuantLib call prices for each of the test cases.

main.c is a program that combines the test case parameters and the QL expected values into a test_data file
It can also trim the s0/v0 grid to between max and min values.

make.sh builds the 'C' program -> executable -> generate_test_data

generate_test_data.sh runs the executable for each file in the directory
edit generate_test_data.sh to generate the fi=ull grid or a sub-grid; see comments in the shell script

## UTILS
one.txt                    a test data file with only one entry
summarise.sh               copy this file to a results diretory and run to get a summary of the results
compare_all_with_ql.sh     run this to test all files in the data directory

NOTES
Use compare_all_with_ql.sh to generate a PASS/FAIL for each test case in each test-suite
Use summarise.sh to get a summary of the results

Use generate_fpga_grid_csv.sh <test case file> to generate a csv output of FPGA values for that test case.
This can be used for the spreadsheets that generate the error surface for QL/Python/FPGA comparisons.
