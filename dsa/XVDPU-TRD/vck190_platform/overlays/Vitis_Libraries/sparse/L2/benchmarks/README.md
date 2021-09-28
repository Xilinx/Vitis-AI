# Benchmark Test Overview

We provide a benchmark for the SpMV function of the Vitis SPARSE library. The benchmark can be run on Alveo U280 (both hw_emu and hw run). The double precision 64-bit floating point data type is used in the benchmark.

## Prerequisites

### Vitis SPARSE Library
- Alveo U280 installed and configured as per https://www.xilinx.com/products/boards-and-kits/alveo/u280.html#gettingStarted (when running hardware)
- Xilinx runtime (XRT) installed
- Xilinx Vitis 2021.1 installed and configured

### Python3
Follow the steps as per https://xilinx.github.io/Vitis_Libraries/blas/2020.2/user_guide/L1/pyenvguide.html to set up Python3 environment.

### Datasets
- Datasets from https://sparse.tamu.edu/ will be downloaded during benchmarking process
- Matrix files: Matrix Market (.mtx) files

Table 1 Datasets for benchmarking

|matrix name   | rows| cols| NNZs  |
|--------------|-----|-----|-------|
|nasa2910      |2910 |2910 |174296 |
|ex9           |3363 |3363 |99471  |
|bcsstk24      |3562 |3562 |159910 |
|bcsstk15      |3948 |3948 |117816 |
|bcsstk28      |4410 |4410 |219024 |
|s3rmt3m3      |5357 |5357 |207695 |
|s2rmq4m1      |5489 |5489 |281111 |
|nd3k          |9000 |9000 |3279690|
|ted_B_unscaled|10605|10605|144579 |
|ted_B         |10605|10605|144579 |
|msc10848      |10848|10848|1229778|
|cbuckle       |13681|13681|676515 |
|olafu         |16146|16146|1015156|
|gyro_k        |17361|17361|1021159|
|bodyy4        |17546|17546|121938 |
|nd6k          |18000|18000|6897316|
|raefsky4      |19779|19779|1328611|
|bcsstk36      |23052|23052|1143140|
|msc23052      |23052|23052|1154814|
|ct20stif      |52329|52329|2698463|
|nasasrb       |54870|54870|2677324|
|bodyy6        |19366|19366|134748 |

## Building

### Download code

The sparse benchmark can be downloaded from [vitis libraries](https://github.com/Xilinx/Vitis_Libraries.git) ``master`` branch.

```
   git clone https://github.com/Xilinx/Vitis_Libraries.git
   cd Vitis_Libraries
   git checkout master
   cd sparse/L2/benchmarks/spmv_double
```

### Setup environment

Setup hardware build and run envrionment using the Vitis and XRT scripts:

```
    source <install path>/Vitis/2021.1/settings64.sh
    source /opt/xilinx/xrt/setup.sh
```

### Build the hardware and host executable

Run Makefile command. For example:

```
    make build TARGET=hw PLATFORM_REPO_PATHS=/opt/xilinx/platforms DEVICE=xilinx_u280_xdma_201920_3
    make host TARGET=hw PLATFORM_REPO_PATHS=/opt/xilinx/platforms DEVICE=xilinx_u280_xdma_201920_3
```

The Makefile supports various build target including hw emulation and hardware (hw_emu, hw)
