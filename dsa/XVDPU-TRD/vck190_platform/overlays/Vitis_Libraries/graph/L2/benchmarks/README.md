# Benchmark Test Overview

Here are benchmarks of the Vitis Graph Library using the Vitis environment and comparing with Spark (v3.0.0) GraphX. It supports software and hardware emulation as well as running hardware accelerators on the Alveo U250.

## Prerequisites

### Vitis Graph Library
- Alveo U250 installed and configured as per [Alveo U250 Data Center Accelerator Card](https://www.xilinx.com/products/boards-and-kits/alveo/u250.html#gettingStarted)
- Xilinx runtime (XRT) installed
- Xilinx Vitis 2020.2 installed and configured

### Spark
- Spark 3.0.0 installed and configured
- Spark running on platform with Intel(R) Xeon(R) CPU E5-2690 v4 @2.600GHz, 56 Threads (2 Sockets, 14 Core(s) per socket, 2 Thread(s) per core)

### Datasets
- Datasets from https://sparse.tamu.edu/
- Format requirement: compressed sparse row (CSR) or compressed sparse column (CSC).

Table 1 Datasets for benchmark

|   Datasets         |  Vertex  |   Edges   |   Degree    |
|--------------------|----------|-----------|-------------|
|  as-Skitter        | 1694616  |  11094209 | 6.546739202 |
|  coPapersDBLP      | 540486   |  15245729 | 28.20744478 |
|  coPapersCiteseer  | 434102   |  16036720 | 36.94228545 |
|  cit-Patents       | 3774768  |  16518948 | 4.37614921  |
|  europe_osm        | 50912018 |  54054660 | 1.061726919 |
|  hollywood         | 1139905  |  57515616 | 50.45649945 |
|  soc-LiveJournal1  | 4847571  |  68993773 | 14.23264827 |
|  ljournal-2008     | 5363260  |  79023142 | 14.73416206 |
|  patients          | 1250000  |  200      |      -      |

## Building

Here, TriangleCount is taken as an example to indicate how to build the application and kernel with the command line Makefile flow.

- ### Download code

These graph benchmarks can be downloaded from [vitis libraries](https://github.com/Xilinx/Vitis_Libraries.git) ``master`` branch.

```
   git clone https://github.com/Xilinx/Vitis_Libraries.git
   cd Vitis_Libraries
   git checkout master
   cd graph
```

- ### Setup environment

Specifying the corresponding Vitis, XRT, and path to the platform repository by running following commands.

```
   source <intstall_path>/installs/lin64/Vitis/2020.2/settings64.sh
   source /opt/xilinx/xrt/setup.sh
   export PLATFORM_REPO_PATHS=/opt/xilinx/platforms
```
