CRC32
========

To profile performance of crc32, we prepare a datapack of 268,435,456 byte messages as kernel input.
Base on U50, We have 1 kernel, each kernel has 1 PU.
Kernel utilization and throughput is shown in table below.

Executable Usage
================

* **Work Directory(Step 1)**

The steps for library download and environment setup can be found in :ref:`l1_vitis_security`. For getting the design,

```
   cd L1/benchmarks/crc32
```

* **Build kernel(Step 2)**

Please check you've installed openSSL and make sure that its version is 1.0.2 or higher. Command to check openSSL version:

```
    openssl version
```

Run the following make command to build your XCLBIN and host binary targeting a specific device. Please be noticed that this process will take a long time, maybe couple of hours.

```
   source /opt/xilinx/Vitis/2021.1/settings64.sh
   source /opt/xilinx/xrt/setenv.sh
   export DEVICE=u50_gen3x16
   export TARGET=hw
   make run 
```

* **Run kernel(Step 3)**

To get the benchmark results, please run the following command.

```
   ./BUILD_DIR/host.exe -xclbin ./BUILD_DIR/CRC32Kernel.xclbin -data PROJECT/data/test.dat -num 16
```

Input Arguments:

```
   Usage: host.exe -[-xclbin]
          -xclbin     binary;
```

* **Example output(Step 4)**

```
   kernel has been created
   kernel start------
   kernel end------
   Execution time 42.357ms
   Write DDR Execution time 1.17767 ms
   Kernel Execution time 40.8367 ms
   Read DDR Execution time 0.047911 ms
   Total Execution time 42.1537 ms
```


Profiling 
=========

The CRC32 is validated on Xilinx Alveo U50 board. 
Its resource, frequency and throughput is shown as below.

    +-----------+----------------+----------------+--------------+-------+----------+-------------+
    | Frequency |       LUT      |       REG      |     BRAM     |  URAM |    DSP   |  Throughput |
    +-----------+----------------+----------------+--------------+-------+----------+-------------+
    | 300 MHz   |     5,322      |     10,547     |     16       |   0   |     0    |   4.7 GB/s  |
    +-----------+----------------+----------------+--------------+-------+----------+-------------+

