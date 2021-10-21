aes256CbcDecrypt
=================

To profile performance of aes256CbcDecrypt, we prepare a datapack of 32K messages, each message is 1Kbyte.
We have 1 kernels, each kernel has 4 PUs.
Kernel utilization and throughput is shown in table below.

Executable Usage
================

* **Work Directory(Step 1)**

The steps for library download and environment setup can be found in :ref:`l1_vitis_security`. For getting the design,

```
   cd L1/benchmarks/aes256CbcDecrypt
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
   ./BUILD_DIR/host.exe -xclbin ./BUILD_DIR/aes256CbcDecryptKernel.xclbin
```

Input Arguments:

```
   Usage: host.exe -[-xclbin]
          -xclbin     binary;
```

* **Example output(Step 4)**

```
   Found Platform
   Platform Name: Xilinx
   Selected Device xilinx_u250_gen3x16_xdma_3_1_202020_1
   INFO: Importing build_dir.sw_emu.xilinx_u250_gen3x16_xdma_3_1_202020_1/aes256CbcDecryptKernel.xclbin
   Loading: 'build_dir.sw_emu.xilinx_u250_gen3x16_xdma_3_1_202020_1/aes256CbcDecryptKernel.xclbin'
   Kernel has been created.
   allocate to DDR
   DDR buffers have been mapped/copy-and-mapped
   4 channels, 2 tasks, 64 messages verified. No error found!
   Kernel has been run for 2 times.
   Total execution time 1413103us
```

Profiling 
=========

The aes256CbcDecrypt is validated on Xilinx Alveo U250 board. 
Its resource, frequency and throughput is shown as below.

    +-----------+------------+------------+---------+----------+-------+--------------+
    |Frequency  |     LUT    |     REG    |   BRAM  |   URAM   |  DSP  |  Throughput  |
    +-----------+------------+------------+---------+----------+-------+--------------+
    | 286MHz    | 203,595    |  312,900   |  761    |    0     |  29   | 4.7GB/s      |
    +-----------+------------+------------+---------+----------+-------+--------------+

