aes256CbcEncrypt
=================

To profile performance of aes256CbcEncrypt, we prepare a datapack of 96K messages, each message is 1Kbyte.
We have 4 kernels, each kernel has 12 PUs.
Kernel utilization and throughput is shown in table below.

Executable Usage
================

* **Work Directory(Step 1)**

The steps for library download and environment setup can be found in :ref:`l1_vitis_security`. For getting the design,

```
   cd L1/benchmarks/aes256CbcEncrypt
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
   ./BUILD_DIR/host.exe -xclbin ./BUILD_DIR/aes256CbcEncryptKernel.xclbin
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
   Selected Device xilinx_u250_xdma_201830_2
   INFO: Importing build_dir.sw_emu.xilinx_u250_xdma_201830_2/aes256CbcEncryptKernel.xclbin
   Loading: 'build_dir.sw_emu.xilinx_u250_xdma_201830_2/aes256CbcEncryptKernel.xclbin'
   Kernel has been created.
   DDR buffers have been mapped/copy-and-mapped
   12 channels, 2 tasks, 64 messages verified. No error found!
   Kernel has been run for 2 times.
   Total execution time 4725069us
```


Profiling 
=========

The aes256CbcEncrypt is validated on Xilinx Alveo U250 board. 
Its resource, frequency and throughput is shown as below.

    +-----------+----------------+----------------+--------------+-------+----------+-------------+
    | Frequency |      LUT       |      REG       |     BRAM     |  URAM |    DSP   |  Throughput |
    +-----------+----------------+----------------+--------------+-------+----------+-------------+
    | 224MHz    |    1,059,093   |    1,010,145   |     654      |   0   |    152   |    5.5GB/s  |
    +-----------+----------------+----------------+--------------+-------+----------+-------------+


