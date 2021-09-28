# Log Analyzer Demo

The demo is an integration frame included 3 part: Grok, GeoIP and JsonWriter. It supports software and hardware emulation as well as running hardware accelerators on the Alveo U200.

## Prerequisites

- Alveo U200 installed and configured as per https://www.xilinx.com/products/boards-and-kits/alveo/u200.html
- Xilinx Vitis 2020.1 or later installed and configured
- Xilinx runtime (XRT) installed

## Building

The demonstration application and kernel is built using a command line Makefile flow.

### Step 1

Setup the build environment using the Vitis and XRT scripts:

```
    source <install path>/Vitis/HEAD/settings64.sh
    source /opt/xilinx/xrt/setup.sh
```

### Step 2

Call the Makefile. For example:

```
    export PLATFORM_REPO_PATHS=/opt/xilinx/platforms
    make run TARGET=hw DEVICE=xilinx_u200_xdma_201830_2
```

The Makefile supports software emulation, hardware emulation and hardware targets ('sw_emu', 'hw_emu' and 'hw', respectively).

## Test Data

The `data` folder in this projects provides a sample input and corresponding golden result.
The Geo-IP information in this folder is also tailored to fit in git-repo.

To obtain full benchmark dataset, please refer to the `data/README.md`.

## Example Output

The testbench of process it via the engine and compare to the expected result, displaying the worst case difference. For example, the following is the key information from the software emulation:

```
    ----------------------log analyzer----------------
    DEBUG: found device 0: xilinx_u200_xdma_201830_2
    INFO: initilized context.
    INFO: initilized command queue.
    INFO: created program with binary build_dir.sw_emu.xilinx_u200_xdma_201830_2/logAnalyzer.xclbin
    INFO: built program.
    load log from disk to in-memory buffer
    load geoip database disk to in-memory buffer
    execute log analyzer
    The log file is partition into 1 slice with max_slice_lnm 102 and  takes 0.006000 ms.
    DEBUG: reEngineKernel has 4 CU(s)
    DEBUG: GeoIP_kernel has 1 CU(s)
    DEBUG: WJ_kernel has 1 CU(s)
    logAnalyzer pipelined, time: 1965.33 ms, size: 0 MB, throughput: 0 GB/s
    -----------------------------Finished logAnalyzer pipelined test----------------------------------------


```
## Profiling

The xclbin's Frequency is 231 MHz, and its hardware resource utilizations are listed in the following table.

| Name             | LUT     |  REG      | BRAM  | URAM  | DSP   |
| ---------------- | ------- |  -------- | ----- | ----- | ------|
| Platform         | 288185  |   414992  |  835  |   0   |   16  |
| GeoIP_kernel_1   |  29760  |    30549  |   24  |  16   |    8  |
| WJ_kernel_1      |  33366  |    33248  |   15  |   0   |    2  |
| reEngineKernel_1 |  41676  |    28726  |   66  |  48   |    3  |
| reEngineKernel_2 |  41664  |    28742  |   66  |  48   |    3  |
| reEngineKernel_3 |  41670  |    28760  |   66  |  48   |    3  |
| reEngineKernel_4 |  41660  |    28726  |   66  |  48   |    3  |

## Benchmark

- Input log: http://www.almhuette-raith.at/apache-log/access.log (1.2GB)
- logAnalyzer Demo execute time: 0.99 s, throughput: 1.2 GB/s
- Baseline `ref_result/ref_result.cpp` execute time: 53.1 s, throughput: 22.6 MB/s
- Accelaration Ratio: 53X

note:

1. The each line in the input log must has less than 4090 characters.
2. The baseline version is a single thread program.
