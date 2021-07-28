## 1. VCK5000 Card Setup in Host

We provide some scripts to help to automatically finish the Alveo card setup process. You could refer to these to understand the required steps. To use the scripts, just input the command below. It will detect Operating System you are using, then download and install the appropriate packages.

**Please note you should use this script in host environment, namely out of the Docker container.**

~~~
source ./install.sh
~~~

The following installation steps were performed in this script.

* Install XRT.
* Install XRM. The [Xilinx Resource Manager (XRM)](https://github.com/Xilinx/XRM/) manages and controls FPGA resources on a machine. It is used by the Alveo runtime.
* Install DPU V4E xclbin for VCK5000.

## 2. Install VCK5000 Shell packages. 
	Please download VCK5000 shell from the following links and install it on your X86 host machine.
  
  |            OS               | RedHat / CentOS                                                |
  |-----------------------------|----------------------------------------------------------------|
  | Deployment Target Platform  | [xilinx-vck5000-es1-gen3x16-platform-2-1.noarch.rpm.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=xilinx-vck5000-es1-gen3x16-platform-2-1.noarch.rpm.tar.gz)      | 
  | Development Target Platform | [xilinx-vck5000-es1-gen3x16-2-202020-1-dev-1-3123623.noarch.rpm](https://www.xilinx.com/member/forms/download/eula-xef.html?filename=xilinx-vck5000-es1-gen3x16-2-202020-1-dev-1-3123623.noarch.rpm) |

  |            OS               | Ubuntu                                                         |
  |-----------------------------|----------------------------------------------------------------|
  | Deployment Target Platform  | [xilinx-vck5000-es1-gen3x16-platform-2-1_all.deb.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=xilinx-vck5000-es1-gen3x16-platform-2-1_all.deb.tar.gz)      | 
  | Development Target Platform | [xilinx-vck5000-es1-gen3x16-2-202020-1-dev_1-3123623_all.deb](https://www.xilinx.com/member/forms/download/eula-xef.html?filename=xilinx-vck5000-es1-gen3x16-2-202020-1-dev_1-3123623_all.deb) |
  
  The following shows the steps to install VCK5000 shell on CentOS 7.8. For Ubuntu OS, please use “sudo apt install …” rather than “sudo yum install…”.

a) Running lspci to check that the VCK5000 Card has been installed

```shell
lspci -vd 10ee:
```

An output similar to the following example is seen. 

```
02:00.0 Processing accelerators: Xilinx Corporation Device 5044

        Subsystem: Xilinx Corporation Device 000e

        Flags: bus master, fast devsel, latency 0, IRQ 16, NUMA node 0

        Memory at 380030000000 (64-bit, prefetchable) [size=128M]

        Memory at 380038020000 (64-bit, prefetchable) [size=128K]

        Capabilities: <access denied>

        Kernel driver in use: xclmgmt

        Kernel modules: xclmgmt

02:00.1 Processing accelerators: Xilinx Corporation Device 5045

        Subsystem: Xilinx Corporation Device 000e

        Flags: bus master, fast devsel, latency 0, IRQ 17, NUMA node 0

        Memory at 380038000000 (64-bit, prefetchable) [size=128K]

        Memory at 380028000000 (64-bit, prefetchable) [size=128M]

        Memory at 380038040000 (64-bit, prefetchable) [size=64K]

        Capabilities: <access denied>

        Kernel driver in use: xocl

        Kernel modules: xocl
```

b) Extract tar.gz files – deployment packages

```bash
tar -xzvf xilinx-vck5000-es1-gen3x16-platform-2-1.noarch.rpm.tar.gz xilinx-sc-fw-vck5000-4.4.6-2.e1f5e26.noarch.rpm xilinx-vck5000-es1-gen3x16-base-2-3123623.noarch.rpm xilinx-vck5000-es1-gen3x16-validate-2-3123623.noarch.rpm
```

c) Install deployment packages in order

```bash
sudo yum install xilinx-sc-fw-vck5000-4.4.6-2.e1f5e26.noarch.rpm
sudo yum install xilinx-vck5000-es1-gen3x16-validate-2-3123623.noarch.rpm
sudo yum install xilinx-vck5000-es1-gen3x16-base-2-3123623.noarch.rpm
```

d) Program VCK5000 card

Enter the following command: 

```bash
sudo /opt/xilinx/xrt/bin/xbmgmt flash --scan
```

An output similar to the following example is seen. 

```bash
Card [0000:02:00.0]

    Card type:         vck5000-es1

    Flash type:        OSPI_VERSAL

    Flashable partition running on FPGA:

        xilinx_vck5000-es1_g3x16_201921_1,[ID=0x5e51824d],[SC=4.4]

    Flashable partitions installed in system:

        xilinx_vck5000-es1_gen3x16_base_2,[ID=0xb376430f2629b15d],[SC=4.4.6]

sudo /opt/xilinx/xrt/bin/xbmgmt flash --update

Card [0000:02:00.0]:

         Status: SC needs updating

         Current SC: 4.4

         SC to be flashed: 4.4.6

         Status: shell needs updating

         Current shell: xilinx_vck5000-es1_g3x16_201921_1

         Shell to be flashed: xilinx_vck5000-es1_gen3x16_base_2

Are you sure you wish to proceed? [y/n]: y

Updating SC firmware on card[0000:02:00.0]

Stopping user function...

....................................

Updating shell on card[0000:02:00.0]

PDI dsabin supports only primary bitstream: /opt/xilinx/firmware/vck5000-es1/gen3x16/base/partition.xsabin

INFO: ***PDI has 29148592 bytes

Successfully flashed Card[0000:02:00.0]

1 Card(s) flashed successfully.

Cold reboot machine to load the new image on card(s).
```

If XRT utility xbmgmt is not functional to install and update platform on VCK5000 ES1 card, [VCK5000 ES1 Flash Programming With Vivado HW Manager](https://www.xilinx.com/member/vck5000-aie/VCK5000%20ES1%20Flash%20Programming%20With%20Vivado%20HW%20Manager.pdf) describes the alternative way to program/revert the flash on VCK5000 ES1 card to factory image by using Vivado via JTAG. After the card has been programmed to factory image, you may still need to upgrade the platform to latest version through XRT utility xbmgmt.

The two .pdi files below are needed for this method as described in [VCK5000 ES1 Flash Programming With Vivado HW Manager](https://www.xilinx.com/member/vck5000-aie/VCK5000%20ES1%20Flash%20Programming%20With%20Vivado%20HW%20Manager.pdf)

[vck5k_ospi.pdi](https://www.xilinx.com/bin/public/openDownload?filename=vck5k_ospi.pdi)

[xilinx_vck5000-es1_g3x16_201921_1.pdi](https://www.xilinx.com/bin/public/openDownload?filename=xilinx_vck5000-es1_g3x16_201921_1.pdi)

e) Cold reboot computer then verify card has been successfully programmed with BASE platform

Enter the following command: 

```bash
sudo /opt/xilinx/xrt/bin/xbmgmt flash --scan
```

An output similar to the following example is seen. 

```bash
Card [0000:02:00.0]

    Card type:         vck5000-es1

    Flash type:        OSPI_VERSAL

    Flashable partition running on FPGA:

        xilinx_vck5000-es1_gen3x16_base_2,[ID=0xb376430f2629b15d],[SC=4.4.6]

    Flashable partitions installed in system:

        xilinx_vck5000-es1_gen3x16_base_2,[ID=0xb376430f2629b15d],[SC=4.4.6]
 ```

f) Verify and validate VCK5000 card

Note: This is a 1RP platform, so no need to install a partition.

Enter the following command: 

```bash
/opt/xilinx/xrt/bin/xbutil validate
```

An output similar to the following example is seen. 

```bash
INFO: Found 1 cards

INFO: Validating card[0]: xilinx_vck5000-es1_gen3x16_base_2

INFO: == Starting AUX power connector check:

AUX power connector not available. Skipping validation

INFO: == AUX power connector check SKIPPED

INFO: == Starting Power warning check:

INFO: == Power warning check PASSED

INFO: == Starting PCIE link check:

INFO: == PCIE link check PASSED

INFO: == Starting SC firmware version check:

INFO: == SC firmware version check PASSED

INFO: == Starting verify kernel test:

INFO: == verify kernel test PASSED

INFO: == Starting IOPS test:

.Maximum IOPS: 91255

INFO: == IOPS test PASSED

INFO: == Starting DMA test:

Host -> PCIe -> FPGA write bandwidth = 9140.510806 MB/s

Host <- PCIe <- FPGA read bandwidth = 8533.351111 MB/s

INFO: == DMA test PASSED

INFO: == Starting device memory bandwidth test:

............

Maximum throughput: 49676 MB/s

INFO: == device memory bandwidth test PASSED

INFO: == Starting PCIE peer-to-peer test:

P2P BAR is not enabled. Skipping validation

INFO: == PCIE peer-to-peer test SKIPPED

INFO: == Starting memory-to-memory DMA test:

M2M is not available. Skipping validation

INFO: == memory-to-memory DMA test SKIPPED

INFO: == Starting host memory bandwidth test:

Host_mem is not available. Skipping validation

INFO: == host memory bandwidth test SKIPPED

INFO: Card[0] validated successfully.

INFO: All cards validated successfully.
```
## 3. Environment Variable Setup in Docker Container

Suppose you have downloaded Vitis-AI, entered Vitis-AI directory, and then started Docker. In the docker container, execute the following steps. You can use the following command to set environment variables. It should be noted that the xclbin file should be in the /opt/xilinx/overlaybins directory.

~~~
cd /workspace/setup/vck5000
source ./setup.sh
~~~

