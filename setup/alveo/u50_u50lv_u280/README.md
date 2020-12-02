# Setup Alveo Accelerator Card with HBM for DPUCAHX8H/L

Xilinx DPU IP family for convolution nerual network (CNN) inference application supports Alveo accelerator cards with HBM now, including Alveo U50, U50LV and U280 cards. The Xilinx DPUs for Alveo-HBM card include ***DPUCAHX8H*** and ***DPUCAHX8L***, the former is for optimized for high throughput and the latter is optimized for MobileNet model and ultra low latency. The on-premise DPUCAHX8H and DPUCAHX8L overlays are released along with Vitis AI, and a few of variants are provided.

Following section will guide you through the Alveo-HBM card preparation steps and on-premise overlays setup flow for Vitis AI.

## Alveo Card and Overlays Setup

We provide some scripts to help to automatically finish the Alveo card and overlay files setup process. You could refer to these to understand the required steps. To use the scripts, just input the command below. It will detect the cards type (U50, U50LV or U280) and Operating System you are using, then download and install the appropriate packages.

**Please note you should use this script in host environment, namely out of the Docker container.** 

~~~
source ./install.sh
~~~

<details>
 <summary><b>Advanced - Step by Step XRT/Platform Setup</b></summary>

If you don't use the script above, you could follow following steps to finish the Alveo card and overlays setup.

**Please note you should use this script in host environment, namely out of the Docker container.** 

### Install XRT

Before you go through the next steps, please ensure the latest Xilinx runtime (XRT) is installed on your host, you can get XRT from these links:

CentOS/Redhat 7.x: [xrt_202020.2.8.726_7.4.1708-x86_64-xrt.rpm](https://www.xilinx.com/bin/public/openDownload?filename=xrt_202020.2.8.726_7.4.1708-x86_64-xrt.rpm)

CentOS/Redhat 8.x: [xrt_202020.2.8.726_8.1.1911-x86_64-xrt.rpm](https://www.xilinx.com/bin/public/openDownload?filename=xrt_202020.2.8.726_8.1.1911-x86_64-xrt.rpm)

Ubuntu 16.04: [xrt_202020.2.8.726_16.04-amd64-xrt.deb](https://www.xilinx.com/bin/public/openDownload?filename=xrt_202020.2.8.726_16.04-amd64-xrt.deb)

Ubuntu 18.04: [xrt_202020.2.8.726_18.04-amd64-xrt.deb](https://www.xilinx.com/bin/public/openDownload?filename=xrt_202020.2.8.726_18.04-amd64-xrt.deb)

Ubuntu 20.04: [xrt_202020.2.8.726_20.04-amd64-xrt.deb](https://www.xilinx.com/bin/public/openDownload?filename=xrt_202020.2.8.726_20.04-amd64-xrt.deb)

### Install the Alveo Card Target Platform

#### Alveo U280 Card
For U280 card, gen3x16 target platform released in the Xilinx website [U280 page](https://www.xilinx.com/products/boards-and-kits/alveo/u280) is used. Please download and install the required gen3x4 target platform files.

CentOS/Redhat:
[xilinx-u280-xdma-201920.3-2789161.x86_64.rpm](https://www.xilinx.com/bin/public/openDownload?filename=xilinx-u280-xdma-201920.3-2789161.x86_64.rpm)

Ubuntu 16.04:
[xilinx-u280-xdma-201920.3-2789161_16.04.deb](https://www.xilinx.com/bin/public/openDownload?filename=xilinx-u280-xdma-201920.3-2789161_16.04.deb)

Ubuntu 18.04:
[xilinx-u280-xdma-201920.3-2789161_18.04.deb](https://www.xilinx.com/bin/public/openDownload?filename=xilinx-u280-xdma-201920.3-2789161_18.04.deb)

#### Alveo U50 Card
For U50 card, gen3x4 version target platform is used. Please download and install the required gen3x4 target platform files.

CentOS/Redhat:
[Xilinx-u50-gen3x4-xdma-2-202010.1_2902115_noarch_rpm.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=Xilinx-u50-gen3x4-xdma-2-202010.1_2902115_noarch_rpm.tar.gz)

Ubuntu 16.04:
[Xilinx-u50-gen3x4-xdma-2-202010.1_2902115_16.04_deb.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=Xilinx-u50-gen3x4-xdma-2-202010.1_2902115_16.04_deb.tar.gz)

Ubuntu 18.04:
[Xilinx-u50-gen3x4-xdma-2-202010.1_2902115_18.04_deb.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=Xilinx-u50-gen3x4-xdma-2-202010.1_2902115_18.04_deb.tar.gz)


#### Alveo U50LV Card

For U50LV card, gen3x4 version target platform is used. Please download and install the required gen3x4 target platform files.

CentOS/Redhat:
[Xilinx-u50lv-gen3x4-xdma-2-202010.1-2902115-noarch_rpm.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=Xilinx-u50lv-gen3x4-xdma-2-202010.1-2902115-noarch_rpm.tar.gz)

Ubuntu 16.04:
[Xilinx-u50lv-gen3x4-xdma-2-202010.1-2902115-16.04_deb.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=Xilinx-u50lv-gen3x4-xdma-2-202010.1-2902115-16.04_deb.tar.gz)

Ubuntu 18.04:
[Xilinx-u50lv-gen3x4-xdma-2-202010.1-2902115-18.04_deb.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=Xilinx-u50lv-gen3x4-xdma-2-202010.1-2902115-18.04_deb.tar.gz)


### Update the Alveo Card Flash
After you have downloaded and installed the platform files above, use following commands and cold reboot your machine to finished the setup.

For Alveo U280:
~~~
sudo /opt/xilinx/xrt/bin/xbmgmt flash --update --shell xilinx_u280_xdma_201920_3
~~~

For Alveo U50:
~~~
sudo /opt/xilinx/xrt/bin/xbmgmt flash --update --shell xilinx_u50_gen3x4_xdma_base_2
~~~

For Alveo U50LV:
~~~
sudo /opt/xilinx/xrt/bin/xbmgmt flash --update --shell xilinx_u50lv_gen3x4_xdma_base_2
~~~

### DPUCAHX8H/L Overlays Installation
#### Get and Decompress Overlays Tarball
In the host or docker, get to the shared Vitis AI git repository directory and use following commands to download and decompress the overlays tarball.

~~~
cd ./Vitis-AI/setup/alveo/u50_u50lv_u280
wget https://www.xilinx.com/bin/public/openDownload?filename=alveo_xclbin-1.3.tar.gz -O alveo_xclbin-1.3.tar.gz
tar xfz alveo_xclbin-1.3.tar.gz
~~~

</details>

---
## DPUCAHX8H/L Overlay Usage

Four kinds of DPUCAHX8H overlays are provided for different Alveo HBM card:
* U50-6E300M: two kernels, six engines, maximal core clock 300MHz
* U50LV-9E275M: two kernels, nine engines, maximal core clock 275MHz
* U50LV-10E275M: two kernels, ten engines, maximal core clock 275MHz
* U280-14E300M: three kernels, fourteen engines, maximal core clock 300MHz

Two kinds of DPUCAHX8L overlays are provided for different Alveo HBM card:
* U50LV-1C250M: one kernel, maximal core clock 250MHz
* U280-2C300M: two kernels, maximal core clock 300MHz

The DPUCAHX8H/L overlays should be used in the **docker contaniner** environment.

Firstly start the CPU or GPU docker, then run the script below to automatically copy the overlays into the correct location. The script will automatically detect the card type and finish the overlay file copy. By default for DPUCAHX8H the 10E275M version is used for U50LV card and you could modify the script to use alternative version.

~~~
cd /workspace/setup/alveo/u50_u50lv_u280
source ./overlay_settle.sh
~~~


<details>
 <summary><b>Advanced - Overlay Selection</b></summary>

###  Copy Overlay Files
Start the CPU or GPU docker, get into the shared Vitis AI git repository directory and use following command to copy the overlay files for different Alveo card. Please note everytime you start a new docker container, you should do this step.

For Alveo U50, use DPUCAHX8H overlay:
~~~
cd /workspace/setup/alveo/u50_u50lv_u280
sudo cp alveo_xclbin-1.3/U50/6E300M/* /usr/lib
~~~

For Alveo U50LV, use DPUCAHX8H U50LV-9E275M overlay:
~~~
cd /workspace/setup/alveo/u50_u50lv_u280
sudo cp alveo_xclbin-1.3//U50lv/9E275M/* /usr/lib
~~~

For Alveo U50LV, use DPUCAHX8H U50LV-10E275M overlay:
~~~
cd /workspace/setup/alveo/u50_u50lv_u280
sudo cp alveo_xclbin-1.3//U50lv/10E275M/* /usr/lib
~~~

For Alveo U280, use DPUCAHX8H overlay:
~~~ 
cd /workspace/setup/alveo/u50_u50lv_u280
sudo cp alveo_xclbin-1.3/U280/14E300M/* /usr/lib
~~~

For Alveo U50LV, use DPUCAHX8L overlay:
~~~ 
cd /workspace/setup/alveo/u50_u50lv_u280
sudo cp alveo_xclbin-1.3/U50lv-V3ME/1E300M/* /usr/lib
~~~

For Alveo U280, use DPUCAHX8L overlay:
~~~ 
cd /workspace/setup/alveo/u50_u50lv_u280
sudo cp alveo_xclbin-1.3/U280-V3ME/2E300M/* /usr/lib
~~~

</details>

**Note:** once you finish copying the overlay files, if you don't need to switch to other overlays in future, you could use below command in **host** to freeze the change you have made to docker container, then you don't need to copy the overlay files again. Please refer to the docker documents of command help for more information.

~~~
docker commit [OPTIONS] CONTAINER [REPOSITORY[:TAG]]
~~~


### DPUCAHX8H Overlay Frequency Scaling Down

The maximal core clock frequency listed in this section is the timing sign-off frequency of each overlays, and the overlays run at their maximal core clock by default. However, because of the power limitation of the card, all CNN models on each Alveo card cannot run at all the maximal frequencies listed here. Sometimes frequency scaling-down operation is necessary. For the safe working frequency on each card for the CNN models and corresponding performance, please refer to Chapter 7 of *Vitis AI Library User Guide* (ug1354). **Higher overlay frequencies then the recommendation in ug1354 could cause system reboot or other damage to your system because of the power consumption exceeding of Alveo card over the PCIe power supply limitation.**

The DPUCAHX8H core clock is generated from an internal DCM module driven by the platform Clock_1 with the default value of 100MHz, and the core clock is always linearly proportional to Clock_1. For example, in U50LV-10E275M overlay, the 275MHz core clock is driven by 100MHz clock source. So to set the core clock of this overlay to 220MHz, we need to set the frequency of Clock_1 to (220/275)*100 = 80MHz.

You could use XRT xbutil tools to scale down the running frequency of the DPUCAHX8H overlay before you run the VART/Library examples. Before the frequency scaling-down operation, the overlays should have been programmed into the FPGA first, please refer to the example commands below to program the FPGA and scale down the frequency. These commands will set the Clock_1 to 80MHz and could be run at host or in the docker.

~~~
/opt/xilinx/xrt/bin/xbutil program -p /usr/lib/dpu.xclbin
/opt/xilinx/xrt/bin/xbutil clock -d0 -g 80
~~~
d0 is the Alveo card device number. For more information about **xbutil** tool, please use refer to XRT documents.
