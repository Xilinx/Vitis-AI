# Setup Alveo Accelerator Card with HBM for DPUCAHX8H/L

Xilinx DPU IP family for convolution neural network (CNN) inference application supports Alveo accelerator cards with HBM now, including Alveo U50, U50LV and U280 cards. The Xilinx DPUs for Alveo-HBM card include ***DPUCAHX8H*** and ***DPUCAHX8L***, the former is for optimized for high throughput and the latter is optimized for MobileNet model and ultra low latency. The on-premise DPUCAHX8H and DPUCAHX8L overlays are released along with Vitis AI, and a few of variants are provided.

Following section will guide you through the Alveo-HBM card preparation steps for Vitis AI.

## Alveo Card Setup

We provide some scripts to help to automatically finish the Alveo card setup process. You could refer to these to understand the required steps. To use the scripts, just input the command below. It will detect the cards type (U50, U50LV or U280) and Operating System you are using, then download and install the appropriate packages.

**Please note you should use this script in host environment, namely out of the Docker container.** 

~~~
source ./install.sh
~~~

<details>
 <summary><b>Advanced - Step by Step XRT/Platform Setup</b></summary>

If you don't use the script above, you could follow following steps to finish the Alveo card setup.

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
[xilinx-u50-gen3x4-xdma-platform-2-1.noarch.rpm.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=xilinx-u50-gen3x4-xdma-platform-2-1.noarch.rpm.tar.gz)

Ubuntu 16.04/18.04/20.04:
[xilinx-u50-gen3x4-xdma-platform_2-1_all.deb.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=xilinx-u50-gen3x4-xdma-platform_2-1_all.deb.tar.gz)

#### Alveo U50LV Card

For U50LV card, gen3x4 version target platform is used. Please download and install the required gen3x4 target platform files.

CentOS/Redhat:
[xilinx-u50lv-gen3x4-xdma-platform-2-1.noarch.rpm.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=xilinx-u50lv-gen3x4-xdma-platform-2-1.noarch.rpm.tar.gz)

Ubuntu 16.04/18.04/20.04:
[xilinx-u50lv-gen3x4-xdma-platform_2-1_all.deb.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=xilinx-u50lv-gen3x4-xdma-platform_2-1_all.deb.tar.gz)

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

</details>

***

**Notes**: if there are more than one supported cards installed on the host, all available cards will be used for DPU applications by default. You can use following command to check the installed Alveo card on the host:

~~~
/opt/xilinx/xrt/bin/xbutil list
~~~

If you want to specify whith cards to use for Vitis AI, you can use the environment variable *XLNX_ENABLE_DEVICES* in the **Docker container**. The example usage is like below:

* To use card 0 for the DPU, use following setting in docker: 

~~~
export XLNX_ENABLE_DEVICES=0.
~~~

* To use card 0, card 1, and card 2 for the DPU, use following setting in docker:

~~~
export XLNX_ENABLE_DEVICES=0,1,2
~~~
