## 1. VCK5000 Card Setup in Host

We provide some scripts to help to automatically finish the Alveo card setup process. You could refer to these to understand the required steps. To use the scripts, just input the command below. It will detect Operating System you are using, then download and install the appropriate packages.

:pushpin: **Note:** 
* You should use this script in host environment, namely out of the Docker container.
* After the script is executed successfully, manually reboot the host server once.

~~~
source ./install.sh
~~~

The following installation steps were performed in this script.

* Install XRT.
* Install XRM. The [Xilinx Resource Manager (XRM)](https://github.com/Xilinx/XRM/) manages and controls FPGA resources on a machine. It is used by the Alveo runtime.
* Install the VCK5000-PROD Card Target Platform.
* Install DPU V4E xclbin for VCK5000-PROD.

:pushpin: **Note:** This version requires the use of a VCK5000-PROD card. VCK5000-ES1 card is no longer updated, if you want to use it, please refer to [Vitis AI 1.4.1](https://github.com/Xilinx/Vitis-AI/tree/v1.4.1).

## 2. Environment Variable Setup in Docker Container

Suppose you have downloaded Vitis-AI, entered Vitis-AI directory, and then started Docker. In the docker container, execute the following steps. You can use the following command to set environment variables. It should be noted that the xclbin file should be in the /opt/xilinx/overlaybins directory.

```
# For VCK5000-PROD card, you can select 8PE 350MHz  DPU IP via the following command.
source /workspace/setup/vck5000/setup.sh DPUCVDX8H

# For VCK5000-PROD card, you can also select 6PE-DWC 350MHz  DPU IP via the following command.
source /workspace/setup/vck5000/setup.sh DPUCVDX8H-DWC
```
