# Setup Versal Accelerator Card

The Xilinx DPUs for VCK5000 card is a High Performance CNN processing engine ***DPUCVDX8H***.  The detailed combination of Alveo card and DPU IP is shown in the table below, you can choose one of them according to your own situation.

| No\. | Accelerator Card | DPU IP |
| ---- | ---- | ----   |
| 1 | VCK5000-PROD | DPUCVDX8H_4pe_miscdwc     |
| 2 | VCK5000-PROD | DPUCVDX8H_6pe_dwc  |
| 3 | VCK5000-PROD | DPUCVDX8H_6pe_misc |
| 4 | VCK5000-PROD | DPUCVDX8H_8pe_normal     |

## 1. VCK5000 Card Setup in Host

We provide some scripts to help to automatically finish the VCK5000-PROD card setup process. You could refer to these to understand the required steps. To use the scripts, just input the command below. It will detect Operating System you are using, then download and install the appropriate packages. Suppose you have downloaded Vitis-AI, entered Vitis-AI directory.

:pushpin: **Note:** 
* You should use this script in host environment, namely out of the Docker container.
* After the script is executed successfully, manually reboot the host server once.
* For cloud DPU, Vitis AI 2.5 applies 2021.2 Tools/Platform/XRT/XRM.

~~~
cd ./setup/vck5000/
source ./install.sh
~~~

The following installation steps were performed in this script.

* Install XRT.
* Install XRM. The [Xilinx Resource Manager (XRM)](https://github.com/Xilinx/XRM/) manages and controls FPGA resources on a machine. It is used by the runtime.
* Install the VCK5000-PROD Card Target Platform.
* Install DPU V4E xclbin for VCK5000-PROD.

:pushpin: **Note:** This version requires the use of a VCK5000-PROD card. VCK5000-ES1 card is no longer updated since Vitis AI 2.0, if you want to use it, please refer to [Vitis AI 1.4.1](https://github.com/Xilinx/Vitis-AI/tree/v1.4.1).

## 2. Environment Variable Setup in Docker Container

Suppose you have downloaded Vitis-AI, entered Vitis-AI directory, and then started Docker. In the docker container, execute the following steps. You can use the following command to set environment variables. It should be noted that the xclbin file should be in the /opt/xilinx/overlaybins directory. There are 4 xclbins to choose from depending on the parameters you use.

1. For 4PE 350Hz, you can select DPU IP via the following command.
    ```
    source /workspace/setup/vck5000/setup.sh DPUCVDX8H_4pe_miscdwc
    ```

2. For 6PE 350Hz with DWC, you can select DPU IP via the following command.
    ```
    source /workspace/setup/vck5000/setup.sh DPUCVDX8H_6pe_dwc
    ```

3. For 6PE 350Hz with MISC, you can select DPU IP via the following command.
    ```
    source /workspace/setup/vck5000/setup.sh DPUCVDX8H_6PE_MISC
    ```

2. For 8PE 350Hz, you can select DPU IP via the following command.
    ```
    source /workspace/setup/vck5000/setup.sh DPUCVDX8H_8pe_normal
    ```
