# Setup Alveo Accelerator Card

Xilinx DPU IP family for convolution neural network (CNN) inference application supports Alveo accelerator cards with HBM now, including Alveo U50LV, U55C, U200 and U250 cards. The Xilinx DPUs for U50LV and U55C card is ***DPUCAHX8H***, the former is for optimized for high throughput and the latter is optimized for MobileNet model and ultra low latency. The Xilinx DPUs for U200 and U250 card is a High Performance CNN processing engine ***DPUCADF8H***. The detailed combination of Alveo card and DPU IP is shown in the table below, you can choose one of them according to your own situation.

| No\. | Accelerator Card | DPU IP |
| ---- | ---- | ----   |
| 1 | U50LV        | DPUCAHX8H         |
| 2 | U50LV        | DPUCAHX8H-DWC     |
| 3 | U55C         | DPUCAHX8H-DWC     |
| 4 | U200         | DPUCADF8H         |
| 5 | U250         | DPUCADF8H         |

:pushpin: **Note:** For DPU DPUCAHX8L IP, Alveo U50 card, and Alveo U280 card, they are no longer updated since Vitis AI 2.0. If you want to use them, please refer to [Vitis AI 1.4.1](https://github.com/Xilinx/Vitis-AI/tree/v1.4.1).

Following section will guide you through the Alveo card preparation steps for Vitis AI.

## Alveo Card Setup in Host

We provide some scripts to help to automatically finish the Alveo card setup process. You could refer to these to understand the required steps. To use the scripts, just input the command below. It will detect the cards type (U50LV, U55C, U200 or U250) and Operating System you are using, then download and install the appropriate packages. Suppose you have downloaded Vitis-AI, entered Vitis-AI directory.

:pushpin: **Note:** 
* You should use this script in host environment, namely out of the Docker container.
* After the script is executed successfully, manually reboot the host server once.
* For cloud DPU, Vitis AI 2.5 applies 2021.2 Tools/Platform/XRT/XRM.

~~~
cd ./setup/alveo/
source ./install.sh
~~~

The following installation steps were performed in this script.

* Install XRT.
* Install XRM. The [Xilinx Resource Manager (XRM)](https://github.com/Xilinx/XRM/) manages and controls FPGA resources on a machine. It is used by the Alveo runtime.
* Install the Alveo Card Target Platform.
* Update the Alveo Card Flash.

Please note the Alveo U250 shell is a DFX-2RP platform. It means a base shell is loaded from flash. However, the user must dynamically load an intermediate shell, after every cold boot. Please see AR: https://www.xilinx.com/support/answers/75975.html. The following command is an example: 

```
sudo /opt/xilinx/xrt/bin/xbmgmt partition --program --name xilinx_u250_gen3x16_xdma_shell_3_1 --card 0000:03:00.0
```
******

## Environment Variable Setup in Docker Container

Suppose you have downloaded Vitis-AI, entered Vitis-AI directory, and then started Docker. In the docker container, execute the following steps.

### DPU IP selection
You can use the following command to set environment variables. It should be noted that the xclbin file should be in the /opt/xilinx/overlaybins directory. There are 4 xclbins to choose from depending on the card and parameters you use.

1. For U200 or U250 card, you can select DPU IP via the following command.
    ```
    source /workspace/setup/alveo/setup.sh DPUCADF8H
    ```

2. For U50LV card, you can select 10PE 275MHz or 8PE 275MHz with DWC DPU IP.
    * For 10PE 275MHz DPU IP, you can select the DPU IP via the following command.
    ```
    source /workspace/setup/alveo/setup.sh DPUCAHX8H
    ```
    * For 8PE-DWC 275MHz DPU IP, you can select the DPU IP via the following command.
    ```
    source /workspace/setup/alveo/setup.sh DPUCAHX8H-DWC
    ```

3. For U55C card, you can select 11PE 300MHz with DWC DPU IP via the following command.
    ```
    source /workspace/setup/alveo/setup.sh DPUCAHX8H-DWC
    ```

### Multi-card selection (optional)
If there are more than one supported cards installed on the host, all available cards will be used for DPU applications by default. You can use following command to check the installed Alveo card on the host:

~~~
/opt/xilinx/xrt/bin/xbutil examine
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

### Scaling down the frequency of the DPU (optional)

Due to the power limitation of the card, in some cases CNN models cannot run at the highest frequencies on Alveo card.
Sometimes frequency scaling-down operation is necessary.

You could use XRT xbutil tools to scale down tihe running frequency of the DPU overlay before
you run the VART/Library examples. Before the frequency scaling-down operation, the corresponding overlays
should be programmed into the FPGA first. Refer to the following example commands to
program the FPGA and scale down the frequency. These commands will set the clock Clock_1 to 70 MHz and can be run at host or in the docker.

```
/opt/xilinx/xrt/bin/xbutil reset -d <user bdf>
/opt/xilinx/xrt/bin/xbutil program -d <user bdf> -u <xclbin path>
/opt/xilinx/xrt/bin/xbutil --legacy clock -d <user bdf> -g 70
```

For more information about xbutil tool, see the [XRT documents](https://xilinx.github.io/XRT/master/html/xbutil.html).
&lt;xclbin path&gt; is the full path of the corresponding xclbin file, usually from /opt/xilinx/overlaybins.
