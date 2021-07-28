# Setup Alveo Accelerator Card

Xilinx DPU IP family for convolution neural network (CNN) inference application supports Alveo accelerator cards with HBM now, including Alveo U50, U50LV, U200, U250 and U280 cards. The Xilinx DPUs for U50, U50LV and U280 card include ***DPUCAHX8H*** and ***DPUCAHX8L***, the former is for optimized for high throughput and the latter is optimized for MobileNet model and ultra low latency. The Xilinx DPUs for U200 and U250 card is a High Performance CNN processing engine ***DPUCADF8H***.

Following section will guide you through the Alveo card preparation steps for Vitis AI.

## Alveo Card Setup in Host

We provide some scripts to help to automatically finish the Alveo card setup process. You could refer to these to understand the required steps. To use the scripts, just input the command below. It will detect the cards type (U50, U50LV, U200, U250 or U280) and Operating System you are using, then download and install the appropriate packages.

**Please note you should use this script in host environment, namely out of the Docker container.** 

~~~
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
Take vitis-ai-caffe and DPUCAHX8H as examples, you can use the following command to set environment variables. It should be noted that the xclbin file should be in the /opt/xilinx/overlaybins directory.

```
cd /workspace/setup/alveo
source setup.sh DPUCAHX8H
```

### Multi-card selection (optional)
If there are more than one supported cards installed on the host, all available cards will be used for DPU applications by default. You can use following command to check the installed Alveo card on the host:

~~~
/opt/xilinx/xrt/bin/xbutil scan
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
program the FPGA and scale down the frequency. These commands will set the Clock_1 to 80
MHz and can be run at host or in the docker.

```
/opt/xilinx/xrt/bin/xbutil reset -d 0
/opt/xilinx/xrt/bin/xbutil program -p <xclbin path>
/opt/xilinx/xrt/bin/xbutil clock -d0 -g 80
```
d0 is the Alveo card device number. For more information about xbutil tool, see the XRT documents.  
&lt;xclbin path&gt; is the full path of the corresponding xclbin file, usually from /opt/xilinx/overlaybins.
