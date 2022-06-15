# Overview
This package contains BERT demo(based on BERT IP) and ViT demo(based on ViT IP).

# System requirement  
To run the demos, make sure that the following requirements are safisfied and you have successfully set up machine's enviroment.

## Hardware requirement
- Accelerator Card: Versal VCK5000 Prod

## Software requirement  
- OS: CentOS 7.8
- General Development Tools   
  - cmake:  > 3.1, 
  - g++:	support c++11
- Xilinx Platform and Runtime

# VCK5000 Card Setup in Host
We provide some scripts to help to automatically finish the VCK5000-PROD card setup process. You could refer to these to understand the required steps. To use the scripts, just input the command below. It will detect Operating System you are using, then download and install the appropriate packages. 

:pushpin: **Note:**
* You should use this script in host environment, namely out of the Docker container.
* After the script is executed successfully, manually reboot the host server once.
* This version requires the use of a VCK5000-PROD card.

## Install XRT:
```
cd ../../setup/alveo/scripts/install_xrt.sh
source ./install_xrt.sh
```

## Install the VCK5000-PROD Card Target Platform:
```
cd ../../setup/vck5000/scripts/
source ./install_vck5000_shell.sh
```
  
:pushpin: **Note:** You need to set up XRT enviroment with the following command before using it:
```
    source /opt/xilinx/xrt/setup.sh
```

# Run demos
After VCK5000 Platform and XRT environment are setup correctly, please refer to the instructions in BERT or ViT folder's README.md to launch demo.

- [BERT_demo](https://www.xilinx.com/bin/public/openDownload?filename=VCK5000_BERT_Demo_2.5.0.tar.gz)
- [VIT_demo](https://www.xilinx.com/bin/public/openDownload?filename=VCK5000_ViT_Demo_2.5.0.tar.gz)

