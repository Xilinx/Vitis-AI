# DPUv3E on U50 with Gen3x4 Platform Usage Guide
DPUv3E on U50 with Gen3x4 Platform is in Early-Access stage now, you could follow the instruction below to try it. It will bring 14% higher performance than the default Gen3x16 Platform version. 

## 1. Card preparation
This step will flash the Alveo U50 card with gen3x4 Deployment Platform. Now **Redhat/Centos7.4-7.6** is supported for this step. Ubuntu 16.04/18.04 support is coming soon.

Firstly please ensure latest version XRT has been installed correctly. Then use following command to check the version of Depolyment shell installed on host and card:
~~~
/opt/xilinx/xrt/bin/xbutil scan
~~~
You could see the running log similar to following:
~~~
INFO: Found total 1 card(s), 1 are usable
-------------------------------------------------------------------------------
System Configuration
OS name:        Linux
Release:        4.15.0-91-generic
Version:        #92-Ubuntu SMP Fri Feb 28 11:09:48 UTC 2020
Machine:        x86_64
Model:          PowerEdge R740xd
CPU cores:      96
Memory:         385419 MB
Glibc:          2.27
Distribution:   Ubuntu 18.04.1 LTS
Now:            Tue May  5 22:42:01 2020
-------------------------------------------------------------------------------
XRT Information
Version:        2.5.309
Git Hash:       9a03790c11f066a5597b133db737cf4683ad84c8
Git Branch:     2019.2_PU2
Build Date:     2020-02-23 18:52:05
XOCL:           2.5.309,9a03790c11f066a5597b133db737cf4683ad84c8
XCLMGMT:        2.5.309,9a03790c11f066a5597b133db737cf4683ad84c8
-------------------------------------------------------------------------------
 [0] 0000:af:00.1 xilinx_u50_gen3x4_xdma_201920_3 user(inst=129)
~~~

If the last line of platform information is not **xilinx_u50_gen3x4_xdma_201920_3**, please run script **card_setup.sh** from host to finish the installation of gen3x4 platform and card flashing. You will need sudo privilege to finish this step.
~~~
./card_setup.sh
~~~
After the scripts execution finished, please cold reboot the machine (power off and power on the machine again) and go on to next step.

## 2. Using DPUv3E gen3x4 version in VART and Vitis AI Library
Now you can follow the instruction in [Vitis-AI-Library](../Vitis-AI-Library/README.md) or [VART](../VART/README.md) to use DPUv3E. Before that, please note you should use following script to replace the relevant steps for U50_xclbin download and installation.
~~~
./xclbin_setup.sh
~~~
### For Vitis-AI-Library
The step 1 in [Quick Start For Alveo](../Vitis-AI-Library/README.md#quick-start-for-alveo) should be replaced.
### For VART
The step 1 in [Quick Start For Alveo](../VART/README.md#quick-start-for-alveo) should be replaced.

