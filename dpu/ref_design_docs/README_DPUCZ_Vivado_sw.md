<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>

# Zynq UltraScale＋ MPSoC DPU v4.1 Vivado Flow Reference Design for Petalinux 2022.2


- [1 Overview](#1-overview)
- [2 Petalinux Building and System Customization](#2-petalinux-building-and-system-customization)
    - [2.1 Create a Petalinux Project](#21-create-a-petalinux-project)
        - [2.1.1 Setup petalinux workspace environment](#211-setup-petalinux-workspace-environment)
        - [2.1.2 Create and configure a project](#212-create-and-configure-a-project)
    - [2.2 Customize RootFS, Kernel, Device Tree and U-boot](#22-customize-rootfs-kernel-device-tree-and-u-boot)
        - [2.2.1 Enable DPU driver for vivado flow](#221-must-enable-dpu-driver-for-vivado-flow)
        - [2.2.2 Customize user packages](#222-optional-customize-user-packages)
        - [2.2.3 Enable Package Management and auto login](#223-optional-enable-package-management-and-auto-login)
	    - [2.2.4 Generate EXT4 RootFS](#224-generate-ext4-rootfs)
    - [2.3 Build project and Generate Images for SD Card](#23-build-project-and-generate-images-for-sd-card)
- [3 Run Resnet50 Example and Vitis-AI Examples](#3-run-resnet50-example-and-vitis-ai-examples)
- [4 Konwn Issues](#4-known-issues)
    - [4.1 Continous memory allocator (CMA)](#41-continous-memory-allocatorcma)
    - [4.2 EXT4 Partition Resize](#42-ext4-partition-resize)
    - [4.3 DDR Qos](#43-ddr-qos)
    - [4.4 do_image_cipo Failed](#44-do_image_cpio-function-failed-issue)


# 1 Overview
This Tutorial shows how to build Linux image using Petalinux build tools for Vivado flow and customize software system.
The Xilinux Deep learning Processing Unit (DPU) Driver must be enable for Vivado flow. Other customizations are optional. Please feel free to pick your desired customization.

Here lists the recommanded designs:
- install Vitis AI Runtime and Library v3.0 to RootFS
- install and run Vitis AI application examples

# 2 Petalinux Building and System Customization
*Version: Petalinux 2022.2*

This section explains about how to use published BSP [ZCU102 BSP](https://www.xilinx.com/member/forms/download/xef.html?filename=xilinx-zcu102-v2022.2-10141622.bsp) to generation your project.

As XSA file is the mandatory input for petalinux project. Here we use pre-built post-impl $TRD_HOME/prj/Vivado/hw/pre-built/top_wrapper.xsa as default. User can input XSA file exported from your Vivado project under directory $TRD_HOME/prj/Vivado/hw/prj.

For default DPU design, please change your directory to $TRD_HOME/prj/Vivado/sw and run `helper_build_bsp.sh`. The xilinx-zcu102 image is located in $TRD_HOME/prj/Vivado/sw/xilinx-zcu102-trd/images/linux/petalinux-sdimage.wic.gz

**NOTE:** The default settings of DPU is 3 cores B4096 with RAM_USAGE_LOW, CHANNEL_AUGMENTATION_ENABLE, DWCV_ENABLE, POOL_AVG_ENABLE, RELU_LEAKYRELU_RELU6, Softmax.


## 2.1 Create a Petalinux Project
### **2.1.1  Setup petalinux workspace environment**
For PetaLinux working environment setup, please refer to the [PetaLinux Tools Documentation (UG1144)](https://docs.xilinx.com/r/en-US/ug1144-petalinux-tools-reference-guide/Setting-Up-Your-Environment) for installation details.

For BASH shell:
```
$ source <PetaLinux_v2022.2_install_path>/settings.sh
```
For C shell:
```
% source <PetaLinux_v2022.2_install_path>/settings.csh
```

### **2.1.2  Create and configure a project**
```
$ petalinux-create -t project -s xilinx-zcu102-v2022.2-10141622.bsp
$ cd xilinx-zcu102-2022.2
$ petalinux-config --get-hw-description=$TRD_HOME/prj/Vivado/hw/prj/ --silentconfig
```
The pre-built vivado design xsa file path is `$TRD_HOME/prj/Vivado/hw/pre-built`, you can set this to your own xsa file path.

## 2.2 Customize RootFS, Kernel, Device Tree and U-boot

### **2.2.1 (MUST) Enable DPU driver for vivado flow**
**Please copy $TRD_HOME/prj/Vivado/sw/meta-vitis/recipes-kernel to <your_petalinux_project_dir>/project-spec/meta-user for DPU v4.1 Vivado Flow Reference Design with Petalinux 2022.2.**
- Run `petalinux-config -c kernel`
- Select and Enable DPU driver
- Exit and Save

```
    Device Drivers -->
        Misc devices -->
            <*> Xilinux Deep learning Processing Unit (DPU) Driver
```

### **2.2.2 (OPTIONAL) Customize user packages**

#### **1. exclude zocl and xrt packages**
Exclude zocl and xrt from rootfs, which are required for Vitis acceleration flow.
- Launch rootfs config `petalinux-config -c rootfs`
- Disable zocl and xrt packages
- Exit and Save

```
    Filesystem Packages -->
        libs -->
            xrt  -->
                [] xrt
            zocl  -->
                [] zocl
```

#### **2. (OPTIONAL) Add user desired packages**

This step is not a must but it makes it easier to find and select all required packages in next step. If this step is skipped, please enable the required packages one by one in next step.

**Add vitis-ai-library to rootfs**
- Copy the lastest [recipes-vitis-ai](../../src/vai_petalinux_recipes/recipes-vitis-ai) to your <your_petalinux_project_dir>/project-spec/meta-user

    **NOTE**: recipes-vitis-ai is used for Vitis flow by default. In vivado flow, please comment out the following line in recipes-vitis-ai/vart/vart_3.0.bb
    ```
    #PACKAGECONFIG:append = " vitis"
    ```

- Append the CONFIG_x lines below to <your_petalinux_project_dir>/project-spec/meta-user/conf/user-rootfsconfig file.

```
CONFIG_vitis-ai-library
CONFIG_vitis-ai-library-dev
CONFIG_vitis-ai-library-dbg
```
**Recommended Packages for easy system management**
```
CONFIG_dnf
CONFIG_nfs-utils
CONFIG_resize-part
```
- **dnf** is used for package management
- **nfs-utils** is used for kernel NFS server and related tools
- **resize-part** can be used for EXT4 partition resize. See [4.2 EXT4 Partition Resize](#42-ext4-partition-resize) for more information.

**Optional Packages for natively build Vitis AI applications on target board**
```
CONFIG_packagegroup-petalinux-self-hosted
CONFIG_cmake
CONFIG_opencl-clhpp-dev
CONFIG_opencl-headers-dev
CONFIG_packagegroup-petalinux-opencv
CONFIG_packagegroup-petalinux-opencv-dev
```
**Optional Packages for running Vitis-AI demo applications with GUI**
```
CONFIG_packagegroup-petalinux-x11
CONFIG_packagegroup-petalinux-v4lutils
CONFIG_packagegroup-petalinux-matchbox
CONFIG_packagegroup-petalinux-gstreamer
```

#### **3. Enable selected rootfs packages**
- Run `petalinux-config -c rootfs`
- Select User Packages
- Select packages listed above as desired to RootFS
- Exit and Save

```
    user packages -->
        [*] dnf
        [*] nfs-utils
        [*] resize-part
        [*] vitis-ai-library
        [*] cmake
        [*] opencl-clhpp-dev
        [*] opencl-headers-dev
        [*] packagegroup-petalinux-self-hosted
        [*] packagegroup-petalinux-opencv
        [*] packagegroup-petalinux-x11
        [*] packagegroup-petalinux-gstreamer
```



### **2.2.3 (OPTIONAL) Enable Package Management and Auto Login**
Package management feature can allow the board to install and upgrade software packages on the fly.
Select autologin on system bootup for development mode, which is not recommanded in your own production.

- Launch rootfs config `petalinux-config -c rootfs`
- Enable package mangement
- Enable auto-login for development stage.
- Exit and Save.

```
    Image Features -->
        [*] package-management
        ()   package-feed-uris
        () package-feed-archs
        -*- debug-tweaks
        [*] auto-login
```

### **2.2.4 Generate EXT4 RootFS**
It's recommended to use EXT4 for Vitis acceleration designs. PetaLinux uses initramfs format for rootfs by default. It can't retain the rootfs changes in run time. Initramfs keeps rootfs contents in DDR, which makes user useable DDR memory reduced. To make the root file system retain changes and to enable maximum usage of available DDR memory, we'll use EXT4 format for rootfs in second partition while keep the first partition FAT32 to store the boot files.

Vitis-AI applications will install additional software packages. If user would like to run Vitis-AI applications, please use EXT4 rootfs. If in any case initramfs would be used, please add all Vitis-AI dependencies to initramfs.

- Run `petalinux-config`
- Select Root filesystem type as EXT4
- Exit and Save
```
   Image Packaging Configuration --->
       Root filesystem type (EXT4 (SD/eMMC/SATA/USB))
       (/dev/mmcblk0p2) Device node of SD device
```


## 2.3 Build project and Generate Images for SD Card
From any directory within the PetaLinux project, build the PetaLinux project.
```
$ petalinux-build
```
Create a boot image (BOOT.BIN) including FSBL, ATF, bitstream, and u-boot.
```
$ cd images/linux
$ petalinux-package --boot --fsbl zynqmp_fsbl.elf --u-boot u-boot.elf --pmufw pmufw.elf --fpga system.bit --force
```
Generate WIC Image for SD Card
```
$ petalinux-package --wic --bootfile "BOOT.BIN boot.scr Image system.dtb" --wic-extra-args "-c gzip"
```
All related files have been packaged in <your-petalinux-project>/images/linux/petalinux-sdimage.wic.gz. Please use Ether to flash the SD card. Refer section "Flashing the OS Image to the SD Card" in [UG1414](https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/Flashing-the-OS-Image-to-the-SD-Card) for details.

# 3 Run Resnet50 Example and Vitis-AI Examples

Execute resnet50 app on your target board as shwon below:
```
root@xilinx-zcu102-trd:~# cd app/
root@xilinx-zcu102-trd:~/app# cp model/resnet50.xmodel .
root@xilinx-zcu102-trd:~/app# env LD_LIBRARY_PATH=samples/lib samples/bin/resnet50 img/bellpeppe-994958.JPEG
score[945]  =  0.992235     text: bell pepper,
score[941]  =  0.00315807   text: acorn squash,
score[943]  =  0.00191546   text: cucumber, cuke,
score[939]  =  0.000904801  text: zucchini, courgette,
score[949]  =  0.00054879   text: strawberry,
root@xilinx-zcu102-trd:~/app#
```
Note: The TRD project has generated the default matching model file in $TRD_HOME/prj/Vivado/sw/meta-vitis/recipes-apps/resnet50/files/app.tar.gz. If users change the DPU settings, the model need to be created again. For other network running, please refer to [Run the Vitis AI Examples](https://github.com/Xilinx/Vitis-AI/tree/master/examples/vai_runtime#running-vitis-ai-examples-1) and [Running Vitis AI Library Examples](https://github.com/Xilinx/Vitis-AI/blob/master/src/vai_library/README.md#running-vitis-ai-library-examples).

# 4 Known Issues

## 4.1 Continous Memory Allocator(CMA)

The DPU require continuous physical memory, which can be implemented by CMA. In platform source, the CMA is set to 256MB by default. If CMA alloc failed, users can modify uboot command line:
```
ZynqMP> setenv bootargs "earlycon console=ttyPS0,115200 clk_ignore_unused root=/dev/mmcblk0p2 rw rootwait cma=512M"
ZynqMP> saveenv
Saving Environment to FAT... OK
ZynqMP> reset
resetting ..
Check CMA when kernel starts up
[    0.000000] cma: Reserved 512 MiB at 0x000000005fc00000
```

## 4.2 EXT4 Partition Resize
 We will use parted, e2fsprogs-resize2fs and resize-part to expand the ext4 partition to make full use of SD card size when running Vitis-AI test case.

If resize-part is installed on target board, please run the command below to resize ext4 partition:
```
root@xilinx-zcu102-trd:~# resize-part /dev/mmcblk0p2
```

## 4.3 DDR QoS
When AXI HP0 port connects to DPU and use DisplayPort to display, if the QoS settings are not modified, the DisplayPort transmission may under-run, producing black frames or screen flicker intermittently during DPU running. Apart from QoS settings, increasing the read and write issuing capability (outstanding commands) of DPU connected AXI FIFO interface S_AXI_HPC{0, 1}_FPD or S_AXI_HP{0:3}_FPD or S_AXI_LPD may keep the ports busy with always some requests in the queue, which could improve DPU performance highly. 

To address this issue, users could Copy $TRD_HOME/app/dpu_sw_optimize.tar.gz to target board, after linux boot-up, execute zynqmp_dpu_optimize.sh on target board:
```
root@xilinx-zcu102-trd:~# tar -zxvf dpu_sw_optimize.tar.gz
root@xilinx-zcu102-trd:~# ./dpu_sw_optimize/zynqmp/zynqmp_dpu_optimize.sh
```
To get more details, please refer to dpu_sw_optimize/zynqmp/README.md.

## 4.4 do_image_cpio Function Failed Issue
CPIO format does not support sizes greater than 2 GB. Please refer [ug1144 Chapter do_image_cpio Function Failed](https://docs.xilinx.com/r/en-US/ug1144-petalinux-tools-reference-guide/do_image_cpio-Function-Failed) for more details.

<hr/>
<p align="center"><sup>Copyright&copy; 2022 Xilinx</sup></p>
