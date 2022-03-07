# Vitis TRD Platform for the vck190 Board

## Building the Platform

**Last Tested Vivado Release: 2020.2**

The platform build process is entirely scripted. Note that as this platform
build process involves cross-compiling Linux, build of the platform is supported
on Linux environments **only** (although it is possible to build inside a VM or
Docker container).

Also note that the default PetaLinux configuration uses local scratchpad areas. This
will *not* work if you are building on a networked file system; Yocto will error out.
Update PetaLinux to change the build area to a locally-mounted hard drive.

After cloning the platform source, and with both Vivado and PetaLinux set up, run
`make all` command. 

Note that by default this Makefile will install the platform to "platform_repo/xilinx_vck190_es1_trd_202020_1/export/xilinx_vck190_es1_trd_202020_1/"

More details you can find from `make help`

## Platform Specification

### General Information

| Type                | Value                       |
| -----------------   | --------------------------- |
| Vitis version       | 2020.2                      |
| PetaLinux version   | 2020.2                      |
| XRT Tag version     | [202020.2.8.0_Petalinux](https://github.com/Xilinx/XRT/releases/tag/202020.2.8.0_PetaLinux)              |
| Associated Document | UG1442                      | 
| Supported Device(s) | XCVC1902-2MSEVSVA2197       |
| Supported Board(s)  | EK-VCK190-G-ED              |

### Interfaces

| Interface | Region | Details            |
| --------- | ------ | ------------------ |
| UART      | PS     |                    |
| GEM       | PS     |                    |
| USB       | PS     |                    |
| SDIO      | PS     |                    |
| HDMI      | PL     |                    |
| MIPI      | PL     |                    |

### Hardware Configurations

| Configuration                 | Values                                                | Details                             |
| ----------------------------- | ----------------------------------------------------- | ------------------------------------|
| LPDDR Size                    | 8GB                                                   |                                     |
| AI Engine                     | Enabled                                               |                                     |
| HDMI                          | Enabled                                               |                                     |
| MIPI                          | Enabled                                               |                                     |

### Software Configurations

Documentation for the VCK190 Base TRD design is available here
User Guide: https://www.xilinx.com/support/documentation/boards_and_kits/vck190/ug1442-vck190-trd.pdf
Tutorial: https://xilinx.github.io/vck190-base-trd/build/html/index.html

# Notes

Use the V++ -p option to generate the sd_card.img file that consists rootfs.ext4 provided by petalinux along with the Image,BOOT.BIN and system.dtb from platform, v++ generated xclbin and host.exe files.

Once the Vitis platform is ready, some example applications to build with these platforms can be found here:
https://github.com/Xilinx/Vitis_Accel_Examples
