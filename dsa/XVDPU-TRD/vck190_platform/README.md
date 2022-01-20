# VCK190 Base TRD

The Versal:tm: Base TRD consists of a series of platforms, accelerators, and Jupyter
notebooks targeting the VCK190 evaluation board. A platform is a Vivado:registered: design
with a pre-instantiated set of I/O interfaces and a corresponding PetaLinux BSP
and image that includes the required kernel drivers and user-space libraries to
exercise those interfaces. Accelerators are mapped to FPGA logic resources
and/or AI Engine cores and stitched into the platform using the Vitis:tm: unified software platform toolchain.

# Build Platform

To build the Vitis embedded platforms from source code in this repository, you will need to have the following tools
installed and follow the build instructions:

- A Linux-based host OS supported by Vitis and PetaLinux
- [Vitis][1] 2021.2
- [PetaLinux][2] 2021.2

[1]: https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/vitis.html
[2]: https://www.xilinx.com/support/download/index.html/content/xilinx/en/downloadNav/embedded-design-tools.html

To learn how to customize Vitis embedded platforms, please refer to [Vitis Platform Creation Tutorials](https://github.com/Xilinx/Vitis-Tutorials/tree/master/Vitis_Platform_Creation).

Vitis and PetaLinux environment need to be setup before building the platform.

```bash
source <Vitis_install_path>/Vitis/2021.2/settings64.sh
source <PetaLinux_install_path>/settings.sh
```
This package comes with sources to generate the Vitis platform with these steps:

1. Generate hardware specification file (XSA) using Vivado.
2. Generate software components of platform (using Petalinux).
3. Generate the Vitis platform by packaging hardware and software together

To generate the Vitis Versal TRD platform, type the following command:
```bash
make all
```

Notes:

- When building PetaLinux image from source code, the build temp directory is set to **/tmp/xilinx_vck190_base-2021.2**. You can update the build temp directory by modifying CONFIG_TMP_DIR_LOCATION option in **<platform_name>/petalinux/xilinx-vck190-base-trd/project-spec/configs/config** file.

The platform file .xpfm will be genereated at **'platforms/xilinx_vck190_mipiRxSingle_hdmiTx_202110_1/vck190_mipiRxSingle_hdmiTx.xpfm'**

# License

Licensed under the Apache License, version 2.0 (the "License"); you may not use this file 
except in compliance with the License.

You may obtain a copy of the License at
[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)


Unless required by applicable law or agreed to in writing, software distributed under the 
License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, 
either express or implied. See the License for the specific language governing permissions 
and limitations under the License.    
