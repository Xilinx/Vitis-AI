<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>


# Introduction
The following are the **Third-Party** open-source dependencies required by the VitisAI 3.5 when building with Petalinux 2023.1 version:

   | Package Name     | Version     |
   | ---------------- | ----------- |
   | glog             | 0.5.0       |
   | google test      | 1.12.1      |
   | json-c           | 0.16        |
   | protobuf         | 3.21.5      |
   | opencv           | 4.6.0       |
   | libeigen         | 3.4.0       |
   | gflags           | 2.2.2       |
   | python3-pybind11 | 2.10.0      |

There are two ways to install the `Vitis AI` libraries: 
* To rebuild the system by configuring PetaLinux. (Build-Time)
* To install VAI3.5 to the target leveraging a pre-built package. (Run-Time) See the board setup instructions for details of the online installation [process](https://xilinx.github.io/Vitis-AI/3.5/html/docs/workflow-system-integration.html#vitis-ai-online-installation)

# To rebuild the system by configuring PetaLinux
If users want to install VAI3.5 into rootfs when generating system image by PetaLinux, users need to get the VAI3.5 recipes.
Users can get recipes of VAI3.5 by the following two ways.
* Using `recipes-vitis-ai` in this repo.
* Upgrading PetaLinux esdk.

## How to use `recipes-vitis-ai`

**Note**  
`recipes-vitis-ai` enables **Vitis flow by default**. If want to use it in vivado flow, please comment the following line in `recipes-vitis-ai/vart/vart_3.5.bb`
```bash
#PACKAGECONFIG_append = " vitis"
```

1. Copy `recipes-vitis-ai` folder to `<petalinux project>/project-spec/meta-user/`
```bash
cp src/vai_petalinux_recipes/recipes-vitis-ai <petalinux project>/project-spec/meta-user/
```

2. Edit `<petalinux project>/project-spec/meta-user/conf/user-rootfsconfig` file, appending the following lines:
```
CONFIG_vitis-ai-library
CONFIG_vitis-ai-library-dev
CONFIG_vitis-ai-library-dbg
CONFIG_vai-benchmark
CONFIG_vai-sample
```

3. Source PetaLinux tool and run `petalinux-config -c rootfs` command. Select the following option.
```
Select user packages --->
Select [*] vitis-ai-library
```
Then, save it and exit.

4. Run `petalinux-build`

**Note the following:**
* The `recipes-vitis-ai/glog` will throw exception info for vitis-ai-library
* After you run the above successfully, the vitis-ai-library, VART3.5 and the dependent packages will all be installed into rootfs image.
* If you want to compile the examples on the target, please select the `vitis-ai-library-dev` and `packagegroup-petalinux-self-hosted`. Then, recompile the system.
* If you want to install pre-built examples into rootfs, please select *vai-sample* with `petalinux-config -c rootfs` command, and recompile the system.
```bash
Select user packages --->
Select [*] vai-sample
```
* If you want to use vaitracer tool, please select the `vitis-ai-library-dbg`. And copy `recipes-vai-kernel` folder to `<petalinux project>/project-spec/meta-user/`. Then, recompile the system.
```bash
cp src/vai_petalinux_recipes/recipes-vai-kernel <petalinux project>/project-spec/meta-user/
```
* If you want to use vai-benchmark tool, please select it via `petalinux-config -c rootfs` command. And please refer to [README](../../examples/vai_library/vai_benchmark/README.md) for details of using.
```bash
Select user packages --->
Select [*] vai-benchmark
```

## How to use `Upgrade PetaLinux esdk`
Run the following commands to upgrade the PetaLinux.
```bash
source <petalinux-v2023.1>/settings
petalinux-upgrade -u 'http://petalinux.xilinx.com/sswreleases/rel-v2023/sdkupdate/2023.1_update1/' -p 'aarch64'
```
```bash
petalinux-config

[INFO] Sourcing buildtools
[INFO] Menuconfig project


*** End of the configuration.
*** Execute 'make' to start the build or try 'make help'.

WARNING: "Your yocto SDK was changed in tool",
Please input "y" to proceed the installing SDK into project, "n" to exit:y
[INFO] Extracting yocto SDK to components/yocto. This may take time!
[INFO] Sourcing build environment
[INFO] Generating kconfig for Rootfs
[INFO] Silentconfig rootfs
[INFO] Generating plnxtool conf
[INFO] Generating workspace directory
[INFO] Successfully configured project
```

Then, users can find `vitis-ai-library_3.5.bb` recipe in `<petalinux porject>/components/yocto/layers/meta-vitis`.
For details about PetaLinux upgrading, refer to [PetaLinux Upgrade](https://docs.xilinx.com/r/en-US/ug1144-petalinux-tools-reference-guide/petalinux-upgrade-Options)

Note that `2023.1_update1` will be released approximately 1 month after Vitis 3.5 release. The name of `2023.1_update1` may be changed. Please modify it accordingly. 

