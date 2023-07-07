<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>


# Introduction
There are two ways to install the `Vitis AI` libraries: 
* To rebuild the system by configuring PetaLinux. (Build-Time)
* To install VAI3.0 to the target leveraging a pre-built package. (Run-Time) See the board setup instructions for details of the online installation [process](https://xilinx.github.io/Vitis-AI/docs/board_setup/vai_install_to_target.html)

# To rebuild the system by configuring PetaLinux
If users want to install VAI3.0 into rootfs when generating system image by PetaLinux, users need to get the VAI3.0 recipes.
Users can get recipes of VAI3.0 by the following two ways.
* Using `recipes-vitis-ai` in this repo.
* Upgrading PetaLinux esdk.

## How to use `recipes-vitis-ai`

**Note**  
`recipes-vitis-ai` enables **Vitis flow by default**. If want to use it in vivado flow, please comment the following line in `recipes-vitis-ai/vart/vart_3.0.bb`  
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
```

3. Source PetaLinux tool and run `petalinux-config -c rootfs` command. Select the following option.
```
Select user packages --->
Select [*] vitis-ai-library
```
Then, save it and exit.

4. Run `petalinux-build`

Note the following:  
* After you run the above successfully, the vitis-ai-library, VART3.0 and the dependent packages will all be installed into rootfs image.  
* If you want to compile the example on the target, please select the `vitis-ai-library-dev` and `packagegroup-petalinux-self-hosted`. Then, recompile the system.   
* If you want to use vaitracer tool, please select the `vitis-ai-library-dbg`. And copy `recipes-vai-kernel` folder to `<petalinux project>/project-spec/meta-user/`. Then, recompile the system.   
```bash
cp src/vai_petalinux_recipes/recipes-vai-kernel <petalinux project>/project-spec/meta-user/
```

## How to use `Upgrade PetaLinux esdk`
Run the following commands to upgrade the PetaLinux.
```bash
source <petalinux-v2022.2>/settings
petalinux-upgrade -u ‘http://petalinux.xilinx.com/sswreleases/rel-v2022/sdkupdate/2022.2_update1/’ -p ‘aarch64’
```
Then, users can find `vitis-ai-library_3.0.bb` recipe in `<petalinux porject>/components/yocto/layers/meta-vitis-ai`.
For details about PetaLinux upgrading, refer to [PetaLinux Upgrade](https://docs.xilinx.com/r/en-US/ug1144-petalinux-tools-reference-guide/petalinux-upgrade-Options)

Note that `2022.2_update1` will be released approximately 1 month after Vitis 3.0 release. The name of `2022.2_update1` may be changed. Please modify it accordingly. 

