<table width="100%">
  <tr width="100%">
    <td align="center"><img src="https://www.xilinx.com/content/dam/xilinx/imgs/press/media-kits/corporate/xilinx-logo.png" width="30%"/><h1>Vitis AI v2.0</h1>
    </td>
 </tr>
 </table>

# Introduction
There are two ways to install the `Vitis AI` libraries, as shown below. 
* To rebuild the system by configuring PetaLinux. (Build-Time)
* To install the VAI2.0 online. (Run-Time)

# To rebuild the system by configuring PetaLinux
If users want to install VAI2.0 into rootfs when generating system image by PetaLinux, users need to get the VAI2.0 recipes.
Users can get recipes of VAI2.0 by the following two ways.
* Using `recipes-vitis-ai` in this repo.
* Upgrading PetaLinux esdk.

## How to use `recipes-vitis-ai`

**Note**  
	`recipes-vitis-ai` enables Vitis flow by default. If want to use it in vivado flow, please comment the following line in `recipes-vitis-ai/vart/vart_2.0.bb`  
	```
	#PACKAGECONFIG_append = " vitis"
	```

1. Copy `recipes-vitis-ai` folder to `<petalinux project>/project-spec/meta-user/`
```
cp Vitis-AI/tools/Vitis-AI-Recipes/recipes-vitis-ai <petalinux project>/project-spec/meta-user/
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
	* After you run the above successfully, the vitis-ai-library, VART2.0 and the dependent packages will all be installed into rootfs image.  
	* If you want to compile the example on the target, select the `vitis-ai-library-dev` and `packagegroup-petalinux-self-hosted`. Then, recompile the system.   
## How to use `Upgrade PetaLinux esdk`
Run the following commands to upgrade the PetaLinux.
```
source <petalinux-v2021.2>/settings
petalinux-upgrade -u ‘http://petalinux.xilinx.com/sswreleases/rel-v2021/sdkupdate/2021.2_update1/’ -p ‘aarch64’
```
Then, users can find `vitis-ai-library_2.0.bb` recipe in `<petalinux porject>/components/yocto/layers/meta-vitis-ai`.
For details about PetaLinux upgrading, refer to [PetaLinux Upgrade](https://xilinx.github.io/kria-apps-docs/main/build/html/docs/build_petalinux.html)

Note that `2021.2_update1` will be released approximately 1 month after Vitis 2.0 release. The name of `2021.2_update1` may change. Please modify it accordingly. 


# To install the VAI2.0 online
To install the VAI2.0 online, execute `dnf install vitis-ai-library` command on board directly.  
Note the following:  
	* Before the release of `Petalinux VAI2.0 update`, the previous version of Vitis AI will be installed. Usually, `Petalinux VAI2.0 update` will be released approximately 1 month after Vitis 2.0 release.   
	* If you use this method, ensure that the board is connected to the Internet.  
