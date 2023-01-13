<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>

# Vitis AI Runtime v3.0 Embedded Quick-start
 
The VART libraries and Vitis-AI applications are typically cross-compiled on the host, leveraging a Vitis-AI specific sysroot and SDK.

Since Vitis-AI has a different release cycle than PetaLinux, Vitis-AI PetaLinux recipes are released slightly later than the public PetaLinux release.  The result is that the SDK and sysroot must be installed as a separate process from the Petalinux install.  Also, it is important to note that the bitbake recipes that are required to build these components are released as a part of the Vitis AI repository, and may be found in the [board_setup](../../board_setup/vai_install_to_target/README.md).  Leveraging the provided recipes, petalinux-build can generate the required sysroot and SDK.

Outside of the petalinux-build flow, users should follow the below simple instructions to create the required sysroot and SDK on the host.

****************
:pushpin:**Important Note!** Installation of the Petalinux build tools and cross-compiler on the host machine must be done on the native Linux host outside of the Vitis AI Docker container.

****************

## Setting Up the Host

An easy way to get started is to simply leverage scripts that are provided to create the required Sysroot and compile the SDK.  Please follow the below steps:

1. If you have not already done so, you may wish to install Petalinux on your host machine.  The Petalinux installer run file is available from the Xilinx [downloads site](https://www.xilinx.com/member/forms/download/xef.html?filename=petalinux-v2022.2-10141622-installer.run).  The process for installing Petalinux is:

```
$ sudo chmod +x ./petalinux-v2022.2-10141622-installer.run
$ ./petalinux-v2022.2-10141622-installer.run -d <install destination>
```

2. Download the cross-compiler installation script [sdk-2022.2.0.0.sh](https://www.xilinx.com/bin/public/openDownload?filename=sdk-2022.2.0.0.sh) to the host.  We recommend installing the SDK to the user's home folder (~/petalinux_sdk_2022.2) for the best user experience.  If you select a different install path, ensure that it has read-write permissions.

```
$ mkdir petalinux_sdk_2022.2
$ cd petalinux_sdk_2022.2
$ wget https://www.xilinx.com/bin/public/openDownload?filename=sdk-2022.2.0.0.sh -O sdk-2022.2.0.0.sh
$ chmod +x sdk-2022.2.0.0.sh
```

3. Execute the cross-compiler installation script.  Note that this script merges both a shell script as well as a binary .zip file where the .zip file is incorporated as a payload within the script.

```
$ ./sdk-2022.2.0.0.sh
```

4. The script will ask you to enter the target directory for SDK installation (default: /opt/petalinux/2022.2).  You can override this to use the current directory (~/petalinux_sdk_2022.2) by entering "./".

5. Once the installation is complete, execute the following command:
```
source ~/petalinux_sdk_2022.2/environment-setup-cortexa72-cortexa53-xilinx-linux
```

****************
:pushpin:**Important Note!** This SDK environment must be sourced prior to attempting cross-compilation of the VART libraries.  If you close the current terminal window, you need to re-execute the above instructions in the new terminal window.

****************

5. Next, we need to download the sysroot tarball [vitis_ai_2022.2-r3.0.0.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_2022.2-r3.0.0.tar.gz).
 
```
wget https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_2022.2-r3.0.0.tar.gz -O vitis_ai_2022.2-r3.0.0.tar.gz
rm -r ~/petalinux_sdk_2022.2/sysroots/cortexa72-cortexa53-xilinx-linux/usr/share/cmake/XRT/
tar -xzvf vitis_ai_2022.2-r3.0.0.tar.gz -C ~/petalinux_sdk_2022.2/sysroots/cortexa72-cortexa53-xilinx-linux
```

****************
:pushpin:**Important Note!** If you fail to execute removal of the /share/cmake/XRT directory you will see an error:

```
Insert the error message 
```
****************

## Compile VART 
Users can now review and modify the VART source code which is included in the Vitis AI installation: `<vitis-ai-install-path>/src/vai_runtime`.  

* unilog
```
cd <vitis-ai-install-path>/src/vai_runtime/unilog
./cmake.sh --clean
```
* xir
```
cd <vitis-ai-install-path>/src/vai_runtime/xir
./cmake.sh --clean
```
* target_factory
```
cd <vitis-ai-install-path>/src/vai_runtime/target_factory
./cmake.sh --clean
```
* vart
```
cd <vitis-ai-install-path>/src/vai_runtime/vart
./cmake.sh --clean
```
After you compile each module of VART, the libraries and test applications will be generated under `~/build/build.linux.2022.2.aarch64.Debug/`.

Take `unilog` as an example.  You will find the compiled `libunilog.so` under `~/build/build.linux.2022.2.aarch64.Debug/unilog/src` and the test applications for this same library under `~/build/build.linux.2022.2.aarch64.Debug/unilog/test`. 

Also, these libraries will be installed into the sysroot by default. The installation path is `<sdk install path>/sysroots/aarch64-xilinx-linux/install/Debug/lib`

To reduce the size of generated libraries, execute the following command:
```
./cmake.sh --cmake-options='-DCMAKE_CXX_FLAGS_RELEASE=-s'
```

For more additional cmake settings, refer to the following table:

<b> Commands for VART module compilation </b>
 
| No\. | Command                  | Comment                                                      |
| :--- | :----------------------- | :----------------------------------------------------------- |
| 1    | ./cmake.sh --help        | Show help                              |
| 2    | ./cmake.sh --clean       | Discard build dir before build                              |
| 3    | ./cmake.sh --build-only  | Build only, will not install the library                  |
| 4    | ./cmake.sh --type[=TYPE] | Build type. VAR {release, debug(default)}            |
| 5    | ./cmake.sh --pack[=FORMAT]           | Enable packing and set package format. VAR {deb, rpm}         |
| 6    | ./cmake.sh --build-dir[=DIR]           | Set customized build directory    |
| 7    | ./cmake.sh --install-prefix[=PREFIX]   | Set customized install prefix         |
| 8    | ./cmake.sh --cmake-options[=OPTIONS]   | Append more cmake options        |
