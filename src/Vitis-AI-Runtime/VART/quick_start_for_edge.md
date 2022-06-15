<table width="100%">
  <tr width="100%">
    <td align="center"><img src="https://www.xilinx.com/content/dam/xilinx/imgs/press/media-kits/corporate/xilinx-logo.png" width="30%"/><h1>Vitis AI Runtime v2.5</h1>
    </td>
 </tr>
 </table>
 
## Setting Up the Host
Follow [Setting Up the Host](../../../setup/mpsoc#step1-setup-cross-compiler) to set up the host automatically for edge.   
Or refer to the following steps to set up the host manually.

1. Download the [sdk-2022.1.0.0.sh](https://www.xilinx.com/bin/public/openDownload?filename=sdk-2022.1.0.0.sh)

2. Install the cross-compilation system environment, follow the prompts to install. 

**Please install it on your local host linux system, not in the docker system.**
```
./sdk-2022.1.0.0.sh
```
Note that the `~/petalinux_sdk_2022.1` path is recommended for the installation. Regardless of the path you choose for the installation, make sure the path has read-write permissions. 
Here we install it under `~/petalinux_sdk_2022.1`.

3. When the installation is complete, follow the prompts and execute the following command.
```
source ~/petalinux_sdk_2022.1/environment-setup-cortexa72-cortexa53-xilinx-linux
```
Note that if you close the current terminal, you need to re-execute the above instructions in the new terminal interface.

4. Download the [vitis_ai_2022.1-r2.5.0.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_2022.1-r2.5.0.tar.gz) and install it to the petalinux system.
```
rm -r ~/petalinux_sdk_2022.1/sysroots/cortexa72-cortexa53-xilinx-linux/usr/share/cmake/XRT/
tar -xzvf vitis_ai_2022.1-r2.5.0.tar.gz -C ~/petalinux_sdk_2022.1/sysroots/cortexa72-cortexa53-xilinx-linux
```

## Compile the VART 
To modify the VART source code, view and modify them under `Vitis-AI/src/Vitis-AI-Runtime/VART`.  

* unilog
```
cd Vitis-AI/src/Vitis-AI-Runtime/VART/unilog
./cmake.sh --clean
```
* xir
```
cd Vitis-AI/src/Vitis-AI-Runtime/VART/xir
./cmake.sh --clean
```
* target_factory
```
cd Vitis-AI/src/Vitis-AI-Runtime/VART/target_factory
./cmake.sh --clean
```
* vart
```
cd Vitis-AI/src/Vitis-AI-Runtime/VART/vart
./cmake.sh --clean
```
After you compile each module of VART, the libraries and test programs will be generated under `~/build/build.linux.2022.1.aarch64.Debug/`.  
Take `unilog` as an example, you will find the `libunilog.so` under `~/build/build.linux.2022.1.aarch64.Debug/unilog/src` and the test programs under
`~/build/build.linux.2022.1.aarch64.Debug/unilog/test`.  
And the libraries will be installed to cross compile system by default. The installation path is `<sdk install path>/sysroots/aarch64-xilinx-linux/install/Debug/lib`

To reduce the size of generated libraries, execute the following command.
```
./cmake.sh --cmake-options='-DCMAKE_CXX_FLAGS_RELEASE=-s'
```

For more parameters setting, refer to the following table.

 <summary><b> Commands for VART module compilation </b></summary>
 
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

