<table class="sphinxhide">
 <tr>
   <td align="center"><img src="https://raw.githubusercontent.com/Xilinx/Image-Collateral/main/xilinx-logo.png" width="30%"/><h1>Vitis AI</h1><h0>Adaptable & Real-Time AI Inference Acceleration</h0>
   </td>
 </tr>
</table>

# Vitis AI Runtime v3.0 Quick-start for RNN
 
## Setting Up the Host
Before you compile the VART for cloud, please load and run the docker container according to the [installation instructions](../../../docs/quick-start/install)

## Compile VART  

* Compile the VART and install the VART. Execute the following command in order.
	* unilog
	```
	cd /workspace/tools/Vitis-AI-Runtime/VART/unilog
	sudo bash cmake.sh --install-prefix=/usr
	```
	* xir
	```
	cd /workspace/tools/Vitis-AI-Runtime/VART/xir
	sudo bash cmake.sh --install-prefix=/usr
	```
	* target_factory
	```
	cd /workspace/tools/Vitis-AI-Runtime/VART/target_factory
	sudo bash cmake.sh --install-prefix=/usr
	```
	* vart
	```
	cd /workspace/tools/Vitis-AI-Runtime/VART/vart
        sudo bash cmake.sh --install-prefix=/usr --cmake-options='-DENABLE_VART_RUNNER=ON -DENABLE_XRNN_RUNNER=ON -DENABLE_CPU_RUNNER=OFF -DENABLE_SIM_RUNNER=OFF'
	```

After you compile each module of VART, the libraries and test programs will be generated under `~/build/build.Ubuntu.18.04.x86_64.Debug/`.  
Take `unilog` as an example, you will find the `libunilog.so` under `~/build/build.Ubuntu.18.04.x86_64.Debug/unilog/src` and the test programs under
`~/build/build.Ubuntu.18.04.x86_64.Debug/unilog/test`.  

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


