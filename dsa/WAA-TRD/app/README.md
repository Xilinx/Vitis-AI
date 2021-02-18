## Cross-compile WAA-TRD example
* Download the [sdk-2020.2.0.0.sh](https://www.xilinx.com/bin/public/openDownload?filename=sdk-2020.2.0.0.sh)

* Install the cross-compilation system environment, follow the prompts to install. 

    **Please install it on your local host linux system, not in the docker system.**
    ```
    ./sdk-2020.2.0.0.sh
    ```
    Note that the `~/petalinux_sdk` path is recommended for the installation. Regardless of the path you choose for the installation, make sure the path has read-write permissions. 
Here we install it under `~/petalinux_sdk`.

* When the installation is complete, follow the prompts and execute the following command.
    ```
    source ~/petalinux_sdk/environment-setup-aarch64-xilinx-linux
    ```
    Note that if you close the current terminal, you need to re-execute the above instructions in the new terminal interface.

* Download the [vitis_ai_2020.2-r1.3.0.tar.gz](https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_2020.2-r1.3.0.tar.gz) and install it to the petalinux system.
    ```
    tar -xzvf vitis_ai_2020.2-r1.3.0.tar.gz -C ~/petalinux_sdk/sysroots/aarch64-xilinx-linux
    ```

* Cross compile `resnet50_waa` example.
    ```
    cd  ~/Vitis-AI/dsa/WAA-TRD/app/resnet50_waa
    bash -x build.sh
    ```
   Cross compile `adas_detection_waa` example.
    ```
    cd  ~/Vitis-AI/dsa/WAA-TRD/app/adas_detection_waa
    bash -x build.sh
    ``` 	
    If the compilation process does not report any error and the executable file `resnet50_waa` & `adas_detection_waa` are generated in the respective example folder, then the host environment is installed correctly.

