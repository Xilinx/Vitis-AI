## Cross-compile WAA-TRD example for Edge Platform
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

* Cross compile `resnet50` example.
    ```
    cd  ~/Vitis-AI/dsa/WAA-TRD/app/resnet50
    bash -x build.sh
    ```
   Cross compile `resnet50_jpeg` example.
    ```
    cd  ~/Vitis-AI/dsa/WAA-TRD/app/resnet50_jpeg
    bash -x build.sh
    ```
   Cross compile `adas_detection` example.
    ```
    cd  ~/Vitis-AI/dsa/WAA-TRD/app/adas_detection
    bash -x build.sh
    ``` 	
    If the compilation process does not report any error and the executable file `resnet50`, 'resnet50_jpeg' & `adas_detection` are generated in the respective example folder, then the host environment is installed correctly.


## Cross-compile resnet50_int8 for cloud platform

**Note that the docker container needs to be loaded and the below commands need to be run in the docker environment. Docker installation instructions are available [here](../../../README.md#Installation)**

* Cross compile `resnet50_int8` example.
    ```
    cd /workspace/dsa/WAA-TRD/app/resnet50_int8
    bash -x build.sh
    ```

## Model Performance
Refer below table for examples with demonstration on complete acceleration of various applications where pre/post-processing of several networks are accelerated on different target platforms.


| No. | Application                                           | Backbone Network | Accelerated Part(s)   | H/W Accelerated Functions                     | DPU Supported (% Improvement Over Non-WAA App) |
|-----|-------------------------------------------------------|------------------|----------------------------|-----------------------------------------------|--------------------------------------------------------------------------|
| 1   | adas_detection                                        | Yolo v3          | Pre-process                | resize, letter box, scale                     | ZCU102  (*64%) , ALVEO-U50 (*44%)                                        |
| 2   | resnet50_int8                                         | resnet-50        | Pre-process                | resize, mean subtraction, scale               | ALVEO-U200 (37%)                                                         |
| 3   | resnet50_jpeg                                         | resnet-50        | Pre-process                | JPEG decoder, resize, mean subtraction, scale | ZCU102  (*69%)                                                           |


`*` Includes imread/jpeg-decoder latency in both WAA and non WAA app