# Build flow  of Resnet-50 example: 
:pushpin: **Note:** This application can be run only on VCK190 ES1 Board

## Generate SD card image

###### **Note:** It is recommended to follow the build steps in sequence.

The following tutorials assume that the Vitis and XRT environment variable is set as given below.

Open a linux terminal. Set the linux as Bash mode.

```
    source < vitis-install-directory >/Vitis/2020.2/settings64.sh
    source < path-to-XRT-installation-directory >/setup.sh
    cd ${VAI_HOME}/Whole-App-Acceleration/apps/resnet50/build_flow/DPUCVDX8G_vck190
    ./run.sh
```

###### **Note:** To be able to build the vck190 platform, GCC version higher than 6.0 is required and the temp directory of petalinux build should not be in a network file system. 

Note that 
- Generated SD card image will be here **${VAI_HOME}/Whole-App-Acceleration/apps/resnet50/build_flow/DPUCVDX8G_vck190/binary_container_1/sd_card.img**.
- Currently, the preprocess accelerator supports FHD image resolution. To change the maximum resolution of input image and other metrics, config params header file of the preprocess accelerator can be modified. Path: Vitis-AI/Whole-App-Acceleration//plugins/blobfromimage/pl/xf_config_params.h


## Installing board image
- Use Etcher software to burn the sd card image file onto the SD card.

## Setting up & running on VCK190

This part is about how to run the Resnet50 example on vck190 board.

  * Copy xmodel and build.sh from downloaded package to SD card 
  ```
    wget https://www.xilinx.com/bin/public/openDownload?filename=waa_versal_sd_card_vai2.0.tar.gz -O waa_versal_sd_card_vai2.0.tar.gz

    tar -xzvf waa_versal_sd_card_vai2.0.tar.gz
  ```
  * Copy app source file from `cd  ${VAI_HOME}/Whole-App-Acceleration/apps/resnet50` to SD card.

  * Download the images at http://image-net.org/download-images and copy images to ` ${VAI_HOME}/Whole-App-Acceleration/apps/resnet50/img` directory 

  * Copy following content of  ${VAI_HOME}/Whole-App-Acceleration/apps/resnet50 directory to the BOOT partition of the SD Card.
    ```
        img
        app_test.sh
        word.txt
    ```


  * Please insert SD_CARD on the VCK190 board. After the linux boot, run:

  * compile `resnet50` example.
    ```
    cd /media/sd-mmcblk0p1/
    bash -x build.sh
    ```
      If the compilation process does not report any error and the executable file `./bin/resnet50.exe` is generated.

  * Performance test with & without waa

    ```
    % cd /media/sd-mmcblk0p1/
    % ./app_test.sh --xmodel_file ./model_vck190/resnet_v1_50_tf.xmodel --image_dir ./img/ --performance_diff

    ```

  * Functionality test with waa
    ```
    % ./app_test.sh --xmodel_file ./model_vck190/resnet_v1_50_tf.xmodel --image_dir ./img/ --verbose

    ```

  * Functionality test without waa
    ```
    % ./app_test.sh --xmodel_file ./model_vck190/resnet_v1_50_tf.xmodel --image_dir ./img/ --verbose --use_sw_pre_proc

    ```