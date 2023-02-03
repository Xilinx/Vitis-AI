Setting up a Zynq UltraScale+ MPSoC Target
==========================================

Introduction
------------

This directory contains instructions for running DPUCZDX8G on Zynq |reg| Ultrascale+ |trade| MPSoC targets. **DPUCZDX8G** is a configurable computation engine dedicated to convolutional neural networks. It supports a highly optimized instruction set, enabling the deployment of most convolutional neural networks.

Step 1: Setup Cross-compiler
----------------------------

.. note:: Perform these steps this on your local host Linux operating system (not inside the docker container). By default, the cross compiler will be installed in ``~/petalinux_sdk_2022.2``.

1. Run the following commands to install the cross-compilation environment:

   .. code-block:: Bash

      cd Vitis-AI/board_setup/mpsoc
      ./host_cross_compiler_setup.sh


2. When the installation is complete, follow the prompts and execute the following command:

   .. code-block:: Bash

      source ~/petalinux_sdk_2022.2/environment-setup-cortexa72-cortexa53-xilinx-linux

   .. note:: If you close the current terminal, you must re-execute the above instructions in the new terminal interface.

Step 2: Setup the Target
------------------------

The Vitis AI Runtime packages, VART samples, Vitis-AI-Library samples, and models are built into the board image, enhancing the user experience. Therefore, the user need not install Vitis AI Runtime packages and model packages on the board separately. However, following these steps, the users can still install the model or Vitis AI Runtime on their image or on the official image.

1. Installing a Board Image.

   a.  Download the SD card image from the appropriate link:

      - `ZCU102 <https://www.xilinx.com/member/forms/download/design-license-xef.html?filename=xilinx-zcu102-dpu-v2022.2-v3.0.0.img.gz>`__
      - `ZCU104 <https://www.xilinx.com/member/forms/download/design-license-xef.html?filename=xilinx-zcu104-dpu-v2022.2-v3.0.0.img.gz>`__
      - `KV260 <https://www.xilinx.com/member/forms/download/design-license-xef.html?filename=xilinx-kv260-dpu-v2022.2-v3.0.0.img.gz>`__

	.. note:: For the ZCU102/ZCU104/KV260, the version of the board image should be 2022.2 or above.

   b.  Use Etcher software to burn the image file onto the SD card.

   c.  Insert the imaged SD card into the target board.

   d.  Plug in the power adapter and boot the board using the serial port to interact with the target.

   e.  Configure the IP address and related settings for the board using the serial port.

   For additional details, refer to `Setting Up the Evaluation Board <https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/Setting-Up-the-Evaluation-Board>`__.

2. (Optional) Run ``zynqmp_dpu_optimize.sh`` to optimize board settings.

   The script runs automatically after the board boots up with the official image. But you can also find the ``dpu_sw_optimize.tar.gz`` in `DPUCZDX8G.tar.gz <https://www.xilinx.com/bin/public/openDownload?filename=DPUCZDX8G.tar.gz>`__.

   .. code-block:: Bash

       cd ~/dpu_sw_optimize/zynqmp/
       ./zynqmp_dpu_optimize.sh

3. (Optional) How to leverage Vitis AI with PetaLinux 2022.2

   You can install the Vitis AI libraries on the target either at build-time or at run-time:

   		- Build-Time: Rebuild the system by configuring PetaLinux. For ``VAI3.0 Recipes``, refer to `Vitis-AI-Recipes <../petalinux-recipes.html>`__
   		- Run-Time: Install Vitis AI online via `dnf`. Execute ``dnf install packagegroup-petalinux-vitisai`` to complete the installation on the target. For more details, refer `VAI3.0 Online Install <../petalinux-recipes.html>`__
   
4. (Optional) How to update Vitis AI Runtime on the target

   If you have an updated version of the Vitis AI Runtime and wish to install the update to your target, follow these steps.

   -  Copy the board_setup/mpsoc folder to the board using scp:

      .. code-block:: Bash

         scp -r board_setup/mpsoc root@IP_OF_BOARD:~/

   -  Log in to the board using ssh. You can also use the serial port to login.
   -  Now, install the Vitis AI Runtime. Execute the following commands:

      ::
		
		cd ~/mpsoc
		bash target_vart_setup.sh
		
		
5. (Optional) Download the model.
   
   You can now select a model from the Vitis AI Model Zoo `Vitis AI Model Zoo <../workflow-model-zoo.html>`__.  Navigate to the  `model-list subdirectory  <https://github.com/Xilinx/Vitis-AI/tree/master/model_zoo/model-list>`__  and select the model that you wish to test. For each model, a YAML file provides key details of the model. In the YAML file there are separate hyperlinks to download the model for each supported target.  Choose the correct link for your target platform and download the model.

   a. Take the ZCU102 ``resnet50`` model as an example.

      .. code-block:: Bash

          cd /workspace
          wget https://www.xilinx.com/bin/public/openDownload?filename=resnet50-zcu102_zcu104_kv260-r3.0.0.tar.gz -O resnet50-zcu102_zcu104_kv260-r3.0.0.tar.gz

   b. Copy the downloaded file to the board using scp with the following command:

      .. code-block:: Bash

          scp resnet50-zcu102_zcu104_kv260-r3.0.0.tar.gz root@IP_OF_BOARD:~/

   c. Log in to the board (via ssh or serial port) and install the model package:

      .. code-block:: Bash

          tar -xzvf resnet50-zcu102_zcu104_kv260-r3.0.0.tar.gz
          cp resnet50 /usr/share/vitis_ai_library/models -r

.. _mpsoc-run-vitis-ai-examples:

Step 3: Run the Vitis AI Examples
---------------------------------

1. Download the `vitis_ai_runtime_r3.0.0_image_video.tar.gz <https://www.xilinx.com/bin/public/openDownload?filename=vitis_ai_runtime_r3.0.0_image_video.tar.gz>`__ from host to the target using scp with the following command:

   ``[Host]$scp vitis_ai_runtime_r3.0.*_image_video.tar.gz root@[IP_OF_BOARD]:~/``

2. Unzip the ``vitis_ai_runtime_r3.0.0_image_video.tar.gz`` package on the target.

      .. code-block:: Bash

       cd ~
       tar -xzvf vitis_ai_runtime_r*3.0._image_video.tar.gz -C Vitis-AI/examples/vai_runtime

3. Navigate to the example directory on the target board. Take ``resnet50`` as an example.

   ``cd ~/Vitis-AI/examples/vai_runtime/resnet50``

4. Run the example.

   ``./resnet50 /usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel``

   For examples with video input, only ``webm`` and ``raw`` format are supported by default with the official system image. If you want to support video data in other formats, you need to install the relevant packages on the system.

Launching Commands for VART Samples on Edge
-------------------------------------------

+-----+--------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| No. | Example Name       | Command                                                                                                                                                                                  |
+=====+====================+==========================================================================================================================================================================================+
| 1   | resnet50           | ./resnet50 /usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel                                                                                                                   |
+-----+--------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 2   | resnet50_pt        | ./resnet50_pt /usr/share/vitis_ai_library/models/resnet50_pt/resnet50_pt.xmodel ../images/001.jpg                                                                                        |
+-----+--------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 3   | resnet50_ext       | ./resnet50_ext /usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel ../images/001.jpg                                                                                             |
+-----+--------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 4   | resnet50_mt_py     | python3 resnet50.py 1 /usr/share/vitis_ai_library/models/resnet50/resnet50.xmodel                                                                                                        |
+-----+--------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 5   | inception_v1_mt_py | python3 inception_v1.py 1 /usr/share/vitis_ai_library/models/inception_v1_tf/inception_v1_tf.xmodel                                                                                      |
+-----+--------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 6   | pose_detection     | ./pose_detection video/pose.webm /usr/share/vitis_ai_library/models/sp_net/sp_net.xmodel /usr/share/vitis_ai_library/models/ssd_pedestrian_pruned_0_97/ssd_pedestrian_pruned_0_97.xmodel |
+-----+--------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 7   | video_analysis     | ./video_analysis video/structure.webm /usr/share/vitis_ai_library/models/ssd_traffic_pruned_0_9/ssd_traffic_pruned_0_9.xmodel                                                            |
+-----+--------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 8   | adas_detection     | ./adas_detection video/adas.webm /usr/share/vitis_ai_library/models/yolov3_adas_pruned_0_9/yolov3_adas_pruned_0_9.xmodel                                                                 |
+-----+--------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 9   | segmentation       | ./segmentation video/traffic.webm /usr/share/vitis_ai_library/models/fpn/fpn.xmodel                                                                                                      |
+-----+--------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| 10  | squeezenet_pytorch | ./squeezenet_pytorch /usr/share/vitis_ai_library/models/squeezenet_pt/squeezenet_pt.xmodel                                                                                               |
+-----+--------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

References
----------

-  `Vitis AI User Guide <https://www.xilinx.com/html_docs/vitis_ai/3_0/index.html>`__


.. |trade|  unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim:
.. |reg|    unicode:: U+000AE .. REGISTERED TRADEMARK SIGN
   :ltrim:
