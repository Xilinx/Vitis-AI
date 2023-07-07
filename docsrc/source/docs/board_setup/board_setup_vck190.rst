Setting up the Versal VCK190
============================

Introduction
------------

This directory contains instructions for running DPUCVDX8G on Versal |trade| AI Core targets. **DPUCVDX8G** is a configurable computation engine dedicated to convolutional neural networks. It supports a highly optimized instruction set, enabling the deployment of most convolutional neural networks.

Step 1: Setup Cross-compiler
----------------------------

.. note:: Perform these steps this on your local host Linux operating system (not inside the docker container). By default, the cross compiler will be installed in ``~/petalinux_sdk_2022.2``.

1. Run the following commands to install the cross-compilation environment:

   .. code-block:: Bash

      cd Vitis-AI/board_setup/vck190
      ./host_cross_compiler_setup.sh


2. When the installation is complete, follow the prompts and execute the following command:

   .. code-block:: Bash

      source ~/petalinux_sdk_2022.2/environment-setup-cortexa72-cortexa53-xilinx-linux

   .. note:: If you close the current terminal, you must re-execute the above instructions in the new terminal interface.

Step 2: Setup the Target
------------------------

The Vitis AI Runtime packages, VART samples, Vitis-AI-Library samples, and models are built into the board image, enhancing the user experience. Therefore, the user need not install Vitis AI Runtime packages and model packages on the board separately. However, following these steps, the users can still install the model or Vitis AI Runtime on their image or on the official image.

1. Installing a Board Image.

   a. Download the SD card image from the following link:

      `VCK190 Production Board <https://www.xilinx.com/member/forms/download/design-license-xef.html?filename=xilinx-vck190-dpu-v2022.2-v3.0.0.img.gz>`__

      .. note:: The version of the VCK190 production board image is 2022.2.

   b.  Use Etcher software to burn the image file onto the SD card.

   c.  Insert the imaged SD card into the target board.

   d.  Plug in the power adapter and boot the board using the serial port to interact with the target.

   e.  Configure the IP address and related settings for the board using the serial port.

   For additional details, refer to `Setting Up the Evaluation Board <https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/Setting-Up-the-Evaluation-Board>`__.

2. (Optional) How to leverage Vitis AI with PetaLinux 2022.2

   You can install the Vitis AI libraries on the target either at build-time or at run-time:

   		- Build-Time: Rebuild the system by configuring PetaLinux. For ``VAI3.0 Recipes``, refer to `Vitis-AI-Recipes <../petalinux-recipes.html>`__
   		- Run-Time: Install Vitis AI online via `dnf`. Execute ``dnf install packagegroup-petalinux-vitisai`` to complete the installation on the target. For more details, refer `VAI3.0 Online Install <../petalinux-recipes.html>`__

3. (Optional) How to update Vitis AI Runtime on the target

   If you have an updated version of the Vitis AI Runtime and wish to install the update to your target, follow these steps.

   -  Copy the board_setup/mpsoc folder to the board using scp:

      .. code-block:: Bash

         scp -r board_setup/vck190 root@IP_OF_BOARD:~/

   -  Log in to the board using ssh. You can also use the serial port to login.
   -  Now, install the Vitis AI Runtime. Execute the following commands:

      ::
		
		cd ~/vck190
		bash target_vart_setup.sh


4. (Optional) Download the model.

   You can now select a model from the Vitis AI Model Zoo `Vitis AI Model Zoo <../workflow-model-zoo.html>`__.  Navigate to the  `model-list subdirectory  <https://github.com/Xilinx/Vitis-AI/tree/master/model_zoo/model-list>`__  and select the model that you wish to test. For each model, a YAML file provides key details of the model. In the YAML file there are separate hyperlinks to download the model for each supported target.  Choose the correct link for your target platform and download the model.

   a. Take the VCK190 ``resnet50`` model as an example.

      .. code-block:: Bash

          cd /workspace
          wget https://www.xilinx.com/bin/public/openDownload?filename=resnet50-vck190-r3.0.0.tar.gz -O resnet50-vck190-r3.0.0.tar.gz

   b. Copy the downloaded file to the board using scp with the following command:

      .. code-block:: Bash

         scp resnet50-vck190-r3.0.0.tar.gz root@IP_OF_BOARD:~/

   c. Log in to the board (via ssh or serial port) and install the model package.

      .. code-block:: Bash

         tar -xzvf resnet50-vck190-r3.0.0.tar.gz
         cp resnet50 /usr/share/vitis_ai_library/models -r

Step 3: Run the Vitis AI Examples
----------------------------------

Refer to :ref:`mpsoc-run-vitis-ai-examples` to run Vitis AI examples.

References
----------

-  `Vitis AI User Guide <https://www.xilinx.com/html_docs/vitis_ai/3_0/index.html>`__


.. |trade|  unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim:
.. |reg|    unicode:: U+000AE .. REGISTERED TRADEMARK SIGN
   :ltrim:
