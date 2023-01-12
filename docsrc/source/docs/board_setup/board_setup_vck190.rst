Setting up the Versal VCK190
==========================================

Introduction
-------------

This directory contains instructions for running DPUCVDX8G on Versal® AI Core platforms. **DPUCVDX8G** is a configurable computation engine dedicated to convolutional neural networks. It includes highly optimized instructions, and supports most convolutional neural networks, such as VGG, ResNet, GoogleNet, YOLO, SSD, MobileNet, FPN, and others. With Vitis™ AI, Xilinx® has integrated all the edge and cloud solutions under a unified API and toolset.

Step 1: Setup Cross-compiler
-----------------------------

1. Run the following command to install cross-compilation system environment:

   .. note:: Install it on your local host Linux system, not in the docker system. By default, the Cross Compiler will be installed in ``~/petalinux_sdk_2022.2`` by default. For the ``VCK190 Production`` board, use ``host_cross_compiler_setup sh``.

   .. code-block::

      ./host_cross_compiler_setup.sh

2. When the installation is complete, follow the prompts and execute the following command:

   .. code-block::

      source ~/petalinux_sdk_2022.2/environment-setup-cortexa72-cortexa53-xilinx-linux

   .. note:: If you close the current terminal, you need to re-execute the above instructions in the new terminal interface.

Step 2: Setup the Target
-------------------------

The Vitis AI Runtime packages, VART samples, Vitis-AI-Library samples, and models are built into the board image, enhancing the user experience. Therefore, the user need not install Vitis AI Runtime packages and model packages on the board separately. However, following these steps, the users can still install the model or Vitis AI Runtime on their image or on the official image.

1. Installing a Board Image.

   a. Download the SD card system image files from the following links:

      `VCK190 Production Board <https://www.xilinx.com/member/forms/download/design-license-xef.html?filename=xilinx-vck190-dpu-v2022.2-v3.0.0.img.gz>`__

      .. note:: The version of the VCK190 production board image is 2022.2.

   b. Use Etcher software to burn the image file onto the SD card.

   c. Insert the SD card with the image into the destination board.

   d. Plug in power and boot the board using the serial port to perate on the system.

   e. Set up the IP information of the board using the serial port.

   For the details, refer `Setting Up the Evaluation Board <https://docs.xilinx.com/r/en-US/ug1414-vitis-ai/Setting-Up-the-Evaluation-Board>`__

2. (Optional) How to install Vitis AI for PetaLinux 2022.2.

   There are two ways to install the dependent libraries of Vitis AI:

   -  **Build-Time**: Rebuild the system by configuring PetaLinux. For ``VAI3.0 Recipes``, refer to `Vitis-AI-Recipes <../petalinux-recipes.html>`__.

   -  **Run-Time**: Install the Vitis AI online via ``dnf``. Execute ``dnf install packagegroup-petalinux-vitisai`` to complete the installation on the target. For more details, refer `VAI3.0 Online Install <../petalinux-recipes.html>`__.


3. (Optional) How to update Vitis AI Runtime and install them separately.

   If you want to update the Vitis AI Runtime or install them to your custom board image, follow these steps:

   a. Copy the following folder to the board using scp.

      .. code-block::

          scp -r board_setup/vck190 root@IP_OF_BOARD:~/

   b. Log in to the board using ssh. You can also use the serial port to login.

   c. Install the Vitis AI Runtime. Execute the following command:

      .. code-block::

          cd ~/vck190
          bash target_vart_setup.sh

4. (Optional) Download the model.

   Click `Xilinx AI Model Zoo <../../../../model_zoo/model-list>`__ to view all the models. For each model, a YAML file is used to describe all the details about the model. You will find download links for the various pre-built Xilinx platforms in the YAML. Choose the corresponding model and download it.

   a. Take ``resnet50`` of VCK190 as an example.

      .. code-block::

          cd /workspace
          wget https://www.xilinx.com/bin/public/openDownload?filename=resnet50-vck190-r3.0.0.tar.gz -O resnet50-vck190-r3.0.0.tar.gz

   b. Copy the downloaded file to the board using scp with the following command:

      .. code-block::

         scp resnet50-vck190-r3.0.0.tar.gz root@IP_OF_BOARD:~/

   c. Log in to the board (using ssh or serial port) and install the model package.

      .. code-block::

         tar -xzvf resnet50-vck190-r3.0.0.tar.gz
         cp resnet50 /usr/share/vitis_ai_library/models -r

Step 3: Run the Vitis AI Examples
----------------------------------

Follow :ref:`mpsoc-run-vitis-ai-examples` to run Vitis AI examples.

References
----------

-  `Vitis AI User Guide <https://www.xilinx.com/html_docs/vitis_ai/3_0/index.html>`__


.. |trade|  unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim:
.. |reg|    unicode:: U+000AE .. REGISTERED TRADEMARK SIGN
   :ltrim:
