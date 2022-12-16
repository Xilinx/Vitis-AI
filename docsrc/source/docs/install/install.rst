=================================================
Adaptable & Real-Time AI Inference Acceleration
=================================================

The purpose of this page is to provide the developer with guidance on the installation of Vitis |trade| AI tools on the development host PC. Instructions for installation of Vitis AI on the target is covered separately in :doc:`../board_setup/board_setup`.

Host / Development Software Installation
------------------------------------------

There are three primary options for installation:

-  [Option1] Directly leverage pre-built Docker containers available from Docker Hub:
   `xilinx/vitis-ai <https://hub.docker.com/r/xilinx/vitis-ai/tags>`__\ 
-  [Option2] Build a custom container to target your local host machine
-  [Option3] Install Vitis AI on AWS or Azure. See the instructions to install Vitis AI on :doc:`AWS <install_on_aws>`, or :doc:`Azure <install_on_azure>`

Pre-requisites
---------------

-  Confirm that your development machine meets the minimum :doc:`Host System Requirements <../reference/system_requirements>`
-  Confirm that you have at least **100GB** of free space in the target partition

Installation Guidance
---------------------

-  If Docker was not previously installed on your development machine with NVIDIA Container Toolkit or ROCm support, visit  :doc:`install_docker`

-  The Docker daemon always runs as the root user. Non-root users must be `added <https://docs.docker.com/engine/install/linux-postinstall/>`__ to the docker group. Do this now.

-  To get started, the developer should clone the Vitis-AI repository as follows:

   .. code-block:: bash

      git clone https://github.com/Xilinx/Vitis-AI
      cd Vitis-AI
   
   .. important:: The below installation commands are only relevant for the current release of Vitis AI. If your intention is to install a Docker for a historic release you must leverage one of previous release :doc:`Docker containers <../reference/docker_image_versions>`
   


Option 1: Leverage the Pre-Built Docker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As of the Vitis AI 3.0 release, pre-built Docker containers are framework specific.  Furthermore, we have extended support to include AMD ROCm enabled GPUs.  Users thus now have three options for a host: CPU-only, CUDA-capable GPUs, and ROCm-capable GPUs.  CUDA-capable GPUs are not supported by pre-built containers, and thus the developer must :ref:`build the container from scripts <build-docker-from-scripts>` 

To download the most up-to-date version of the pre-built docker, you will need execute the appropriate command, using the below general format:

::

   docker pull xilinx/vitis-ai-<Framework>-<Arch>:latest
 

Where ``<Framework>`` and ``<Arch>`` can be selected as in the table below:

.. list-table:: Vitis AI Pre-built Container Options
   :widths: 50 25 25
   :header-rows: 1

   * - Desired Framework
     - <Arch>
     - <Framework>
   * - TensorFlow 1.15 cpu-only
     - cpu
     - tf1
   * - TensorFlow 2 cpu-only
     - cpu
     - tf2
   * - PyTorch cpu-only
     - cpu
     - pytorch
   * - 
     - 
     - 
   * - TensorFlow 2 ROCm
     - rocm
     - tf2
   * - TF2 with AI Optimizer ROCm
     - rocm
     - opt_tf2
   * - PyTorch with AI Optimizer ROCm
     - rocm
     - opt_pytorch


*Specific Examples:*

	- Tensorflow 2 CPU docker : ``docker pull xilinx/vitis-ai-tensorflow2-cpu:latest``
	- Tensorflow 2 ROCm docker: ``docker pull xilinx/vitis-ai-tensorflow2-rocm:latest``
	- PyTorch ROCm docker: ``docker pull xilinx/vitis-ai-pytorch-rocm:latest``

.. important:: 

	Note that the ``cpu`` option *does not provide GPU acceleration support* which is **strongly recommended** for acceleration of the quantization process. The pre-built ``cpu`` container should only be used when a GPU is not available on the host machine.


	
Next, a quick and simple test of the docker container can be accomplished by executing the following command:

::

   docker run hello-world

This command downloads a test image and runs it in a container. When the container runs, it prints a message and exits.

Next, you can now start the Vitis AI Docker using the following command:

::

   <Vitis-AI install path>/Vitis-AI/docker_run.sh xilinx/vitis-ai-<cpu|rockm>-<tf1|tf2|opt_tf1|opt_tf2|pytorch|opt_pytorch>:latest


.. _build-docker-from-scripts:

Option 2: Build the Docker container from Xilinx Recipes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As of this release, a single unified docker build script has been provided.  This script enables developers to build a container for a specific framework.  This single unified script supports CPU-only hosts, GPU-capable hosts as well as the AMD ROCm-capable hosts.

In most cases, developers will want to leverage the GPU or ROCm enabled Dockers as they provide support for accelerated quantization and pruning. For NVIDIA graphics cards that meet Vitis AI CUDA requirements (:doc:`listed here <../reference/system_requirements>`) you can leverage the ``gpu`` Docker.

.. important:: 

   - If you are targeting Alveo and wish to enable X11 support, :doc:`script modifications <Alveo_X11>` are required. 
   - If you are building the Docker from within China, :doc:`script modifications <China_Ubuntu_servers>` are strongly recommended.

How to Build the Container
..........................

The command that you will use to build the container is of the following format: ``./docker_build.sh -t <DOCKER_TYPE> -f <FRAMEWORK>``

Where the supported build options are:

.. list-table:: Vitis AI Docker Container Build Options
   :widths: 20 30 50
   :header-rows: 1

   * - DOCKER_TYPE (-t)
     - TARGET_FRAMEWORK (-f)
     - Desired Environment
   * - cpu
     - tf1
     - TensorFlow 1.15 cpu-only
   * - 
     - tf2
     - TensorFlow 2 cpu-only
   * - 
     - pytorch
     - PyTorch cpu-only
   * - 
     - 
     - 
   * - gpu
     - tf1
     - TensorFlow 1.15 CUDA-gpu
   * - 
     - opt_tf1
     - TF1 with AI Optimizer CUDA-gpu
   * - 
     - tf2
     - TensorFlow 2
   * - 
     - opt_tf2
     - TF2 with AI Optimizer CUDA-gpu
   * - 
     - pytorch
     - PyTorch
   * - 
     - opt_pytorch
     - PyTorch with AI Optimizer CUDA-gpu
   * - 
     - 
     - 
   * - rocm
     - tf2
     - TensorFlow 2 ROCm-gpu
   * - 
     - opt_tf2
     - TF2 with AI Optimizer ROCm-gpu
   * - 
     - pytorch
     - PyTorch ROCm-gpu
   * - 
     - opt_pytorch
     - PyTorch with AI Optimizer ROCm-gpu
	 
.. important:: 

	Note that the ``cpu`` option *does not provide GPU acceleration support* which is **strongly recommended** for acceleration of the quantization process. The pre-built ``cpu`` container should only be used when a GPU is not available on the host machine.


As an example, the developer should use the following commands to build a Pytorch CUDA GPU docker with support for the Vitis AI Optimizer. Adjust your path to ``<Vitis-AI install path>/Vitis-AI/docker`` directory as necessary.

::

   cd <Vitis-AI install path>/Vitis-AI/docker
   ./docker_build.sh -t gpu -f opt_pytorch
   
You may also ``run docker_build.sh --help`` for additional information.

.. warning:: This process may take several hours to complete. It’s time to go off and get a coffee, tea, water or whatever suits your fancy. When you come back, assuming that the build is successful, move on to the steps below. If the build was unsuccessful, inspect the log output for specifics. In many cases, a specific package could not be located, most likely due to remote server connectivity. Often, simply re-running the build script will result in success. In the event that you continue to run into problems, please reach out for support.


Once the build script has completed, do an initial test of your docker using the following commands:

::

   sudo systemctl restart docker
   docker run hello-world

This command downloads a test image and runs it in a container. When the container runs, it prints a message and exits.

If the Docker has been enabled with CUDA-capable GPU support, confirm that the GPU is visible by executing the following command from within the container:

::

   docker run --gpus all nvidia/cuda:11.0-base nvidia-smi

This should result in an output similar to the below:

::

/Thu Dec  8 21:39:42 2022       
/+-----------------------------------------------------------------------------+
/| NVIDIA-SMI 470.161.03   Driver Version: 470.161.03   CUDA Version: 11.4     |
/|-------------------------------+----------------------+----------------------+
/| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
/| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
/|                               |                      |               MIG M. |
/|===============================+======================+======================|
/|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0 Off |                  N/A |
/|  0%   40C    P8     1W / 120W |     15MiB /  5944MiB |      0%      Default |
/|                               |                      |                  N/A |
/+-------------------------------+----------------------+----------------------+
/                                                                               
/+-----------------------------------------------------------------------------+
/| Processes:                                                                  |
/|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
/|        ID   ID                                                   Usage      |
/|=============================================================================|
/+-----------------------------------------------------------------------------+





.. note:: If CUDA GPU support was expected, but was not enabled in your container, check your NVIDIA driver version and CUDA version versus the :doc:`Host System Requirements <../reference/system_requirements>` and verify your installation of the NVIDIA Container Toolkit (:doc:`install_docker`). If you missed a step, you can simply rectify the problem and re-run ``docker_build.sh``.

You can now start the Docker for Vitis AI using the following command:

::

   ../docker_run.sh xilinx/vitis-ai-<gpu|cpu|rockm>-<tf1|tf2|opt_tf1|opt_tf2|pytorch|opt_pytorch>:latest


.. important:: Use ``./docker_run.sh`` as a script reference should you have customized requirements for launching your Docker container.

In most cases, you have now completed the installation. Congratulations!

If you have previously been instructed by your ML Specialist or FAE to leverage a specific patch for support of certain features, you should now follow the instructions :doc:`patch instructions <patch_instructions>` to complete your installation.


.. |trade|  unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim:
.. |reg|    unicode:: U+000AE .. REGISTERED TRADEMARK SIGN
   :ltrim:

