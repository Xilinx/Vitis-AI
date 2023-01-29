Host Installation Instructions
=================================================

The purpose of this page is to provide the developer with guidance on the installation of Vitis |trade| AI tools on the development host PC. Instructions for installation of Vitis AI on the target are covered separately in :doc:`../board_setup/board_setup`.

There are three primary options for installation:

**[Option1]** Directly leverage pre-built Docker containers available from Docker Hub: `xilinx/vitis-ai <https://hub.docker.com/r/xilinx/>`__.

**[Option2]** Build a custom container to target your local host machine.

**[Option3]** Install Vitis AI on AWS or Azure. See the instructions to install Vitis AI on :doc:`AWS <install_on_aws>`, or :doc:`Azure <install_on_azure>`.


In addition, Vitis AI supports three host types: 

	- CPU-only with no GPU acceleration
	- CUDA-capable GPUs
	- AMD ROCm |trade| GPUs
	
   ..  important::
       These installation instructions are only relevant for the current release of Vitis AI. If your intention is to pull a Docker container for a historic release you must leverage one of the previous release :doc:`Docker containers <../reference/docker_image_versions>`.
	   

Pre-requisites
---------------

-  Confirm that your development machine meets the minimum :doc:`Host System Requirements <../reference/system_requirements>`.
-  Confirm that you have at least **100GB** of free space in the target partition.


Preparing for the Installation
------------------------------

Refer to the relevant section (CPU-only, ROCm, CUDA) below to prepare your selected host for Docker installation.

CPU-only Host Initial Preparation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

CPU hosts require no special preparation.


ROCm GPU Host Initial Preparation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For ROCm hosts, developers should prepare their host by referring to the `ROCm Docker installation documentation <https://github.com/RadeonOpenCompute/ROCm-docker/blob/master/quick-start.md>`__.


CUDA GPU Host Initial Preparation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are leveraging a Vitis AI Docker Image with CUDA-capable GPU acceleration, you must install the NVIDIA Container Toolkit, which enables GPU support inside the Docker container. Please refer to the official NVIDIA `documentation <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`__ for additional information.

For Ubuntu distributions, NVIDIA driver and Container Toolkit installation can generally be accomplished as in the following example (use sudo for non-root users):

.. code-block::

     apt purge nvidia* libnvidia*
     apt install nvidia-driver-xxx
     apt install nvidia-container-toolkit

``xxx`` is the version of driver that you are choosing to install (i.e, ``nvidia-driver-510``), and is a version that meets Vitis AI :doc:`Host System Requirements <../reference/system_requirements>`.

A simple test to confirm driver installation is to execute ``nvidia-smi``.  This command can be used as an initial test outside of the Docker environment, and also can be used as a simple test inside of a Docker container following the installation of Docker and the Nvidia Container Toolkit.

.. code-block::

     nvidia-smi

The output should appear similar to the below, indicating the activation of the driver, and the successful installation of CUDA:

.. code-block::

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

Refer `NVIDIA driver installation <https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/index.html>`__ for further details of driver installation.

Docker Install and Verification
-------------------------------

Once you are confident that your host has been prepared according to the above guidance refer to official Docker `documentation <https://docs.docker.com/engine/install/>`__ to install the Docker engine.

   ..  important::
       The Docker daemon always runs as the root user. Non-root users must be `added <https://docs.docker.com/engine/install/linux-postinstall/>`__ to the docker group. Do this now.


Next, perform a quick and simple test of your Docker installation by executing the following command.  This command will download a test image from Docker Hub and run it in a container. When the container runs successfully, it prints a "Hello World" message and exits.


.. code-block::

	docker run hello-world


Finally, verify that the version of Docker that you have installed meets the minimum :doc:`Host System Requirements <../reference/system_requirements>` by running the following command:

.. code-block::

	docker --version


Clone The Repository
--------------------

If you have not already done so, you should now clone the Vitis AI repository to the host machine as follows:

..  code-block::
	bash

	git clone https://github.com/Xilinx/Vitis-AI
	cd Vitis-AI
	
	
Leverage Vitis AI Containers
----------------------------

You are now ready to start working with the Vitis AI Docker container.  At this stage you will choose whether you wish to use the pre-built container, or build the container from scripts.

Starting with the Vitis AI 3.0 release, pre-built Docker containers are framework specific.  Furthermore, we have extended support to include AMD ROCm enabled GPUs.
Users thus now have three options for the host Docker:
	- CPU-only
	- CUDA-capable GPUs
	- ROCm-capable GPUs

CUDA-capable GPUs are not supported by pre-built containers, and thus the developer must :ref:`build the container from scripts <build-docker-from-scripts>`.

Option 1: Leverage the Pre-Built Docker
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To download the most up-to-date version of the pre-built docker, you will need execute the appropriate command, using the following general format:

.. code-block::

    docker pull xilinx/vitis-ai-<Framework>-<Arch>:latest

Where ``<Framework>`` and ``<Arch>`` can be selected as in the table below:

.. list-table:: Vitis AI Pre-built Container Options
   :widths: 50 25 25
   :header-rows: 1

   * - Desired Docker
     - <Framework>
     - <Arch>
   * - PyTorch cpu-only
     - pytorch
     - cpu
   * - TensorFlow 2 cpu-only
     - tensorflow2
     - cpu
   * - TensorFlow 1.15 cpu-only
     - tensorflow
     - cpu
   * - PyTorch ROCm
     - pytorch
     - rocm
   * - TensorFlow 2 ROCm
     - tensorflow2
     - rocm
   * - PyTorch with AI Optimizer ROCm
     - opt-pytorch
     - rocm
   * - TF2 with AI Optimizer ROCm
     - opt-tensorflow2
     - rocm


*Specific Examples:*

	- PyTorch CPU-only docker: ``docker pull xilinx/vitis-ai-pytorch-cpu:latest``
	- PyTorch ROCm docker: ``docker pull xilinx/vitis-ai-pytorch-rocm:latest``
	- TensorFlow 2 CPU docker : ``docker pull xilinx/vitis-ai-tensorflow2-cpu:latest``
	- TensorFlow 2 ROCm docker: ``docker pull xilinx/vitis-ai-tensorflow2-rocm:latest``

.. important:: The ``cpu`` option *does not provide GPU acceleration support* which is **strongly recommended** for acceleration of the Vitis AI :ref:`Quantization process <quantization-process>`. The pre-built ``cpu`` container should only be used when a GPU is not available on the host machine.  The :ref:`AI Optimizer containers <model_optimization>` are only required for pruning and require a license.

Next, you can now start the Vitis AI Docker using the following commands:

.. code-block::

	cd <Vitis-AI install path>/Vitis-AI
	./docker_run.sh xilinx/vitis-ai-<pytorch|opt-pytorch|tensorflow2|opt-tensorflow2|tensorflow>-<cpu|rocm>:latest

    
.. _build-docker-from-scripts:

Option 2: Build the Docker Container from Xilinx Recipes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

As of this release, a single unified docker build script is provided.  This script enables developers to build a container for a specific framework.  This single unified script supports CPU-only hosts, GPU-capable hosts, and AMD ROCm-capable hosts.

In most cases, developers will want to leverage the GPU or ROCm-enabled Dockers as they provide support for accelerated quantization and pruning. For NVIDIA graphics cards that meet Vitis AI CUDA requirements (:doc:`listed here <../reference/system_requirements>`) you can leverage the ``gpu`` Docker.

.. important::

   - If you are targeting Alveo |trade| and wish to enable X11 support, :doc:`script modifications <Alveo_X11>` are required.
   - If you are building the Docker from within China, :doc:`script modifications <China_Ubuntu_servers>` are strongly recommended.

Navigate to the docker subdirectory in the Vitis AI install path:

.. code-block::

    cd <Vitis-AI install path>/Vitis-AI/docker

Here you will find the docker_build.sh script that will be used to build the container.  Execute the script as follows: ``./docker_build.sh -t <DOCKER_TYPE> -f <FRAMEWORK>``

The supported build options are:

.. list-table:: Vitis AI Docker Container Build Options
   :widths: 20 30 50
   :header-rows: 1

   * - DOCKER_TYPE (-t)
     - TARGET_FRAMEWORK (-f)
     - Desired Environment
   * - cpu
     - pytorch
     - PyTorch cpu-only
   * -
     - tf2
     - TensorFlow 2 cpu-only
   * - 
     - tf1
     - TensorFlow 1.15 cpu-only
   * -
     -
     -
   * - gpu
     - pytorch
     - PyTorch CUDA-gpu
   * -
     - opt_pytorch
     - PyTorch with AI Optimizer CUDA-gpu
   * -
     - tf2
     - TensorFlow 2 CUDA-gpu
   * -
     - opt_tf2
     - TensorFlow 2 with AI Optimizer CUDA-gpu
   * - 
     - tf1
     - TensorFlow 1.15 CUDA-gpu
   * -
     - opt_tf1
     - TensorFlow 1.15 with AI Optimizer CUDA-gpu
   * -
     -
     -
   * - rocm
     - pytorch
     - PyTorch ROCm-gpu
   * -
     - opt_pytorch
     - PyTorch with AI Optimizer ROCm-gpu
   * - 
     - tf2
     - TensorFlow 2 ROCm-gpu
   * -
     - opt_tf2
     - TensorFlow 2 with AI Optimizer ROCm-gpu

.. important:: The ``cpu`` option *does not provide GPU acceleration support* which is **strongly recommended** for acceleration of the Vitis AI :ref:`Quantization process <quantization-process>`. The pre-built ``cpu`` container should only be used when a GPU is not available on the host machine.  The :ref:`AI Optimizer containers <model_optimization>` are only required for pruning and require a license.

As an example, the developer should use the following commands to build a Pytorch CUDA GPU docker with support for the Vitis AI Optimizer. Adjust your path to ``<Vitis-AI install path>/Vitis-AI/docker`` directory as necessary.

.. code-block::

    cd <Vitis-AI install path>/Vitis-AI/docker
    ./docker_build.sh -t gpu -f opt_pytorch

You may also ``run docker_build.sh --help`` for additional information.

.. warning:: The ``docker_build`` process may take several hours to complete. Assuming the build is successful, move on to the steps below. If the build was unsuccessful, inspect the log output for specifics. In many cases, a specific package could not be located, most likely due to remote server connectivity. Often, simply re-running the build script will result in success. In the event that you continue to run into problems, please reach out for support.

If the Docker has been enabled with CUDA-capable GPU support, do a final test to ensure that the GPU is visible by executing the following command:

.. code-block::

   docker run --gpus all nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04 nvidia-smi

This should result in an output similar to the below:

.. code-block::

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

.. note:: If CUDA GPU support was expected but was not enabled, check your NVIDIA driver and CUDA versions versus the :doc:`Host System Requirements <../reference/system_requirements>` and verify your installation of the NVIDIA Container Toolkit. If you missed a step, you can rectify the problem and re-run ``docker_build.sh``.

You can now start the Docker for Vitis AI using the following command:

.. code-block::

	cd <Vitis-AI install path>/Vitis-AI
	./docker_run.sh xilinx/vitis-ai-<pytorch|opt-pytorch|tensorflow2|opt-tensorflow2|tensorflow>-<cpu|gpu|rocm>:latest

.. important:: Use ``./docker_run.sh`` as a script reference should you have customized requirements for launching your Docker container.

In most cases, you have now completed the installation. Congratulations!

If you have previously been instructed by your ML Specialist or FAE to leverage a specific patch for support of certain features, you should now follow the instructions :doc:`patch instructions <patch_instructions>` to complete your installation.

.. |trade|  unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim:
.. |reg|    unicode:: U+000AE .. REGISTERED TRADEMARK SIGN
   :ltrim:
