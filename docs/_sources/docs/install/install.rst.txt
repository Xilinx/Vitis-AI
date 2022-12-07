=================================================
Adaptable & Real-Time AI Inference Acceleration
=================================================

The purpose of this page is to provide the developer with guidance on the installation of Vitis |trade| AI tools on the development host PC. Instructions for installation of Vitis AI on the target is covered separately in :doc:`../board_setup/board_setup`.

Host / Development Software Installation
------------------------------------------

There are three primary options for installation:

-  [Option1] Directly leverage pre-built Docker containers available from Docker Hub:
   `xilinx/vitis-ai <https://hub.docker.com/r/xilinx/vitis-ai/tags>`__\ 
-  [Option2] Build a custom container locally using the Vitis AI :doc:`Board Setup Documentation <../board_setup/board_setup>`
-  [Option3] Install Vitis AI on AWS or Azure. See the instructions to install Vitis AI on :doc:`AWS <install_on_aws>`, or :doc:`Azure <install_on_azure>`

Pre-requisites
---------------

-  Confirm that your development machine meets the minimum :doc:`Host System Requirements <../reference/system_requirements>`
-  Confirm that you have at least **100GB** of free space in the target partition

Installation Steps
------------------

-  If Docker was not previously installed on your development machine with NVIDIA Container Toolkit support,visit  :doc:`install_docker`

-  The Docker daemon always runs as the root user. Non-root users must be `added <https://docs.docker.com/engine/install/linux-postinstall/>`__ to the docker group. Do this now.

-  Clone the Vitis-AI repository to obtain the examples, reference code, and scripts 

   .. code-block:: bash

      git clone https://github.com/Xilinx/Vitis-AI
      cd Vitis-AI
   
   .. important:: The below installation commands are only relevant for the current release of Vitis AI. If your intention is to install a Docker for a historic release you must leverage one of previous release :doc:`Docker containers <../reference/docker_image_versions>`
   


Option 1: Leverage the pre-built Docker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Note that this option *does not provide GPU acceleration support* which is **strongly recommend** for acceleration of the quantization process. The pre-built Docker should only be used when a GPU is not available on the host machine.

To download the most up-to-date version of the pre-built docker, execute the following command:

::

   docker pull xilinx/vitis-ai-cpu:latest  

A quick and simple test of docker can be accomplished by executing the following command:

::

   docker run hello-world

This command downloads a test image and runs it in a container. When the container runs, it prints a message and exits.

You can now start the Docker for Vitis AI using the following command:

::

   ../../docker_run.sh xilinx/vitis-ai-cpu:latest



Option 2: Build the Docker container from Xilinx Recipes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In most cases, Developers will want to leverage the GPU-enabled Docker as it provides support for accelerated quantization. If you have a compatible NVIDIA graphics card that meets Vitis AI CUDA requirements (:doc:`listed here <../reference/system_requirements>`), you should leverage the GPU-capable Docker.

We have also provided the CPU recipe should customization of the CPU-only Docker be required.

.. important:: 

   - If you are targeting Alveo and wish to enable X11 support, :doc:`script modifications <Alveo_X11>` are required. 
   - If you are building the Docker from within China, :doc:`script modifications <China_Ubuntu_servers>` are strongly recommended.

GPU Docker
..........

Use the following commands to build the GPU docker. Adjust your path to the ``Vitis-AI/docker`` directory as necessary.

::

   cd ../../docker
   ./docker_build_gpu.sh

.. warning:: This process may take several hours to complete. It’s time to go off and get a coffee, tea, water or whatever suits your fancy. When you come back, assuming that the build is successful, move on to the steps below. If the build was unsuccessful, inspect the log output for specifics. In many cases, a specific package could not be located, most likely due to remote server connectivity. Often, simply re-running the build script will result in success. In the event that you continue to run into problems, please reach out for support.

You should now do an initial test of your GPU docker using the following commands:

::

   docker run hello-world

This command downloads a test image and runs it in a container. When the container runs, it prints a message and exits.

Now, test that the Docker has been enabled with GPU support by executing the following command from within the container:

::

   docker run --gpus all nvidia/cuda:11.0-base nvidia-smi

This should result in an output similar to the below:

::

   ----need to fill this in---

.. note:: If GPU support was not enabled in your container, check your NVIDIA driver version and CUDA version described in :doc:`Host System Requirements <../reference/system_requirements>` and verify your installation of the NVIDIA Container Toolkit (:doc:`install_docker`). If you missed a step, you can simply rectify the problem and re-run ``docker_build_gpu.sh``.

You can now start the Docker for Vitis AI using the following command:

::

   ../../docker_run.sh xilinx/vitis-ai-gpu:latest

.. important:: Use ``./docker_run.sh`` as a code reference should you have customized requirements for launching your Docker container.

CPU Docker
...........

Use the following commands to build the CPU docker. Adjust your path to the ``Vitis-AI/docker`` directory as necessary.

::

   cd ../../docker
   ./docker_build_cpu.sh

A quick and simple test of docker can be accomplished by executing the following command:

::

   docker run hello-world

This command downloads a test image and runs it in a container. When the container runs, it prints a message and exits.

You can now start the Docker for Vitis AI using the following command:

::

   ../../docker_run.sh xilinx/vitis-ai-cpu:latest


In most cases, you have now completed the installation. Congratulations!

If you have previously been instructed by your ML Specialist or FAE to leverage a specific patch for support of certain features, you should now follow the instructions :doc:`patch instructions <patch_instructions>` to complete your installation.


.. |trade|  unicode:: U+02122 .. TRADEMARK SIGN
   :ltrim:
.. |reg|    unicode:: U+000AE .. REGISTERED TRADEMARK SIGN
   :ltrim:

