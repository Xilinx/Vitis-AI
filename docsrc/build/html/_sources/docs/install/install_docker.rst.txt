Installing Docker
=================

.. important:: In most cases, Developers will want to leverage the CUDA-capable or ROCm Dockers as they support accelerated quantization. Before installing Docker for CUDA-capable GPUs, ensure that you understand the NVIDIA driver, CUDA :doc:`Host System Requirements <../reference/system_requirements>` for Vitis AI.

.. note:: For ROCm distributions, developers should reference `ROCm docker installation <https://github.com/RadeonOpenCompute/ROCm-docker/blob/master/quick-start.md>`__ for further details of docker installation.

Installing NVIDIA Container Toolkit
-----------------------------------

If you are building the Vitis AI Docker Image with CUDA-capable GPU acceleration, you must install the NVIDIA Container Toolkit, which enables GPU support inside the Docker container. Please refer to the official NVIDIA `documentation <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`__ for additional information.

For Ubuntu distributions, NVIDIA driver and Container Toolkit installation can generally be accomplished as displayed in the following example (use sudo for non-root users):

.. code-block::

     apt purge nvidia* libnvidia*
     apt install nvidia-driver-xxx
     apt install nvidia-container-toolkit

Where xxx is the version of driver that you are choosing to install (i.e, *nvidia-driver-510*), and is a version that meets Vitis AI :doc:`Host System Requirements <../reference/system_requirements>`.

A simple test to confirm driver installation is to run nvidia-smi:

.. code-block::

     nvidia-smi

The output should appear similar to this:

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

Docker Install
--------------

Once you are confident that your system meets any pre-requisites for Vitis AI Docker CUDA or ROCm GPU support, refer to official Docker `documentation <https://docs.docker.com/engine/install/>`__ to install the Docker engine.
