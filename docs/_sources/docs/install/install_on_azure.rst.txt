Installing Vitis AI on Microsoft Azure
======================================

The following steps will build a VM capable of running Vitis™ AI.

`Start an Azure VM <https://docs.microsoft.com/en-us/azure/virtual-machines/windows/quick-create-portal>`__ of type `NP10 <https://docs.microsoft.com/en-us/azure/virtual-machines/np-series>`__ using the Cannonical `Ubuntu 18.04 LTS <https://azuremarketplace.microsoft.com/en-us/marketplace/apps/canonical.0001-com-ubuntu-server-bionic?tab=overview>`__ app.

After starting this instance you must ssh to your cloud instance to complete the following steps:

1. Disable Kernel Auto-Upgrade

   .. code-block::

       sudo sed -i 's/1/0/g' /etc/apt/apt.conf.d/20auto-upgrades

2. Update Ubuntu Packages List, and Upgrade existing packages

   .. code-block::

        sudo apt-get update && sudo apt-get upgrade

3. Install Xilinx® XRT

   .. code-block::

        git clone https://github.com/Xilinx/XRT.git
        cd XRT
        git checkout 2021.1
        sudo ./src/runtime_src/tools/scripts/xrtdeps.sh
        cd build
        ./build.sh
        sudo apt install ./Release/*-xrt.deb
        sudo apt install ./Release/*-azure.deb
        cd

        # XCLMGMT Driver needs to be removed within the VM
        sudo modprobe -r xclmgmt
        sudo reboot

        # At this point, re-establish your ssh connection
        ssh ...

4. Install XRM

   .. code-block::

        wget https://www.xilinx.com/bin/public/openDownload?filename=xrm_202110.1.2.1539_18.04-x86_64.deb -O xrm.deb
        sudo apt install ./xrm.deb

5. Install the DPU Accelerator (FPGA Binary)

   .. code-block::

        wget https://www.xilinx.com/bin/public/openDownload?filename=dpu-azure-1.4.0.xclbin -O dpu-azure.xclbin
        sudo mkdir -p /opt/xilinx/overlaybins/DPUCADF8H
        sudo cp dpu-azure.xclbin /opt/xilinx/overlaybins/DPUCADF8H
        sudo chmod -R a+rx /opt/xilinx/overlaybins/DPUCADF8H

6. `Install Docker <https://docs.docker.com/engine/install/ubuntu/>`__

   .. code-block::

        curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
        sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
        sudo apt-get update
        sudo apt-get install docker-ce docker-ce-cli containerd.io
        sudo usermod -aG docker ubuntu
        # At this point, re-establish your ssh connection, so the ubuntu user can run docker commands
        exit
        ssh ...

7. Clone Vitis AI and Launch Docker Container

   .. code-block::

        git clone https://github.com/Xilinx/Vitis-AI.git
        cd Vitis-AI
        ./docker_run.sh xilinx/vitis-ai-cpu

`Run Examples <../../examples/DPUCADF8H/README.md>`__.
