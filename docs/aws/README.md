# Vitis-AI on Amazon AWS
The following steps will guide you on how to build up an AMI capable of running Vitis-AI

## Steps

- [Start an AWS EC2 instance](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/launch-marketplace-console.html) of type [f1.2xlarge](https://aws.amazon.com/ec2/instance-types/f1/)  using the Cannonical [Ubuntu 18.04 LTS](https://aws.amazon.com/marketplace/pp/Canonical-Group-Limited-Ubuntu-1804-LTS-Bionic/B07CQ33QKV) AMI
```
After starting this instance you must ssh to your cloud instance to complete the following steps
```

- Disable Kernel Auto-Upgrade
```
sudo sed -i 's/1/0/g' /etc/apt/apt.conf.d/20auto-upgrades
```

- Update Ubuntu Packages List, and Upgrade existing packages
```
sudo apt-get update && sudo apt-get upgrade
```

- Install AWS FPGA Management Library and Xilinx XRT
```
git clone https://github.com/Xilinx/XRT.git
cd XRT 
git checkout 2021.1
sudo ./src/runtime_src/tools/scripts/xrtdeps.sh

cd

git clone https://github.com/aws/aws-fpga.git
cd aws-fpga
source sdk_setup.sh

cd

cd XRT/build
./build.sh
sudo apt install ./Release/*-xrt.deb
sudo apt install ./Release/*-aws.deb
cd
```

- Install XRM
```
wget https://www.xilinx.com/bin/public/openDownload?filename=xrm_202110.1.2.1539_18.04-x86_64.deb -O xrm.deb
sudo apt install ./xrm.deb
```

- Install the DPU Accelerator (FPGA Binary)
```
wget https://www.xilinx.com/bin/public/openDownload?filename=dpu-aws-1.4.0.xclbin -O dpu-aws.xclbin
sudo mkdir -p /opt/xilinx/overlaybins/DPUCADF8H
sudo cp dpu-aws.xclbin /opt/xilinx/overlaybins/DPUCADF8H
sudo chmod -R a+rx /opt/xilinx/overlaybins/DPUCADF8H
```

- [Install Docker](https://docs.docker.com/engine/install/ubuntu/)
```
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io
sudo usermod -aG docker ubuntu
# At this point, re-establish your ssh connection, so the ubuntu user can run docker commands
exit
ssh ...
```

- Clone Vitis-AI and Launch Docker Container
```
git clone https://github.com/Xilinx/Vitis-AI.git
cd Vitis-AI
./docker_run.sh xilinx/vitis-ai-cpu
```

- [Run Examples](../../examples/DPUCADF8H/README.md)
