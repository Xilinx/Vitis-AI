#
# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#!/usr/bin/env bash
NNDCT_REPO="${NNDCT_REPO:-https://github.com/Xilinx/Vitis-AI.git}"
NNDCT_SRC="${NNDCT_SRC:-src/vai_quantizer/vai_q_pytorch}"
BRANCH="${BRANCH:-master}"

valid_torch_version=("1.4" "1.7.1" "1.8.0" "1.12.1" "2.0.0")
if [ -z "$1" ]; then
  echo "Usage: $0 torch_version"
  echo "  i.e: 1.4 "
  echo "       1.4,1,7.1"
  echo "  current valid torch version supported by this scripts are: ${valid_torch_version[*]} "
  echo "  if the version is not included, pleaase update this script in pip install section by refer to pytorch version history"
  exit 2
fi
IFS=',' read -r -a torch_version <<< "$1"


for torch_input  in ${torch_version[@]}; do
   if [[ ! " ${valid_torch_version[*]} " =~ " ${torch_input} " ]]; then
      echo "${torch_input} is not currently support version in this script, please update the script first to support this version"
      echo "Current support torch versions are:" 
      echo "${valid_torch_version[*]}"
      echo "if the version is not included, pleaase update this script in pip install section by refer to pytorch version history"
      exit 2
   fi
done


echo -e "\n#### Creating a new conda environment by cloning vitis-ai-pytorch and activate it..."
sudo chmod 777 /opt/vitis_ai/conda 
cd /scratch/
wget -O conda-channel.tar.gz --progress=dot:mega https://www.xilinx.com/bin/public/openDownload?filename=conda-channel-3.5.0.tar.gz
tar -xzvf conda-channel.tar.gz
source /opt/vitis_ai/conda/etc/profile.d/conda.sh
sudo conda env export -n vitis-ai-pytorch >/tmp/pytorch.yaml
sed -i '/artifactory/d' /tmp/pytorch.yaml
sed -i '/prefix/d' /tmp/pytorch.yaml
sed -i '/torchvision/d' /tmp/pytorch.yaml
sed -i 's/python-graphviz/graphviz/g' /tmp/pytorch.yaml
if [ -f "/opt/vitis_ai/conda/.condarc" ]; then
   rm /opt/vitis_ai/conda/.condarc -f
fi

echo -e "\n#### Checkout code of vai_q_pytorch ..."
echo -e "\e[91m>>>> You can apply your local code of vai_q_pytorch and comment out the following lines of git command <<<<\e[m"

git init code_vaiq && cd code_vaiq
git config core.sparsecheckout true
echo 'src/vai_quantizer/vai_q_pytorch' >> .git/info/sparse-checkout 
git remote add origin $NNDCT_REPO
git pull origin $BRANCH
if [ $? -eq 0 ]; then
  echo -e "\n#### Vai_q_pytorch code is checked out successfully."
else
  echo -e "\n#### Vai_q_pytorch code is NOT checked out successfully."
  exit 2
fi


conda config --env --append channels file:///scratch/conda-channel
for torch_to_install  in ${torch_version[@]}; do
  conda env create  -n vitis-ai-pytorch-$torch_to_install -f /tmp/pytorch.yaml -v

  if [ $? -eq 0 ]; then
    echo -e "\n#### vitis-ai-pytorch-$torch_to_install environment is created successfully."
  else
    echo -e "\n#### New conda environment is NOT created correctly."
    exit 2
  fi

  conda activate  vitis-ai-pytorch-$torch_to_install
  if [ $? -eq 0 ]; then
    echo -e "\n#### vitis-ai-pytorch-$torch_to_install is activated successfully."
  else
    echo -e "\n#### vitis-ai-pytorch-$torch_to_install is NOT activated correctly."
    exit 2
  fi

  echo -e "\n#### Removing original pytorch related packages ..."
  mamba uninstall -y pytorch pytorch_nndct
  pip uninstall torchvision

  echo -e "\n#### Installing target pytorch packages ..."
  echo -e "\e[91m>>>> Edit this line of command to set the target version, refer to https://pytorch.org/get-started/previous-versions/ <<<<\e[m"
#### Installing pytorch 1.7.1 packages ...torchvision==0.5.0+cu100 -f https://download.pytorch.org/whl/torch_stable.html
  if [ -d "/usr/local/cuda" ]; then
    
    #mamba install -y pytorch==$torch_to_install  -c pytorch
    #pip  install torchvision==0.8.2
   install_version=""
   if [ $torch_to_install == "1.4" ];then
     install_version="torch==1.4.0 torchvision==0.5.0"
   elif [ $torch_to_install == "1.7.1" ];then
     install_version="torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html"
   elif [ $torch_to_install == "1.8.0" ];then
     install_version="torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html"
   elif [ $torch_to_install == "1.12.1" ];then
     install_version="torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116"
   elif [ $torch_to_install == "2.0.0" ];then
     install_version="torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118" 
	 #"torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117"
   else
     echo -e "\e[91m>>>> please provide the detailed torch version install list in the script, refer to the other torch versions <<<<\e[m"
     exit 2
   fi
   pip install $install_version
  else
   #mamba install -y pytorch==$torch_to_install cpuonly -c pytorch
    #pip install torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
   install_version=""
   if [ $torch_to_install == "1.4" ];then
	   install_version="torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html"
   elif [ $torch_to_install == "1.7.1" ];then
     install_verscon="torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html"
   elif [ $torch_to_install == "1.8.0" ];then
     install_version="torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html"
   elif [ $torch_to_install == "1.12.1" ];then
     install_version="torch==1.12.1+cpu torchvision==0.13.1+cpu torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cpu"
   elif [ $torch_to_install == "2.0.0" ];then
     install_version="torch==2.0.0+cpu torchvision==0.15.1+cpu torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cpu"
   else
     echo "please provide the detailed torch version install list in the script, refer to the other torch versions "
     exit 2
   fi
   pip install $install_version
  fi

  if [ $? -eq 0 ]; then
    echo -e "\n#### Pytorch packages is replaced successfully."
  else
    echo -e "\n#### Pytorch packages is NOT replaced correctly."
    exit 2
  fi

  cd /scratch/code_vaiq/$NNDCT_SRC 

  if [ $? -eq 0 ]; then
    echo -e "\n#### Vai_q_pytorch code is checked out successfully."
  else
    echo -e "\n#### Vai_q_pytorch code is NOT checked out successfully."
    exit 2
  fi

  echo -e "\n#### Installing vai_q_pytorch ..."
#pip install -r requirements.txt
  cd pytorch_binding 
  if [ ! -d "/usr/local/cuda" ]; then
    unset CUDA_HOME
  fi
    python setup.py bdist_wheel -d ./
    pip install ./pytorch_nndct-*.whl
  if [ $? -eq 0 ]; then
    echo -e "\n#### Vai_q_pytorch is compiled and installed successfully."
  else
    echo -e "\n#### Vai_q_pytorch is NOT compiled and installed successfully."
    exit 2
  fi

  mamba install -y python=3.8 --force-reinstall -c conda-forge 
  pip install --force-reinstall scipy protobuf==3.20.3
done
  sudo rm -fr /scratch/*
  echo -e "\n#### Cleaned up /scratch ."
