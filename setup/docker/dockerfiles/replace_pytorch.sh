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

if [ -z "$1" ]; then
  echo "Usage: $0 new_conda_env_name"
  exit 2
fi

if [ -d "/usr/local/cuda" ]; then
  sudo apt update
  sudo apt-get install cuda-toolkit-10-0
  sudo rm /etc/alternatives/g++
  sudo rm /etc/alternatives/gcc
  sudo rm /etc/alternatives/gcov
  sudo ln -s /usr/bin/g++-7 /etc/alternatives/g++
  sudo ln -s /usr/bin/gcc-7 /etc/alternatives/gcc
  sudo ln -s /usr/bin/gcov-7 /etc/alternatives/gcov

  if [ $? -eq 0 ]; then
    echo -e "\n#### NVCC is installed successfully."
  else
    echo -e "\n#### NVCC is NOT installed successfully."
    exit 2
  fi
fi

eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"

echo -e "\n#### Creating a new conda environment by cloning vitis-ai-pytorch and activate it..."
conda create -n $1 --clone vitis-ai-pytorch 
if [ $? -eq 0 ]; then
  echo -e "\n#### New conda environment is created successfully."
else
  echo -e "\n#### New conda environment is NOT created correctly."
  exit 2
fi

conda activate $1
if [ $? -eq 0 ]; then
  echo -e "\n#### New conda environment is activated successfully."
else
  echo -e "\n#### New conda environment is NOT activated correctly."
  exit 2
fi

echo -e "\n#### Removing original pytorch packages ..."
pip uninstall -y torch torchvision  

echo -e "\n#### Removing original vai_q_pytorch ..."
pip uninstall  -y pytorch_nndct

echo -e "\n#### Installing pytorch 1.7.1 packages ..."
echo -e "\e[91m>>>> Edit this line of command to set the target version, refer to https://pytorch.org/get-started/previous-versions/ <<<<\e[m"
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 -f https://download.pytorch.org/whl/torch_stable.html
#pip install torch==1.6.0+cu101 torchvision==0.7.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
#pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html

if [ $? -eq 0 ]; then
  echo -e "\n#### Pytorch packages is replaced successfully."
else
  echo -e "\n#### Pytorch packages is NOT replaced correctly."
  exit 2
fi

echo -e "\n#### Checkout code of vai_q_pytorch ..."
echo -e "\e[91m>>>> You can apply your local code of vai_q_pytorch and comment out the following lines of git command <<<<\e[m"
git init code_vaiq && cd code_vaiq 
git config core.sparsecheckout true
echo 'tools/Vitis-AI-Quantizer/vai_q_pytorch/' >> .git/info/sparse-checkout 
git remote add origin https://github.com/Xilinx/Vitis-AI.git
git pull origin master
cd tools/Vitis-AI-Quantizer/vai_q_pytorch/

if [ $? -eq 0 ]; then
  echo -e "\n#### Vai_q_pytorch code is checked out successfully."
else
  echo -e "\n#### Vai_q_pytorch code is NOT checked out successfully."
  exit 2
fi

echo -e "\n#### Installing vai_q_pytorch ..."
pip install -r requirements.txt
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
