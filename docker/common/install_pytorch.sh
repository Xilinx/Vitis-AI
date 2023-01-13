#!/bin/bash                                                                                                                                                                     
  
set -ex
sudo chmod 777 /scratch

if [[ ${DOCKER_TYPE} == 'cpu' ]]; then
    # will have two conda env torch and wego-torch
    ./install_torch.sh &&
    ./install_wego_torch.sh
else
    ./install_torch.sh
fi
