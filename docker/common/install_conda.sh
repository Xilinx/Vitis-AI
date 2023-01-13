#!/bin/bash                                                                                                                                                                     
  
set -ex
if [[ ${DOCKER_TYPE} =~ .*'rocm'*  && ${TARGET_FRAMEWORK} =~ .*"pytorch"  ]]; then
   ln -s /opt/conda $VAI_ROOT/conda;

else

    export HOME=~vitis-ai-user
    if [[ -d "/root/.local" ]]; then
       sudo  chmod -R 777 /root/.local
    fi
    sudo chmod 777 /root /root/.local  /root/.local/bin || true 

cd /tmp \
    && wget --progress=dot:mega https://github.com/conda-forge/miniforge/releases/download/4.10.3-5/Mambaforge-4.10.3-5-Linux-x86_64.sh -O miniconda.sh \
    && /bin/bash ./miniconda.sh -b -p $VAI_ROOT/conda \
    && . $VAI_ROOT/conda/etc/profile.d/conda.sh \
    && conda install -y  conda-build \
    && rm -fr /tmp/miniconda.sh \
    &&  /$VAI_ROOT/conda/bin/conda clean -y --force-pkgs-dirs 
fi

echo ". $VAI_ROOT/conda/etc/profile.d/conda.sh" >> ~vitis-ai-user/.bashrc
sudo ln -s $VAI_ROOT/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
