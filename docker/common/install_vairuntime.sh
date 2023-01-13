#!/bin/bash                                                                                                                                                                     
  
set -ex

if [[ "${XRT_URL}" =~ ".deb" ]]; then
   cd /tmp && wget --progress=dot:mega -O xrt.deb ${XRT_URL} \
    && apt-get update -y \
    && apt-get install -y ./xrt.deb
 
else
    # in case parsing xrt folder
   cd /tmp && wget --progress=dot:mega  -r -nd  --no-parent -A "*`lsb_release -r -s`*-xrt.deb"  ${XRT_URL} \
    && apt-get update -y \
    && apt-get install -y ./*xrt*.deb

   # XRT_URL="${XRT_URL}/*`lsb_release -r -s`*-xrt.deb"
     #-r -nd  --no-parent -A "*18.04-amd64-xrt.deb" 
fi
if [[ $? -ne 0 ]];then
        exit 1
fi

if [[ "${XRM_URL}" =~ ".deb" ]]; then
    wget --progress=dot:mega -O xrm.deb ${XRM_URL} \
      && apt-get install -y ./xrm.deb \
      && rm -fr /tmp/*

else    # in case parsing xrt folder
   wget --progress=dot:mega   -r -nd  --no-parent -A "*`lsb_release -r -s`*x86_64.deb"   ${XRM_URL} \
   && apt-get install -y ./*xrm*.deb \
   && rm -fr /tmp/*

fi

if [[ $? -ne 0 ]];then                                                                                                                                                              
        exit 1
fi

if [[ "${VAI_DEB_CHANNEL}" =~ ".tar.gz" ]]; then
#download link
    cd /tmp/ && wget --progress=dot:mega -O vairuntime.tar.gz ${VAI_DEB_CHANNEL} \
     &&  tar xvf vairuntime.tar.gz \
     && apt-get install -y ./*/*.deb \
     && ldconfig \
     && rm -fr /tmp/*

else
# repo 
   mkdir -p /etc/apt/sources.list.d \
    && echo "deb [trusted=yes] ${VAI_DEB_CHANNEL} focal main" |tee /etc/apt/sources.list.d/xlnx.list \
    && cat /etc/apt/sources.list.d/xlnx.list \
    && apt-get update -y \
    && apt-get install -y \
        libunilog \
        libtarget-factory \
        libxir \
        libvart \
        libvitis_ai_library \
        librt-engine \
        aks \
    && rm -rf /etc/apt/sources.list.d/xlnx.list \
    && ldconfig \
    && cd /usr/lib && mv  libvart-dpu-runner.so  libvart-dpu-runner.so_org &&  ln -s librt-engine.so libvart-dpu-runner.so
fi

if [[ $? -ne 0 ]];then                                                                                                                                                              
        exit 1
fi

