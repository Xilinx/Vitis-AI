echo -e "\e[1;90m
==========================================
\e[m \e[1;91m
__      ___ _   _                   _____
\ \    / (_) | (_)            /\   |_   _|
 \ \  / / _| |_ _ ___ ______ /  \    | |
  \ \/ / | | __| / __|______/ /\ \   | |
   \  /  | | |_| \__ \     / ____ \ _| |_
    \/   |_|\__|_|___/    /_/    \_\_____|
\e[m \e[1;90m
==========================================
\e[m"
echo -e "Docker Image Version:\e[32m $VERSION \e[m \e[31m (${DOCKER_TYPE^^}) \e[m"
echo -e "Vitis AI Git Hash:\e[32m $GIT_HASH \e[m"
echo "Build Date: $BUILD_DATE"
echo "WorkFlow: $TARGET_FRAMEWORK"
echo -e ""

tensorflow2="`conda env list |grep vitis-ai-tensorflow2`"
if [ X"$tensorflow2" != "X" ];then
   conda activate vitis-ai-tensorflow2
fi

tensorflow="`conda env list |grep vitis-ai-tensorflow|grep -v tensorflow2`"
if [ X"$tensorflow" != "X" ];then
   conda activate vitis-ai-tensorflow
fi

pytorch="`conda env list |grep vitis-ai-pytorch`"
if [ X"$pytorch" != "X" ];then
   conda activate vitis-ai-pytorch
fi

optimizer_pytorch="`conda env list |grep optimizer_pytorch`"
if [ X"$optimizer_pytorch" != "X" ];then
   conda activate vitis-ai-optimizer_pytorch
fi
optimizer_tensorflow2="`conda env list |grep optimizer_tensorflow2`"
if [ X"$optimizer_tensorflow2" != "X" ];then
   conda activate vitis-ai-optimizer_tensorflow2
fi
optimizer_tensorflow="`conda env list |grep optimizer_tensorflow|grep -v optimizer_tensorflow2`"
if [ X"$optimizer_tensorflow" != "X" ];then
   conda activate vitis-ai-optimizer_tensorflow 
fi


wego_tf1="`conda env list |grep vitis-ai-wego-tf1`"
if [ X"$wego_tf1" != "X" ];then
   conda activate vitis-ai-wego-tf1
fi

wego_tf2="`conda env list |grep vitis-ai-wego-tf2`"
if [ X"$wego_tf2" != "X" ];then
   conda activate vitis-ai-wego-tf2
fi   

wego_torch="`conda env list |grep vitis-ai-wego-torch`"
if [ X"$wego_torch" != "X" ];then
   conda activate vitis-ai-wego-torch
fi   

