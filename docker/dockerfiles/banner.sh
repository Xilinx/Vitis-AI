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
echo -e "Docker Image Version:\e[32m $VERSION \e[m \e[31m $DOCKER_TYPE \e[m"
echo -e "Vitis AI Git Hash:\e[32m $GIT_HASH \e[m"
echo "Build Date: $BUILD_DATE"
echo -e ""
echo -e "For TensorFlow 1.15 Workflows do:"
echo -e "    \e[31m conda activate vitis-ai-tensorflow \e[m"
echo -e "For PyTorch Workflows do:"
echo -e "    \e[31m conda activate vitis-ai-pytorch \e[m"
echo -e "For TensorFlow 2.8 Workflows do:"
echo -e "    \e[31m conda activate vitis-ai-tensorflow2 \e[m"


darknet="`conda env list |grep darknet`"
if [ X"$darknet" != "X" ];then
   echo -e "For Darknet Optimizer Workflows do:"
   echo -e "    \e[31m conda activate vitis-ai-optimizer_darknet \e[m"
fi

optimizer_pytorch="`conda env list |grep optimizer_pytorch`"
if [ X"$optimizer_pytorch" != "X" ];then
   echo -e "For PyTorch Optimizer Workflows do:"
   echo -e "    \e[31m conda activate vitis-ai-optimizer_pytorch \e[m"
fi
optimizer_tensorflow="`conda env list |grep optimizer_tensorflow`"
if [ X"$optimizer_tensorflow" != "X" ];then
   echo -e "For TensorFlow 1.15 Optimizer Workflows do:"
   echo -e "    \e[31m conda activate vitis-ai-optimizer_tensorflow \e[m"
fi
optimizer_tensorflow2="`conda env list |grep optimizer_tensorflow2`"
if [ X"$optimizer_tensorflow2" != "X" ];then
   echo -e "For TensorFlow 2.8 Optimizer Workflows do:"
   echo -e "    \e[31m conda activate vitis-ai-optimizer_tensorflow2 \e[m"
fi


wego_tf1="`conda env list |grep vitis-ai-wego-tf1`"
if [ X"$wego_tf1" != "X" ];then
   echo -e "For WeGo Tensorflow 1.15 Workflows do:"
   echo -e "    \e[31m conda activate vitis-ai-wego-tf1 \e[m"
fi

wego_tf2="`conda env list |grep vitis-ai-wego-tf2`"
if [ X"$wego_tf2" != "X" ];then
   echo -e "For WeGo Tensorflow 2.8 Workflows do:"
   echo -e "    \e[31m conda activate vitis-ai-wego-tf2 \e[m"
fi   

wego_torch="`conda env list |grep vitis-ai-wego-torch`"
if [ X"$wego_torch" != "X" ];then
   echo -e "For WeGo Torch Workflows do:"
   echo -e "    \e[31m conda activate vitis-ai-wego-torch \e[m"
fi   

