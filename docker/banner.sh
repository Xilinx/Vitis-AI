echo "
==========================================
__      ___ _   _                   _____ 
\ \    / (_) | (_)            /\   |_   _|
 \ \  / / _| |_ _ ___ ______ /  \    | |  
  \ \/ / | | __| / __|______/ /\ \   | |  
   \  /  | | |_| \__ \     / ____ \ _| |_ 
    \/   |_|\__|_|___/    /_/    \_\_____|

==========================================
"
echo "Docker Image Version: $VERSION"
echo "Build Date: $DATE"
echo "VAI_ROOT=$VAI_ROOT"
echo -e "For TensorFlow Workflows do:\n  conda activate vitis-ai-tensorflow"
echo -e "For Caffe Workflows do:\n  conda activate vitis-ai-caffe"
echo -e "For Neptune Workflows do:\n  conda activate vitis-ai-neptune"

nndct="`conda env list |grep pytorch`"
if [ X"$nndct" != "X" ];then
   echo -e "For pytorch Workflows do:\n  conda activate vitis-ai-pytorch"
fi

darknet="`conda env list |grep darknet`"
if [ X"$darknet" != "X" ];then
   echo -e "For optimizer_darknet Workflows do:\n  conda activate vitis-ai-optimizer_darknet"
fi

caffe_opt="`conda env list |grep optimizer_caffe`"
if [ X"$caffe_opt" != "X" ];then
   echo -e "For optimizer_caffe Workflows do:\n  conda activate vitis-ai-optimizer_caffe"
fi
optimizer_tensorflow="`conda env list |grep optimizer_tensorflow`"
if [ X"$optimizer_tensorflow" != "X" ];then
   echo -e "For optimizer_tensorflow Workflows do:\n  conda activate vitis-ai-optimizer_tensorflow"
fi
