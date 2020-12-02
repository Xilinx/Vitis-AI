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
echo -e "For PyTorch Workflows do:\n  conda activate vitis-ai-pytorch"
echo -e "For TensorFlow 2.3 Workflows do:\n  conda activate vitis-ai-tensorflow2"


darknet="`conda env list |grep darknet`"
if [ X"$darknet" != "X" ];then
   echo -e "For Darknet Optimizer Workflows do:\n  conda activate vitis-ai-optimizer_darknet"
fi

caffe_opt="`conda env list |grep optimizer_caffe`"
if [ X"$caffe_opt" != "X" ];then
   echo -e "For Caffe Optimizer Workflows do:\n  conda activate vitis-ai-optimizer_caffe"
fi
optimizer_tensorflow="`conda env list |grep optimizer_tensorflow`"
if [ X"$optimizer_tensorflow" != "X" ];then
   echo -e "For TensorFlow 1.15 Workflows do:\n  conda activate vitis-ai-optimizer_tensorflow"
fi

lstm="`conda env list |grep -i lstm`"
if [ X"$lstm" != "X" ];then
   echo -e "For LSTM Workflows do:\n  conda activate vitis-ai-lstm"
fi
