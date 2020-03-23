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
